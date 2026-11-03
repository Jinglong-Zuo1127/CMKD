from .. import basic_balancer
from .. import balancers
import torch.nn.functional as F
from CLIP import clip
import torch

import torch.nn as nn
# from sklearn.manifold import TSNE
from dist_zoo.RKD import RKDLoss
from dist_zoo.AB import ABLoss
from dist_zoo.AT import Attention
from dist_zoo.FitNet import HintLoss
from dist_zoo.CC import Correlation
from dist_zoo.SP import Similarity
from dist_zoo.FSP import FSP
from dist_zoo.FT import FactorTransfer
from dist_zoo.KDSVD import KDSVD
from dist_zoo.VID import VIDLoss
from dist_zoo.NST import NSTLoss
from dist_zoo.PKT import PKT

import numpy as np
import json
import os
def extract_task_gradient(model, task_loss, param_filter):
    """
    提取单个任务对共享参数的梯度
    Args:
        model: 模型实例（支持DataParallel）
        task_loss: 单个任务的损失值
        param_filter: 筛选共享参数的函数（返回bool）
    Returns:
        task_grad: 该任务的梯度向量（展平后的numpy数组）
    """
    # 清空原有梯度
    model.zero_grad(set_to_none=True)
    # 仅对当前任务损失求导
    task_loss.backward(retain_graph=True)

    grad_list = []
    # 遍历模型参数（适配DataParallel）
    for name, param in model.named_parameters():
        if param_filter(name, param):
            if param.grad is not None:
                # 展平梯度并转为numpy
                grad_flat = param.grad.detach().cpu().numpy().flatten()
                grad_list.append(grad_flat)

    # 拼接所有共享参数的梯度为一维向量
    task_grad = np.concatenate(grad_list) if grad_list else np.array([])
    return task_grad


def compute_gradient_conflict(task_grad_dict):
    """
    计算任务间梯度冲突指标（新增：标准化任务对命名）
    Args:
        task_grad_dict: {task_id: 梯度向量}的字典
    Returns:
        conflict_metrics: 包含余弦相似度、冲突分数等的字典
    """
    task_ids = sorted(list(task_grad_dict.keys()))  # 排序任务ID，保证任务对顺序统一
    n_tasks = len(task_ids)
    if n_tasks < 2:
        return {
            "avg_cos_sim": 1.0,
            "conflict_score": 0.0,
            "cos_sim": {},
            "grad_norm": {tid: np.linalg.norm(grad) for tid, grad in task_grad_dict.items()}
        }

    # 1. 计算所有任务对的余弦相似度（任务ID排序，避免L-C/C-L两种键）
    cos_sim_dict = {}
    cos_sim_list = []
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            tid1, tid2 = task_ids[i], task_ids[j]
            grad1, grad2 = task_grad_dict[tid1], task_grad_dict[tid2]

            # 计算余弦相似度（避免除零）
            norm1 = np.linalg.norm(grad1)
            norm2 = np.linalg.norm(grad2)
            if norm1 == 0 or norm2 == 0:
                cos_sim = 0.0
            else:
                cos_sim = np.dot(grad1, grad2) / (norm1 * norm2)

            cos_sim_key = f"{tid1}-{tid2}"  # 排序后任务对，键唯一
            cos_sim_dict[cos_sim_key] = cos_sim
            # print(f"{tid1}, {tid2}: {cos_sim:.4f}")
            cos_sim_list.append(cos_sim)

    # 2. 计算平均余弦相似度和冲突分数（冲突分数=1-平均余弦相似度）
    avg_cos_sim = np.mean(cos_sim_list)
    conflict_score = 1.0 - avg_cos_sim

    # 3. 计算各任务梯度的L2范数
    grad_norm_dict = {tid: np.linalg.norm(grad) for tid, grad in task_grad_dict.items()}

    return {
        "avg_cos_sim": float(avg_cos_sim),
        "conflict_score": float(conflict_score),
        "cos_sim": cos_sim_dict,
        "grad_norm": grad_norm_dict
    }
class ClipTeacherModel(torch.nn.Module):
    """CLIP教师模型"""

    def __init__(self, device='cuda', model_name='ViT-B/16'):
        super().__init__()
        self.device = device
        self.model, _ = clip.load(model_name, device=device)
        self.class_names = ["bird", "car", "chair", "dog", "person"]
        self.text_inputs = torch.cat([
            clip.tokenize(f"a photo of a {c}") for c in self.class_names
        ]).to(device)
        self.model = self.model.float()
        self.model.eval()  # 固定为评估模式

        # 存储选定层特征
        self.hooks = []
        self.intermediate_features = {}
        self.register_layers([8,9,10,11])
    def _get_hook(self, name):
        def hook(module, input, output):
            if name == "vit_layer10" or name == "vit_layer11":
                self.intermediate_features[name] = output[0].transpose(0, 1).detach()
            else:
                self.intermediate_features[name] = output.transpose(0,1).detach()
        return hook
    def register_layers(self, layers):

        # for h in self.hooks:
        #     h.remove()
        for idx in layers:
            # print(f"register layer {idx}")
            layer = self.model.visual.transformer.resblocks[idx]
            self.hooks.append(layer.register_forward_hook(self._get_hook(f"vit_layer{idx}")))








    def forward(self, images):
        images = images
        with torch.no_grad():
            # ViT作为编码器
            image_features, attn_weights, cls_token, feats = self.model.encode_image(images)
            # ResNet作为编码器
            # image_features, x1, x2, x3, x4 = self.model.encode_image(images)
            text_features = self.model.encode_text(self.text_inputs)

            # 归一化特征
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 计算相似度（logits）
            logit_scale = self.model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            # ViT作为编码器
            return logits, attn_weights, cls_token, feats
            # ResNet作为编码器
            # return logits, x1, x2, x3, x4

def compute_cka(feat1, feat2):
    x = feat1.view(feat1.size(0), -1)
    y = feat2.view(feat2.size(0), -1)
    def center(m):
        gram = m @ m.t()
        n = gram.shape[0]
        unit = torch.ones([n, n], device=gram.device) / n
        H = torch.eye(n, device=gram.device) - unit
        return H @ gram @ H

    K = center(x)
    L = center(y)
    hsic = torch.sum(K * L)
    varK = torch.sqrt(torch.sum(K * K))
    varL = torch.sqrt(torch.sum(L * L))
    return hsic / (varK * varL + 1e-6)


def compute_rbf_kernel(X, sigma=None):
    """
    计算 RBF 核矩阵 K
    X: (n, d) 维度的特征矩阵
    sigma: 带宽。如果为 None，则使用中值技巧计算。
    """
    # 1. 计算成对距离平方 ||x_i - x_j||^2
    # 利用公式: ||a-b||^2 = ||a||^2 + ||b||^2 - 2a*b
    dist_sq = torch.cdist(X, X, p=2) ** 2

    # 2. 中值技巧确定 sigma
    if sigma is None:
        # 取距离平方的中位数的平方根作为 sigma
        median_dist_sq = torch.median(dist_sq)
        sigma = torch.sqrt(median_dist_sq)

    # 3. 计算 RBF 矩阵
    K = torch.exp(-dist_sq / (2 * sigma ** 2))
    return K


def center_kernel(K):
    """中心化核矩阵 (H K H)"""
    n = K.shape[0]
    unit = torch.ones([n, n], device=K.device) / n
    I = torch.eye(n, device=K.device)
    H = I - unit
    return torch.matmul(torch.matmul(H, K), H)


def compute_hsic(K, L):
    """计算中心化后的 HSIC"""
    Kc = center_kernel(K)
    Lc = center_kernel(L)
    # HSIC 是两个中心化矩阵元素积之和
    return (Kc * Lc).sum()


def rbf_cka(X, Y):
    """
    计算 RBF CKA 相似度
    X: 模型 1 的特征 (n, d1)
    Y: 模型 2 的特征 (n, d2)
    """
    K = compute_rbf_kernel(X)
    L = compute_rbf_kernel(Y)

    hsic_kl = compute_hsic(K, L)
    hsic_kk = compute_hsic(K, K)
    hsic_ll = compute_hsic(L, L)

    return hsic_kl / (torch.sqrt(hsic_kk) * torch.sqrt(hsic_ll))
def compute_rbf_kernel(X, sigma=None):
    """
    计算 RBF 核矩阵 K
    X: (n, d) 维度的特征矩阵
    sigma: 带宽。如果为 None，则使用中值技巧计算。
    """
    # 1. 计算成对距离平方 ||x_i - x_j||^2
    # 利用公式: ||a-b||^2 = ||a||^2 + ||b||^2 - 2a*b
    dist_sq = torch.cdist(X, X, p=2) ** 2

    # 2. 中值技巧确定 sigma
    if sigma is None:
        # 取距离平方的中位数的平方根作为 sigma
        median_dist_sq = torch.median(dist_sq)
        sigma = torch.sqrt(median_dist_sq)

    # 3. 计算 RBF 矩阵
    K = torch.exp(-dist_sq / (2 * sigma ** 2))
    return K


def center_kernel(K):
    """中心化核矩阵 (H K H)"""
    n = K.shape[0]
    unit = torch.ones([n, n], device=K.device) / n
    I = torch.eye(n, device=K.device)
    H = I - unit
    return torch.matmul(torch.matmul(H, K), H)


def compute_hsic(K, L):
    """计算中心化后的 HSIC"""
    Kc = center_kernel(K)
    Lc = center_kernel(L)
    # HSIC 是两个中心化矩阵元素积之和
    return (Kc * Lc).sum()


def rbf_cka(X, Y):
    """
    计算 RBF CKA 相似度
    X: 模型 1 的特征 (n, d1)
    Y: 模型 2 的特征 (n, d2)
    """
    K = compute_rbf_kernel(X)
    L = compute_rbf_kernel(Y)

    hsic_kl = compute_hsic(K, L)
    hsic_kk = compute_hsic(K, K)
    hsic_ll = compute_hsic(L, L)

    return hsic_kl / (torch.sqrt(hsic_kk) * torch.sqrt(hsic_ll))
@balancers.register("ourbase_vlcs")
class OurBaseBalancer(basic_balancer.BasicBalancer):
    def __init__(self, compute_stats=False):
        super().__init__(compute_stats)
        self.teacher_model = None  # 教师模型实例
        self.kd_weight = 1.0  # 蒸馏损失权重
        self.feat_weight = 0.5  # 特征蒸馏权重
        # 添加任务ID到数字的映射
        self.task_id_to_index = {'V': 0, 'L': 1, 'C': 2, 'S': 3}
        self.teacher_model = None
        self.mse_criterion = nn.MSELoss()
        self.proj = nn.Conv2d(1024, 768, 1).to('cuda')

    def setup_teacher(self, device='cuda'):
        """初始化教师模型"""
        if self.teacher_model is None:
            self.teacher_model = ClipTeacherModel(device=device)
            print("CLIP teacher model initialized")

    @staticmethod
    def compute_kd_loss(student_logits, teacher_logits, T=1.0):
        """计算知识蒸馏损失"""
        student_log_softmax = F.log_softmax(student_logits / T, dim=1)
        teacher_softmax = F.softmax(teacher_logits / T, dim=1)
        return F.kl_div(student_log_softmax, teacher_softmax, reduction='batchmean') * (T * T)



    def step_with_model(self, data: torch.Tensor, targets: dict, model: torch.nn.Module, criteria: dict, args, epoch:int,
                        **kwargs) -> None:
        # 确保教师模型已初始化
        if not hasattr(self, 'teacher_model') or self.teacher_model is None:
            self.setup_teacher(device=data.device)
        # 计算教师logits（整个batch）
        with torch.no_grad():
            # teacher_logits = self.teacher_model(data)
            # ViT作为编码器
            teacher_logits, tea_attn_weights, tea_token, tea_feat = self.teacher_model(data)
            # ResNet 作为编码器
            # teacher_logits, tea_x1, tea_x2, tea_x3, tea_x4 = self.teacher_model(data, return_features=False)
        # 获取任务标签
        task_label = torch.argmax(kwargs['task_targets'], dim=1)


        # 根据架构选择不同的损失计算函数
        if 'grad' in args.arch:
            losses, hrepr = self.compute_losses_task_multidomain(
                data=data, targets=targets,
                task_targets=kwargs['task_targets'], model=model,
                criteria=criteria
            )

            # 为Grad架构计算蒸馏损失
            for task_id in model.decoders.keys():
                # 获取属于当前任务的样本
                task_idx = self.task_id_to_index[task_id]
                task_mask = (task_label == task_idx)
                if task_mask.sum() == 0:
                    continue

                # 计算教师logits（仅当前任务）
                task_data = data[task_mask]
                task_teacher_logits = teacher_logits[task_mask]

                # Grad架构需要任务特定的前向传播
                if isinstance(model, torch.nn.DataParallel):
                    task_hrepr = model.module.encoder(task_data, task_label=kwargs['task_targets'][task_mask])
                else:
                    task_hrepr = model.encoder(task_data, task_label=kwargs['task_targets'][task_mask])

                # 新增：对特征表示进行展平处理
                avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
                task_hrepr = avgpool(task_hrepr)  # 将特征图转换为特征向量
                task_hrepr = task_hrepr.view(task_hrepr.size(0), -1)  # 展平为2D张量

                # 计算学生logits
                if isinstance(model, torch.nn.DataParallel):
                    task_student_logits = model.module.decoders[task_id](task_hrepr)
                else:
                    task_student_logits = model.decoders[task_id](task_hrepr)

                # 计算蒸馏损失
                kd_loss = self.compute_kd_loss(task_student_logits, task_teacher_logits)
                losses[task_id] += self.kd_weight * kd_loss

        elif 'soft_router' in args.arch:

            # compute_losses_soft_multidomain这个函数可以获取到encoder的特征,可以模仿这个函数获取encoder的特征得到学生ViT特征
            losses, hrepr = self.compute_losses_soft_multidomain(
                data=data, targets=targets,
                task_targets=kwargs['task_targets'], model=model,
                criteria=criteria
            )


            for task_id in model.decoders.keys():

                # 获取属于当前任务的样本
                task_idx = self.task_id_to_index[task_id]
                task_mask = (task_label == task_idx)
                if task_mask.sum() == 0:
                    continue
                # 真实标签
                task_true_labels = targets[task_mask]
                # 计算学生logits（仅当前任务）
                task_hrepr = hrepr[task_mask]
                task_teacher_logits = teacher_logits[task_mask]

                 # SoftRouter使用相同的表示
                if isinstance(model, torch.nn.DataParallel):
                    task_student_logits = model.module.decoders[task_id](task_hrepr)
                else:
                    task_student_logits = model.decoders[task_id](task_hrepr)
                kd_loss = self.compute_kd_loss(task_student_logits, task_teacher_logits)
                # if task_id == 'L':
                #     losses[task_id] += 10 * kd_loss
                # else:
                #     losses[task_id] += 0.5 * kd_loss
                losses[task_id] += 0.5 * kd_loss
            # # ======================================================================
            #
            # ==============================加入特征蒸馏损失==============================
            if isinstance(model, torch.nn.DataParallel):
                hrepr, attn_weights, proj_feat, proj_token, scores, x3 = model.module.encoder(data, return_scores=True)
            else:
                hrepr, attn_weights, proj_feat, proj_token, scores, x3 = model.encoder(data, return_scores=True)

            # print("tea_shape:", tea_attn_weights[0].shape)
            # print("tea_shape:", tea_attn_weights[1].shape)
            # print("tea_shape:", tea_attn_weights[2].shape)
            # print("tea_shape:", tea_attn_weights[3].shape)
            # print(attn_weights[0].shape)
            pca_loss =  0.2 * self.mse_criterion(attn_weights[0],tea_attn_weights[2][:,1:,1:].detach())+0.05 * self.mse_criterion(attn_weights[1],tea_attn_weights[3][:,1:,1:].detach())
            gl_loss = self.mse_criterion(proj_token,tea_token.detach()) +  1.0 * self.mse_criterion(proj_feat,tea_feat.detach())

            # cka = rbf_cka(x3.view(x3.size(0),-1).detach(), tea_feat.view(tea_feat.size(0),-1).detach())
            # print("cka:", cka)
            # cka = rbf_cka(proj_feat.view(proj_feat.size(0),-1).detach(), tea_feat.view(tea_feat.size(0),-1).detach())
            # print("pro:", cka)
            # cka = compute_cka(x3.detach(), tea_feat.detach())
            # print("cka:", cka)
            # pro = compute_cka(proj_feat.detach(), tea_feat.detach())
            # print("pro:", pro)
            # vit_feats = self.teacher_model.intermediate_features
            # res_feats = model.encoder.intermediate_features
            # print(f"ResNet keys: {res_feats.keys()}")
            # print(f"ViT keys: {vit_feats.keys()}")
            # # print(f"shape:{vit_feats['vit_layer11'].shape}")
            # cka_1_8 = compute_cka(res_feats['layer1'], vit_feats['vit_layer8'])
            # cka_1_9 = compute_cka(res_feats['layer1'], vit_feats['vit_layer9'])
            # cka_1_10 = compute_cka(res_feats['layer1'], vit_feats['vit_layer10'])
            # cka_1_11 = compute_cka(res_feats['layer1'], vit_feats['vit_layer11'])
            # print(f"CKA (Res-L1, ViT-L8): {cka_1_8.item():.4f}")
            # print(f"CKA (Res-L1, ViT-L9): {cka_1_9.item():.4f}")
            # print(f"CKA (Res-L1, ViT-L10): {cka_1_10.item():.4f}")
            # print(f"CKA (Res-L1, ViT-L11): {cka_1_11.item():.4f}")
            # cka_2_8 = compute_cka(res_feats['layer2'], vit_feats['vit_layer8'])
            # cka_2_9 = compute_cka(res_feats['layer2'], vit_feats['vit_layer9'])
            # cka_2_10 = compute_cka(res_feats['layer2'], vit_feats['vit_layer10'])
            # cka_2_11 = compute_cka(res_feats['layer2'], vit_feats['vit_layer11'])
            # print(f"CKA (Res-L2, ViT-L8): {cka_2_8.item():.4f}")
            # print(f"CKA (Res-L2, ViT-L9): {cka_2_9.item():.4f}")
            # print(f"CKA (Res-L2, ViT-L10): {cka_2_10.item():.4f}")
            # print(f"CKA (Res-L2, ViT-L11): {cka_2_11.item():.4f}")
            # cka_3_8 = compute_cka(res_feats['layer3'], vit_feats['vit_layer8'])
            # cka_3_9 = compute_cka(res_feats['layer3'], vit_feats['vit_layer9'])
            # cka_3_10 = compute_cka(res_feats['layer3'], vit_feats['vit_layer10'])
            # cka_3_11 = compute_cka(res_feats['layer3'], vit_feats['vit_layer11'])
            # print(f"CKA (Res-L3, ViT-L8): {cka_3_8.item():.4f}")
            # print(f"CKA (Res-L3, ViT-L9): {cka_3_9.item():.4f}")
            # print(f"CKA (Res-L3, ViT-L10): {cka_3_10.item():.4f}")
            # print(f"CKA (Res-L3, ViT-L11): {cka_3_11.item():.4f}")
            # cka_4_8 = compute_cka(res_feats['layer4'], vit_feats['vit_layer8'])
            # cka_4_9 = compute_cka(res_feats['layer4'], vit_feats['vit_layer9'])
            # cka_4_10 = compute_cka(res_feats['layer4'], vit_feats['vit_layer10'])
            # cka_4_11 = compute_cka(res_feats['layer4'], vit_feats['vit_layer11'])
            # print(f"CKA (Res-L4, ViT-L8): {cka_4_8.item():.4f}")
            # print(f"CKA (Res-L4, ViT-L9): {cka_4_9.item():.4f}")
            # print(f"CKA (Res-L4, ViT-L10): {cka_4_10.item():.4f}")
            # print(f"CKA (Res-L4, ViT-L11): {cka_4_11.item():.4f}")
            # ----------计算特征相似度------------
            # x3 = self.proj(x3)
            # x3 = x3.reshape(x3.shape[0], x3.shape[1], -1).transpose(1, 2)
            # sim = F.cosine_similarity(x3,tea_feat.detach(), dim=-1)
            # avg_sim = sim.mean(dim=1).mean(dim=0)
            # sim = F.cosine_similarity(proj_feat, tea_feat.detach(), dim=-1)
            # avg_sim1 = sim.mean(dim=1).mean(dim=0)
            # print("our:", avg_sim1)
            # print("direct:", avg_sim)

            # ----------计算CKA值------------
            # x3 = x3.reshape(x3.shape[0], x3.shape[1], -1).transpose(1, 2)
            # print("x3_shape:", x3.shape)
            # print("tea_shape:", tea_feat.shape)
            # print("proj_feat:", proj_feat.shape)
            # from q import cka_linear_friendly
            # cka_12 = cka_linear_friendly(x3, tea_feat)
            # print("dir_cka:", cka_12)
            # cka_13 = cka_linear_friendly(proj_feat, tea_feat)
            # print("pro_cka:", cka_13)


            # ----------加入其它蒸馏方法------------
            # 将CNN通道数转换成768
            # x3 = self.proj(x3)
            # tea_feat = tea_feat.transpose(1, 2).reshape(tea_feat.shape[0], 768, 14, 14)
            # PKT特征蒸馏
            # pkt = PKT()
            # pkt_loss = pkt(x3, tea_feat.detach())
            # KDSVD特征蒸馏
            # kdsvd = KDSVD()
            # kdsvd_loss = kdsvd(x3, tea_feat.detach())
            # FSP特征蒸馏

            # fsp = FSP()
            # fsp_loss = fsp(x3, tea_feat.detach())
            # FT特征蒸馏
            # ft = FactorTransfer()
            # ft_loss = ft(x3, tea_feat.detach())
            # VID特征蒸馏
            # vid = VIDLoss()
            # vid_loss = vid(x3, tea_feat.detach())
            # NST特征蒸馏
            # nst = NSTLoss()
            # nst_loss = nst(x3, tea_feat.detach())
            # SP特征蒸馏
            # sp = Similarity()
            # sp_loss = sp(x3, tea_feat.detach())
            # AT特征蒸馏
            # at = Attention()
            # at_loss = at(x3, tea_feat.detach())
            # print("at loss", at_loss)
            # AB特征蒸馏
            # ab = ABLoss(len(x3[1:-1]))
            # ab_loss = ab(x3, tea_feat.detach())
            # CC特征蒸馏
            # cc = Correlation()
            # cc_loss = cc(x3, tea_feat.detach())
            # FitNet特征蒸馏
            # fitnet = HintLoss()
            # fitnet_loss = fitnet(x3, tea_feat.detach())
            # print("fitnet_loss", fitnet_loss)
            # RKD特征蒸馏
            # rkd = RKDLoss()
            # rkd_loss = rkd(x3, tea_feat.detach())
            # print("rkd loss", rkd_loss)

            # print("shape:", tea_token.shape) #[32,768]


        else:
            losses, hrepr = self.compute_losses_multidomain(
                data=data, targets=targets,
                task_targets=kwargs['task_targets'], model=model,
                criteria=criteria
            )

            # 为其他架构计算蒸馏损失

            # for task_id in model.decoders.keys():
            #     # 获取属于当前任务的样本
            #     task_idx = self.task_id_to_index[task_id]
            #     task_mask = (task_label == task_idx)
            #     if task_mask.sum() == 0:
            #         continue
            #
            #
            #     # 计算学生logits（仅当前任务）
            #     task_hrepr = hrepr[task_mask]
            #     task_teacher_logits = teacher_logits[task_mask]
            #
            #     if isinstance(model, torch.nn.DataParallel):
            #         task_student_logits = model.module.decoders[task_id](task_hrepr)
            #     else:
            #         task_student_logits = model.decoders[task_id](task_hrepr)
            #
            #     # 计算蒸馏损失
            #     kd_loss = self.compute_kd_loss(task_student_logits, task_teacher_logits)
            #     # print("kdloss:",kd_loss)
            #     losses[task_id] += self.kd_weight * kd_loss
            # ==============================加入特征蒸馏损失==============================

            if isinstance(model, torch.nn.DataParallel):
                # ViT作为编码器采用跨架构映射器返回的Transformer特征
                _, attn_weights, proj_feat, proj_token = model.module.encoder(data)
                # ResNet作为编码器，返回CNN特征
                # _, x1, x2, x3, x4 = model.module.encoder(data)
            else:
                # ViT作为编码器采用跨架构映射器返回的Transformer特征
                _, attn_weights, proj_feat, proj_token = model.encoder(data)
                # ResNet作为编码器，返回CNN特征
                # _, stu_x1, stu_x2, stu_x3, stu_x4 = model.encoder(data)
            # print("attn_weights[0]:",attn_weights[0].shape)
            # print("tea_attn_weights[2][:,1:,1:]:", tea_attn_weights[2][:,1:,1:].shape)
            # print("attn_weights[1]:", attn_weights[1].shape)
            # print("tea_attn_weights[3][:,1:,1:]:", tea_attn_weights[3][:, 1:, 1:].shape)
            # print("proj_token:", proj_token.shape)
            # print("tea_token:", tea_token.shape)
            # print("proj_feat:", proj_feat.shape)
            # print("tea_feat:", tea_feat.shape)
            # pca_loss =  0.2 * self.mse_criterion(attn_weights[0],tea_attn_weights[2][:,1:,1:].detach())+0.05 * self.mse_criterion(attn_weights[1],tea_attn_weights[3][:,1:,1:].detach())
            # gl_loss = self.mse_criterion(proj_token,tea_token.detach()) +  1.0 * self.mse_criterion(proj_feat,tea_feat.detach())
            # feat_loss = self.mse_criterion(stu_x1, tea_x1)
            # print("feat_loss:", feat_loss)



        # 计算总损失并回传

        total_loss = sum(losses.values())
        # ViT作为图像编码器时损失计算
        total_loss = total_loss
        # total_loss = total_loss + 1.0 * (pca_loss +  gl_loss)
        # total_loss = total_loss + sum(at_loss)
        # total_loss = total_loss + sum(pkt_loss)
        # total_loss = total_loss + 0.5 * cl_loss + 0.1 * patch_loss
        # ResNet作为CLIP图像编码器时损失计算
        # total_loss = total_loss + 1.0 * feat_loss
        total_loss.backward()
        self.set_losses(losses)

        # ===================== 新增：梯度冲突计算逻辑 =====================
        # 仅在需要统计时计算（compute_stats=True）



