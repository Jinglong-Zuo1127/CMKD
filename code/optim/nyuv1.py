import torch
from .. import basic_balancer
from .. import balancers

@balancers.register("ourbase_vlcs")
class OurBaseBalancer(basic_balancer.BasicBalancer):


    def step_with_model(self, data: torch.Tensor, targets: dict, model: torch.nn.Module, criteria: dict, args,
                        **kwargs) -> None:



        if 'grad' in args.arch:
            losses, hrepr = self.compute_losses_task(data, targets, model, criteria)
        else:
            losses, hrepr = self.compute_losses(data, targets, model, criteria)


        total_loss = sum(losses.values())
        total_loss.backward()
        self.set_losses(losses)

import torch
from .. import basic_balancer
from .. import balancers
import torch.nn.functional as F
import torch.nn as nn
from CLIP import clip
from dist_zoo.RKD import RKDLoss
class ClipTeacherModel(torch.nn.Module):
    """CLIP教师模型 - OfficeHome版本"""

    def __init__(self, device='cuda', model_name='ViT-B/16'):
        super().__init__()
        self.device = device
        self.model, _ = clip.load(model_name, device=device)

        self.model = self.model.float()
        self.model.eval()  # 固定为评估模式

    def forward(self, images):
        """计算教师logits（与VLCS实现一致）"""
        with torch.no_grad():
            # image_features, attn_weights, cls_token, feats = self.model.encode_image(images)
            # text_features = self.model.encode_text(self.text_inputs)
            images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
            image_features, attn_weights, cls_token, feats = self.model.encode_image(images)

        return None, attn_weights, cls_token, feats


@balancers.register("ourbase_nyuv")
class OurBaseBalancer(basic_balancer.BasicBalancer):
    def __init__(self, compute_stats=False):
        super().__init__(compute_stats)
        self.teacher_model = None
        self.mse_criterion = nn.MSELoss()
    def setup_teacher(self, device='cuda'):
        """初始化教师模型"""
        if self.teacher_model is None:
            self.teacher_model = ClipTeacherModel(device=device)
            print("CLIP teacher model for OfficeHome initialized")

    @staticmethod
    def compute_kd_loss(student_logits, teacher_logits, T=1.0):
        """计算知识蒸馏损失（与VLCS实现一致）"""
        student_log_softmax = F.log_softmax(student_logits / T, dim=1)
        teacher_softmax = F.softmax(teacher_logits / T, dim=1)
        return F.kl_div(student_log_softmax, teacher_softmax, reduction='batchmean') * (T * T)


    def step_with_model(self, data: torch.Tensor, targets: dict, model: torch.nn.Module, criteria: dict, args,
                        **kwargs) -> None:

        # 确保教师模型已初始化
        if not hasattr(self, 'teacher_model') or self.teacher_model is None:
            self.setup_device = data.device
            self.setup_teacher(device=data.device)


        # 计算教师logits（整个batch）- 不再需要domain信息
        with torch.no_grad():
            teacher_logits, tea_attn_weights, tea_token, tea_feat = self.teacher_model(data)



        if 'soft_router' in args.arch:
            losses, hrepr = self.compute_losses(data, targets, model, criteria)
            # ==============================加入特征蒸馏损失==============================
            if isinstance(model, torch.nn.DataParallel):
                hrepr, attn_weights, proj_feat, proj_token, scores, x3 = model.module.encoder(data, return_scores=True)
            else:
                hrepr, attn_weights, proj_feat, proj_token, scores,_ = model.encoder(data, return_scores=True)

            pca_loss = 0.2 * self.mse_criterion(attn_weights[0],tea_attn_weights[2][:, 1:, 1:].detach()) + 0.05 * self.mse_criterion(attn_weights[1], tea_attn_weights[3][:, 1:, 1:].detach())
            #
            gl_loss = self.mse_criterion(proj_token, tea_token.detach()) + 1.0 * self.mse_criterion(proj_feat,tea_feat.detach())



        # 计算总损失并回传
        total_loss = sum(losses.values())
        total_loss = total_loss + 0.5 * ( pca_loss + gl_loss)
        # total_loss = total_loss
        total_loss.backward()
        self.set_losses(losses)
