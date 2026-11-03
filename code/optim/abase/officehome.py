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

        # OfficeHome的实际65个类别名称
        self.class_names = [
            'Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator',
            'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp',
            'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder',
            'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop',
            'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen',
            'Pencil', 'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors',
            'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'Table', 'Telephone',
            'ToothBrush', 'Toys', 'Trash_Can', 'TV', 'Webcam'
        ]

        # 生成文本提示: "a photo of a {class_name}"
        # 注意：将下划线替换为空格，使描述更自然
        self.text_inputs = torch.cat([
            clip.tokenize(f"a photo of a {c.replace('_', ' ')}") for c in self.class_names
        ]).to(device)
        self.model = self.model.float()
        self.model.eval()  # 固定为评估模式

    def forward(self, images):
        """计算教师logits（与VLCS实现一致）"""
        with torch.no_grad():
            image_features, attn_weights, cls_token, feats = self.model.encode_image(images)
            text_features = self.model.encode_text(self.text_inputs)

            # 归一化特征
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 计算相似度（logits）
            logit_scale = self.model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

        return logits, attn_weights, cls_token, feats


@balancers.register("ourbase_officehome")
class OurBaseBalancer(basic_balancer.BasicBalancer):
    def __init__(self, compute_stats=False):
        super().__init__(compute_stats)
        self.teacher_model = None  # 教师模型实例
        self.kd_weight = 0.1  # 蒸馏损失权重（默认值）
        self.kd_freq_weight = 0.01
        # 添加任务ID到数字的映射
        self.task_id_to_index = {'P': 0, 'A': 1, 'C': 2, 'R': 3}
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

        # 获取任务标签
        task_targets = kwargs['task_targets']
        task_label = torch.argmax(task_targets, dim=1)

        # 计算教师logits（整个batch）- 不再需要domain信息
        with torch.no_grad():
            teacher_logits, tea_attn_weights, tea_token, tea_feat = self.teacher_model(data)

        # 根据模型架构选择不同的损失计算方式
        if 'grad' in args.arch:
            losses, hrepr = self.compute_losses_task_multidomain(data=data, targets=targets,
                                                                 task_targets=kwargs['task_targets'], model=model,
                                                                 criteria=criteria)
            # 为每个任务计算蒸馏损失
            for task_id in model.decoders.keys():
                task_idx = self.task_id_to_index[task_id]
                task_mask = (task_label == task_idx)
                if task_mask.sum() == 0:
                    continue

                # 提取当前任务的教师logits
                task_teacher_logits = teacher_logits[task_mask]

                # Grad架构需要任务特定的前向传播
                if isinstance(model, torch.nn.DataParallel):
                    task_hrepr = model.module.encoder(data[task_mask], task_label=task_targets[task_mask])
                else:
                    task_hrepr = model.encoder(data[task_mask], task_label=task_targets[task_mask])

                # 对特征表示进行展平处理 - 修复维度问题
                avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
                task_hrepr = avgpool(task_hrepr).flatten(1)  # 保持二维 (batch_size, features)

                # 计算学生logits
                if isinstance(model, torch.nn.DataParallel):
                    task_student_logits = model.module.decoders[task_id](task_hrepr)
                else:
                    task_student_logits = model.decoders[task_id](task_hrepr)

                # 维度检查 - 确保是二维张量
                if task_student_logits.dim() < 2:
                    task_student_logits = task_student_logits.unsqueeze(0)
                    task_teacher_logits = task_teacher_logits.unsqueeze(0)

                # 计算蒸馏损失
                kd_loss = self.compute_kd_loss(task_student_logits, task_teacher_logits)
                losses[task_id] += self.kd_weight * kd_loss

        elif 'soft_router' in args.arch:
            losses, hrepr = self.compute_losses_soft_multidomain(data=data, targets=targets,
                                                                 task_targets=kwargs['task_targets'], model=model,
                                                                 criteria=criteria)
            # ==============================加入logits损失==============================
            for task_id in model.decoders.keys():
                task_idx = self.task_id_to_index[task_id]
                task_mask = (task_label == task_idx)
                if task_mask.sum() == 0:
                    continue

                # 提取当前任务的教师logits
                task_teacher_logits = teacher_logits[task_mask]

                # 提取当前任务的学生logits
                if isinstance(hrepr, list):
                    task_hrepr = hrepr[task_idx][task_mask]
                else:
                    task_hrepr = hrepr[task_mask]

                # 计算学生logits
                if isinstance(model, torch.nn.DataParallel):
                    task_student_logits = model.module.decoders[task_id](task_hrepr)
                else:
                    task_student_logits = model.decoders[task_id](task_hrepr)

                if task_student_logits.dim() < 2:
                    task_student_logits = task_student_logits.unsqueeze(0)
                    task_teacher_logits = task_teacher_logits.unsqueeze(0)

                # 计算蒸馏损失
                kd_loss = self.compute_kd_loss(task_student_logits, task_teacher_logits)
                losses[task_id] += 0.9 * kd_loss

            # ==============================加入特征蒸馏损失==============================
            if isinstance(model, torch.nn.DataParallel):
                hrepr, attn_weights, proj_feat, proj_token, scores, x3 = model.module.encoder(data, return_scores=True)
            else:
                hrepr, attn_weights, proj_feat, proj_token, scores = model.encoder(data, return_scores=True)

            # rkd = RKDLoss()
            # rkd_loss = rkd(x3, tea_token.detach())
            # print("rkd loss", rkd_loss)
            pca_loss = 0.2 * self.mse_criterion(attn_weights[0],tea_attn_weights[2][:, 1:, 1:].detach()) + 0.05 * self.mse_criterion(attn_weights[1], tea_attn_weights[3][:, 1:, 1:].detach())
            #
            gl_loss = self.mse_criterion(proj_token, tea_token.detach()) + 1.0 * self.mse_criterion(proj_feat,tea_feat.detach())
            # sim = F.cosine_similarity(proj_feat,tea_feat.detach(), dim=-1)
            # avg_sim = sim.mean(dim=1).mean(dim=0)
            # print("our:", avg_sim)

        else:
            losses, hrepr = self.compute_losses_multidomain(data=data, targets=targets,
                                                            task_targets=kwargs['task_targets'], model=model,
                                                            criteria=criteria)
            # 为每个任务计算蒸馏损失
            for task_id in model.decoders.keys():
                task_idx = self.task_id_to_index[task_id]
                task_mask = (task_label == task_idx)
                if task_mask.sum() == 0:
                    continue

                # 提取当前任务的教师logits
                task_teacher_logits = teacher_logits[task_mask]

                # 提取当前任务的学生logits
                task_hrepr = hrepr[task_mask]

                if isinstance(model, torch.nn.DataParallel):
                    task_student_logits = model.module.decoders[task_id](task_hrepr)
                else:
                    task_student_logits = model.decoders[task_id](task_hrepr)

                # 计算蒸馏损失
                kd_loss = self.compute_kd_loss(task_student_logits, task_teacher_logits)
                losses[task_id] += self.kd_weight * kd_loss


        # 计算总损失并回传
        total_loss = sum(losses.values())
        total_loss = total_loss + 0.9 * ( pca_loss + gl_loss)
        # total_loss = total_loss
        total_loss.backward()
        self.set_losses(losses)
