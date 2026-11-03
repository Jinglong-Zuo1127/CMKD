import torch
from .. import basic_balancer
from .. import balancers
import torch.nn.functional as F
import torch.nn as nn
from CLIP import clip
import numpy as np
from clip1 import MedCLIPFinetuner
from ClipTeacher_medmnist import ClipTeacherModel

@balancers.register("ourbase_medmnist")
class OurBaseBalancer(basic_balancer.BasicBalancer):
    def __init__(self, compute_stats=False):
        super().__init__(compute_stats)
        self.teacher_model = None
        self.kd_weight = 0.005  # 知识蒸馏损失权重
        self.temperature = 1.0  # 蒸馏温度
        self.mse_criterion = nn.MSELoss()

    def load_teacher_model(self, model_path, device='cuda'):
        """加载微调后的MedCLIPFinetuner模型"""
        # 初始化教师模型以获取任务信息
        teacher_base = ClipTeacherModel(device=device)

        # 计算任务信息
        num_classes = [len(teacher_base.task_labels[i]) for i in range(9)]
        label_offsets = torch.cumsum(torch.tensor([0] + num_classes[:-1]), dim=0).tolist()

        # 任务名称
        task_names = ['pathmnist', 'octmnist', 'organcmnist', 'bloodmnist',
                      'breastmnist', 'dermamnist', 'organamnist',
                      'organsmnist', 'pneumoniamnist']

        # 构建任务信息
        task_info = {
            'task_labels': teacher_base.task_labels,
            'text_inputs': teacher_base.text_inputs,
            'num_classes': {i: num_classes[i] for i in range(9)},
            'label_offsets': label_offsets,
            'task_names': task_names
        }

        # 创建MedCLIPFinetuner实例
        model = MedCLIPFinetuner(teacher_base.model, task_info, device=device)

        # 加载微调权重
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        print(f"Successfully loaded fine-tuned teacher model from {model_path}")
        return model

    @staticmethod
    def compute_kd_loss(student_logits, teacher_logits, T=1.0):
        """计算知识蒸馏损失"""
        student_log_softmax = F.log_softmax(student_logits / T, dim=1)
        teacher_softmax = F.softmax(teacher_logits / T, dim=1)
        return F.kl_div(student_log_softmax, teacher_softmax, reduction='batchmean') * (T * T)


    def step_with_model(self, data: torch.Tensor, targets: dict, model: torch.nn.Module, criteria: dict, args,
                        **kwargs) -> None:
        # 确保教师模型已初始化
        if self.teacher_model is None:
            # 使用与MTLTrainer相同的路径加载教师模型
            self.teacher_model = self.load_teacher_model(
                model_path='/home/zjl/code/MoLA/best_finetuned.pth',
                device=data.device
            )

        # 获取任务标签（转换为任务ID）
        task_label = torch.argmax(kwargs['task_targets'], dim=1)

        # 根据架构选择不同的损失计算函数
        if 'grad' in args.arch:
            losses, hrepr = self.compute_losses_task_tasklabel(
                data=data, targets=targets,
                task_targets=kwargs['task_targets'], model=model,
                criteria=criteria, num_task=9
            )
        else:
            losses, hrepr = self.compute_losses_tasklabel(
                data=data, targets=targets,
                task_targets=kwargs['task_targets'], model=model,
                criteria=criteria, num_task=9
            )

        # 确保损失字典使用字符串键
        losses = {str(k): v for k, v in losses.items()}

        # 获取教师模型的logits
        with torch.no_grad():
            teacher_logits, _, _ ,tea_attn_weights, tea_token, tea_feat= self.teacher_model(data)

        # 统一蒸馏损失计算方法 - 按任务单独计算
        for task_id in range(9):
            task_mask = (task_label == task_id)
            if task_mask.sum() == 0:
                continue

            # 根据架构获取学生模型logits
            if 'grad' in args.arch:
                # Grad架构需要任务特定的前向传播
                if isinstance(model, torch.nn.DataParallel):
                    task_hrepr = model.module.encoder(data[task_mask], task_label=kwargs['task_targets'][task_mask])
                    task_student_logits = model.module.decoders[str(task_id)](task_hrepr)
                else:
                    task_hrepr = model.encoder(data[task_mask], task_label=kwargs['task_targets'][task_mask])
                    task_student_logits = model.decoders[str(task_id)](task_hrepr)
            else:
                # 其他架构使用共享的表示
                if isinstance(model, torch.nn.DataParallel):
                    task_student_logits = model.module.decoders[str(task_id)](hrepr[task_mask])
                else:
                    task_student_logits = model.decoders[str(task_id)](hrepr[task_mask])

            # 计算当前任务的蒸馏损失
            kd_loss = self.compute_kd_loss(
                task_student_logits,
                teacher_logits[task_id][task_mask],
                T=self.temperature
            )

            # 转换为字符串键
            task_id_str = str(task_id)

            # 添加蒸馏损失到原始损失
            if task_id_str in losses:
                losses[task_id_str] += self.kd_weight * kd_loss
                # print("kd_loss:",kd_loss)
            else:
                # 理论上不应该出现，但做安全处理
                losses[task_id_str] = self.kd_weight * kd_loss
                # print("not kd_loss:", kd_loss)

        # ==============================加入特征蒸馏损失==============================
        if isinstance(model, torch.nn.DataParallel):
            hrepr, attn_weights, proj_feat, proj_token, scores, direct = model.module.encoder(data, return_scores=True)
        else:
            hrepr, attn_weights, proj_feat, proj_token, scores, direct = model.encoder(data, return_scores=True)
        pca_loss = 0.2 * self.mse_criterion(attn_weights[0], tea_attn_weights[2][:, 1:,1:].detach()) + 0.05 * self.mse_criterion(attn_weights[1], tea_attn_weights[3][:, 1:, 1:].detach())

        gl_loss = self.mse_criterion(proj_token, tea_token.detach()) + 1.0 * self.mse_criterion(proj_feat,tea_feat.detach())
        # sim = F.cosine_similarity(proj_feat, tea_feat.detach(), dim=-1)
        # avg_sim = sim.mean(dim=1).mean(dim=0)
        # sim1 = F.cosine_similarity(direct, tea_feat.detach(), dim=-1)
        # avg_sim1 = sim1.mean(dim=1).mean(dim=0)
        # print("our:", avg_sim)
        # print("direct:", avg_sim1)

        sim = F.cosine_similarity(direct, tea_token.detach(), dim=-1)
        avg_sim = sim.mean().item()

        sim = F.cosine_similarity(proj_token, tea_token.detach(), dim=-1)
        avg_sim1 = sim.mean().item()

        print("our:", avg_sim1)
        print("direct:", avg_sim)


        # 计算总损失并回传
        total_loss = sum(losses.values())
        total_loss = total_loss + 0.05 * pca_loss + 0.05 * gl_loss
        total_loss.backward()

        # 记录所有损失
        self.set_losses(losses)
