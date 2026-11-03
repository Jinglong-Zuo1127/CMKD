from .. import basic_balancer
from .. import balancers
import torch.nn.functional as F
from CLIP import clip
import torch

import torch.nn as nn
import numpy as np
import json
import os
from layers.router import build_router
from code.benchmarks.vlcs import TeacherModel,VLCSBenchmark
import code.utils.utils as utils

parser = utils.common_argparser()
args, _ = parser.parse_known_args()

specific_parser = VLCSBenchmark.get_arg_parser(parser)
args = specific_parser.parse_args()

teacher = TeacherModel(args) # 创建模型实例
router = build_router(num_experts=args.n_expert)
teacher = teacher.to('cuda')
router = router.to('cuda')
teacher.router = router
for module in teacher.modules():
    if hasattr(module, 'router'):
        module.router = router
# checkpoint = torch.load("/home/zuojinlong/code/MoLA/logs/vlcs_resnet50/ourbase_teacher/lora_soft_router_teacher/20/best_test.pth", map_location='cpu')  # 加载权重

checkpoint = torch.load("/home/zuojinlong/code/MoLA/logs/vlcs_resnet50/ourbase_teacher/lora_soft_router_teacher/33/best_test.pth", map_location='cpu')



if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    # 例如来自训练脚本保存的 checkpoint
    # state_dict = checkpoint['state_dict']
    # print("\n权重文件的键名数量:", len(state_dict.keys()))
    # print("权重文件的键名（前20个）:")
    # for key in state_dict.keys():
    #     print(key)

    # print(checkpoint['state_dict'].keys)
    teacher.load_state_dict(checkpoint['state_dict'])
else:
    # 直接是 state_dict
    teacher.load_state_dict(checkpoint)

# ==================== 2. 设置模型为评估模式 ====================
teacher.eval()
mse_loss = nn.MSELoss()
task_id_to_index = {'V': 0, 'L': 1, 'C': 2, 'S': 3}
@balancers.register("ourbase_teacher")
class OurBaseBalancer(basic_balancer.BasicBalancer):

    @staticmethod
    def compute_kd_loss(student_logits, teacher_logits, T=1.0):
        """计算知识蒸馏损失"""
        student_log_softmax = F.log_softmax(student_logits / T, dim=1)
        teacher_softmax = F.softmax(teacher_logits / T, dim=1)
        return F.kl_div(student_log_softmax, teacher_softmax, reduction='batchmean') * (T * T)
    def step_with_model(self, data: torch.Tensor, targets: dict, model: torch.nn.Module, criteria: dict, args, epoch:int,
                        **kwargs) -> None:

        task_label = torch.argmax(kwargs['task_targets'], dim=1)
        # 根据架构选择不同的损失计算函数
        if 'grad' in args.arch:
            losses, hrepr = self.compute_losses_task_multidomain(
                data=data, targets=targets,
                task_targets=kwargs['task_targets'], model=model,
                criteria=criteria
            )



        elif 'soft_router' in args.arch:

            losses, hrepr = self.compute_losses_soft_multidomain(
                data=data, targets=targets,
                task_targets=kwargs['task_targets'], model=model,
                criteria=criteria
            )

            for task_id in model.decoders.keys():

                # 获取属于当前任务的样本
                task_idx = task_id_to_index[task_id]
                task_mask = (task_label == task_idx)
                if task_mask.sum() == 0:
                    continue
                # 真实标签
                task_true_labels = targets[task_mask]
                # 计算学生logits（仅当前任务）
                task_hrepr = hrepr[task_mask]
                # 教师模型logits
                if isinstance(model, torch.nn.DataParallel):
                    task_teacher_logits = teacher.module.decoders[task_id](task_hrepr)
                else:
                    task_teacher_logits = teacher.decoders[task_id](task_hrepr)

                # 学生模型logits
                if isinstance(model, torch.nn.DataParallel):
                    task_student_logits = model.module.decoders[task_id](task_hrepr)
                else:
                    task_student_logits = model.decoders[task_id](task_hrepr)

                kd_loss = self.compute_kd_loss(task_student_logits, task_teacher_logits)
                # print(kd_loss.item())
                # if task_id == "V":
                #     losses[task_id] += 10 * kd_loss
                # else:
                #     losses[task_id] += 0.1 * kd_loss
                losses[task_id] += 0.1 * kd_loss

            if isinstance(model, torch.nn.DataParallel):
                teacher_hrepr, attn_weights, proj_feat, proj_token, scores, teacher_x3 = teacher.module.encoder(data, return_scores=True)
            else:
                teacher_hrepr, attn_weights, proj_feat, proj_token, scores, teacher_x3 = teacher.encoder(data, return_scores=True)

            if isinstance(model, torch.nn.DataParallel):
                hrepr, attn_weights, proj_feat, proj_token, scores, x3 = model.module.encoder(data, return_scores=True)
            else:
                hrepr, attn_weights, proj_feat, proj_token, scores, x3 = model.encoder(data, return_scores=True)

            distill_loss = mse_loss(hrepr, teacher_hrepr)


        else:
            losses, hrepr = self.compute_losses_multidomain(
                data=data, targets=targets,
                task_targets=kwargs['task_targets'], model=model,
                criteria=criteria
            )



        # 计算总损失并回传

        total_loss = sum(losses.values())
        total_loss = total_loss + distill_loss * 0.1
        total_loss.backward()
        self.set_losses(losses)



