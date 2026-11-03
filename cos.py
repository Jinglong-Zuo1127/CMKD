import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

# 导入你项目中的组件
from code.benchmarks.mtl_benchmark import get_benchmark_class
from code.optim.abase.vlcs import ClipTeacherModel
from layers.router import build_router  # 用于修复报错
import code.utils.utils as utils


## 1. 定义线性对齐层 (Linear Aligner)
class LinearAligner(nn.Module):
    def __init__(self, student_dim, teacher_dim):
        super(LinearAligner, self).__init__()
        # 按照论文 Definition 2，使用线性层映射特征空间 [cite: 214]
        self.aligner = nn.Linear(student_dim, teacher_dim)

    def forward(self, x):
        return self.aligner(x)


## 2. 修复 Router 的函数
def fix_model_router(model, args):
    """
    手动为模型及其所有子模块挂载 Router，防止 'NoneType' object is not callable 报错
    """
    print("正在检测并修复模型 Router...")
    # 创建一个通用的 Router
    router = build_router(num_experts=args.n_expert).cuda()
    model.router = router
    # 递归遍历所有子模块，如果模块有 router 属性则赋值
    for module in model.modules():
        if hasattr(module, 'router'):
            module.router = router
    return model


def train_and_eval_transferability(args):
    # 基础设置
    utils.fix_seed(46)
    device = torch.device('cuda')

    # 加载数据集
    benchmark = get_benchmark_class(args.benchmark)(args)
    train_loader = DataLoader(benchmark.datasets['train'], batch_size=32, shuffle=True, num_workers=8)
    test_loader = DataLoader(benchmark.datasets['valid'], batch_size=32, shuffle=False)

    # 加载学生模型
    student_model = benchmark.get_model(args).to(device)
    # --- 核心修复步骤 ---
    student_model = fix_model_router(student_model, args)
    # ------------------
    student_model.eval()

    # 加载教师模型
    teacher_model = ClipTeacherModel(device=device).to(device)
    teacher_model.eval()

    # 维度定义：ResNet50 是 2048，CLIP ViT-B/16 是 768
    student_dim = 1024
    teacher_dim = 768

    # 定义 Aligner 和 优化器
    aligner = LinearAligner(student_dim, teacher_dim).to(device)
    optimizer = torch.optim.Adam(aligner.parameters(), lr=1e-3)

    # 按照你的要求：仅使用欧氏距离（MSE）作为损失函数
    criterion = nn.MSELoss()

    print("\n--- 步骤 1: 训练 Aligner (仅优化欧氏距离) ---")
    epochs = 10
    for epoch in range(epochs):
        aligner.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for data, _, task_targets in pbar:
            images = data.to(device)
            task_targets = task_targets.to(device)

            with torch.no_grad():
                # 1. 提取教师特征 (CLIP)
                # 根据你的代码，teacher_model 返回 (logits, attn, token, feats) [cite: 541]
                _, _, _, f_t = teacher_model(images)

                # 2. 提取学生原始特征
                # 由于你的模型 forward 返回很多值，这里假设取第一个输出并进行池化
                # 注意：必须传入 task_label 否则模型内部 forward 可能会报错
                # 找到 task_label 的索引
                task_label = torch.argmax(task_targets, dim=1)

                # 获取学生特征。根据你的 vlcs.py，encoder 返回 (hrepr, attn, proj_feat, proj_token, ...)
                # 我们需要 penultimate 特征，通常是经过池化后的 hrepr
                student_outputs = student_model.encoder(images)
                f_s = student_outputs[4]  # 获取 hrepr
                f_s = f_s.reshape(f_s.shape[0],f_s.shape[1],-1).transpose(1,2)
                # 如果是 4D 特征图 [B, C, H, W]，转为 2D 向量 [B, C]
                # if len(f_s.shape) == 4:
                #     f_s = F.adaptive_avg_pool2d(f_s, (1, 1)).flatten(1)

            # 训练对齐层
            optimizer.zero_grad()
            # print("fs", f_s.shape)
            aligned_f_s = aligner(f_s)
            # print("align",aligned_f_s.shape)
            # print("f_t", f_t.shape)
            loss = criterion(aligned_f_s, f_t.detach())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"mse_loss": f"{loss.item():.4f}"})

        print(f"Epoch 平均 Loss: {total_loss / len(train_loader):.6f}")

    print("\n--- 步骤 2: 计算余弦相似度 (Transferability Score) ---")
    aligner.eval()
    cos_sims = []
    with torch.no_grad():
        for data, _, _ in tqdm(test_loader):
            images = data.to(device)
            # 提取学生和教师特征
            student_outputs = student_model.encoder(images)
            f_s = student_outputs[4]
            f_s = f_s.reshape(f_s.shape[0],f_s.shape[1],-1).transpose(1,2)

            _, _, _, f_t = teacher_model(images)

            # 计算对齐后的相似度
            aligned_f_s = aligner(f_s)
            sim = F.cosine_similarity(aligned_f_s, f_t, dim=1)
            cos_sims.append(sim.cpu().numpy())

    final_transferability = np.concatenate(cos_sims).mean()
    print(f"\n==========================================")
    print(f"最终可迁移性得分 (Cosine Similarity): {final_transferability:.4f}")
    print(f"==========================================\n")


if __name__ == "__main__":
    parser = utils.common_argparser()
    # 确保传入必要的参数，如 --n_expert
    args, _ = parser.parse_known_args()

    # 如果你的模型需要特定的参数，可以在这里手动指定
    # args.benchmark = "vlcs_resnet50"

    train_and_eval_transferability(args)