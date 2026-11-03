# -*- coding: utf-8 -*-
"""
计算已训练好的 .pth 模型的 PL exponent α 及其他质量指标
"""

import torch
import weightwatcher as ww
import pandas as pd
import matplotlib.pyplot as plt
from torch.distributed.checkpoint import state_dict

from code.benchmarks.vlcs import VLCSModel_LoRASoftRouter,VLCSBenchmark
import code.utils.utils as utils
from code.benchmarks.mtl_benchmark import get_benchmark_class
# ==================== 1. 加载你的模型 ====================
# 你需要根据你的模型定义来加载 .pth 文件
# 这里给出两种常见情况：

# 情况 A：如果你使用的是 torchvision 中的预训练模型（仅作演示）
import torchvision.models as models
model = models.resnet18(pretrained=True)  # 以 ResNet18 为例

# 情况 B：如果你的模型是自定义的，请按以下方式加载
# 假设你已经定义了模型类 MyModel，且 .pth 文件保存了 state_dict
from code.models.vlcs import ResNet50_LoraSoftRouter  # 替换为你的模型定义导入
# parser = utils.common_argparser()
# args, _ = parser.parse_known_args()
#
# specific_parser = VLCSBenchmark.get_arg_parser(parser)
# args = specific_parser.parse_args()
#
# model = VLCSModel_LoRASoftRouter(args) # 创建模型实例
# checkpoint = torch.load("/home/zuojinlong/code/MoLA/logs/vlcs_resnet50/ourbase_vlcs/lora_soft_router/1/final_test.pth", map_location='cpu')  # 加载权重
# 如果 .pth 是整个模型（包含结构），可以直接 torch.load 得到模型
# 但通常 .pth 是 state_dict，需要 load_state_dict
# if isinstance(checkpoint, dict):
#     print("Keys in checkpoint:", checkpoint.keys())
#     # 如果包含 'state_dict'，通常是训练时保存的完整 checkpoint
#     if 'state_dict' in checkpoint:
#         state_dict = checkpoint['state_dict']
#         print("state_dict keys sample:", list(state_dict.keys())[:])
#     else:
#         state_dict = checkpoint
#         print("Direct state_dict keys sample:", list(state_dict.keys())[:5])
# else:
#     # 可能是整个模型对象
#     print("Checkpoint is a model object, not a state_dict.")
#     model = checkpoint  # 可以直接使用这个 model
# model_keys = set(model.state_dict().keys())
# print("模型实例的键名数量:", len(model_keys))
# print("模型实例的键名（前20个）:")
# for key in model_keys:
#     print(key)
# if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
#     # 例如来自训练脚本保存的 checkpoint
#     state_dict = checkpoint['state_dict']
#     print("\n权重文件的键名数量:", len(state_dict.keys()))
#     print("权重文件的键名（前20个）:")
#     for key in state_dict.keys():
#         print(key)
#
#     # print(checkpoint['state_dict'].keys)
#     model.load_state_dict(checkpoint['state_dict'])
# else:
#     # 直接是 state_dict
#     model.load_state_dict(checkpoint)

# ==================== 2. 设置模型为评估模式 ====================
model.eval()

# ==================== 3. 创建 WeightWatcher 实例并分析 ====================
watcher = ww.WeightWatcher(model=model)

# 运行分析，返回每个层的详细结果
# fit='PL' 表示使用幂律拟合（也可以使用 'TPL' 截断幂律，默认是 'PL'）
details = watcher.analyze(fit='PL', plot=False)  # plot=True 会为每层绘制 ESD 图，数据多时慎用

# ==================== 4. 查看结果 ====================
# 4.1 详细层信息
print("=== 各层分析结果（前5行）===")
print(details.head())

# 4.2 查看所有层的 alpha 值
print("\n=== 各层的 alpha 值 ===")
print(details['alpha'])

# 4.3 获取整个模型的汇总指标
summary = watcher.get_summary()
print("\n=== 模型汇总指标 ===")
print(summary)

# ==================== 5. 可视化分析 ====================
# 5.1 绘制 alpha 的分布直方图
plt.figure(figsize=(8, 5))
details['alpha'].plot.hist(bins=50, color='green', alpha=0.7, density=True)
plt.xlabel(r"Power Law Exponent $\alpha$", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.title(r'Distribution of $\alpha$ across Layers', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 5.2 如果需要，也可以绘制 alpha 随层 id 的变化
plt.figure(figsize=(10, 5))
# 按层顺序绘制，假设 details 是按层顺序排列的
plt.plot(details.index, details['alpha'], marker='o', linestyle='-', markersize=3)
plt.xlabel("Layer ID (index)", fontsize=12)
plt.ylabel(r"$\alpha$", fontsize=12)
plt.title(r'$\alpha$ vs Layer ID', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ==================== 6. 简单解读 ====================
print("\n=== 指标解读参考 ===")
print("- 理想的 α 通常在 2~4 之间，表示良好的多尺度相关性。")
print("- α > 6 可能表示该层训练不足或过参数化。")
print("- α < 2 可能表示该层过度训练或过拟合。")
print("- 平均 α 越小（接近2），通常模型质量越好。")