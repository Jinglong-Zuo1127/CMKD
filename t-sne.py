import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# 导入你项目中的自定义模块（根据实际路径调整）
from code.benchmarks.mtl_benchmark import get_benchmark_class
from code.data.datasets.my_vlcs import VLCS_MultiDomain  # 注意路径可能不同
from code.benchmarks.vlcs import VLCSModel_LoRASoftRouter, VLCSBenchmark   # 或直接从模型定义文件导入
import code.utils.utils as utils
from layers.router import build_router
def extract_features(model, loader, device):
    """从验证集/测试集中提取特征和任务标签"""
    model.eval()
    features = []
    task_labels = []

    with torch.no_grad():
        for data in loader:
            images = data[0].to(device)
            task_y = data[2].to(device)          # one-hot 向量

            # 前向传播，获取特征
            # VLCSModel_LoRASoftRouter 的 forward 返回：
            # x4, [attn_qk,attn_vv], vit_feat, cls_proj, x3
            # 我们取 cls_proj 作为特征（768维）
            x, _, _, cls_proj, _ = model.encoder(images, return_scores=False)
            pooled = torch.mean(x, dim=[2, 3])
            features.append(pooled.cpu().numpy())

            # 将 one-hot 转换为类别索引
            task_labels.append(torch.argmax(task_y, dim=1).cpu().numpy())

    features = np.concatenate(features, axis=0)
    task_labels = np.concatenate(task_labels, axis=0)
    return features, task_labels

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---------- 配置参数（需与训练时一致）----------
    parser = utils.common_argparser()
    args, _ = parser.parse_known_args()

    specific_parser = VLCSBenchmark.get_arg_parser(parser)

    args = specific_parser.parse_args()
    model = VLCSModel_LoRASoftRouter(args)  # 创建模型实例
    router = build_router(num_experts=args.n_expert)
    model.router = router
    for module in model.modules():
        if hasattr(module, 'router'):
            module.router = router
    model = model.to(device)

    # 加载训练好的权重
    checkpoint_path = '/home/zuojinlong/code/MoLA/logs/vlcs_resnet50/ourbase_vlcs/lora_soft_router/22/best_test.pth'   # 替换为实际路径
    state_dict = torch.load(checkpoint_path, map_location=device)['state_dict']
    model.load_state_dict(state_dict)
    print("模型加载成功！")

    # ---------- 准备数据集 ----------
    # 使用测试集（split='test'），预处理应与训练时测试阶段一致
    dataset = VLCS_MultiDomain(root_path='/home/zuojinlong/code/MoLA/code/data/datasets/VLCS', split='test')
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # ---------- 提取特征 ----------
    features, task_labels = extract_features(model, loader, device)
    print(f"特征形状: {features.shape}, 标签形状: {task_labels.shape}")

    # （可选）如果样本太多，随机采样一部分
    # sample_indices = np.random.choice(len(features), size=5000, replace=False)
    # features = features[sample_indices]
    # task_labels = task_labels[sample_indices]

    # ---------- 标准化（t-SNE 前可选）----------
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # ---------- 运行 t-SNE ----------
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
    features_2d = tsne.fit_transform(features_scaled)

    # ---------- 可视化 ----------
    plt.figure(figsize=(10, 8))
    task_names = ['V (PASCAL)', 'L (LABELME)', 'C (CALTECH)', 'S (SUN)']
    colors = plt.cm.tab10(np.linspace(0, 1, 4))

    for i in range(4):
        mask = (task_labels == i)
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                    c=[colors[i]], label=task_names[i], s=5, alpha=0.6)

    plt.legend()
    plt.title('t-SNE visualization of VLCS task features (cls_proj)')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.tight_layout()
    plt.savefig('tsne_vlcs.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    main()