import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from CLIP import clip
from ClipTeacher_medmnist import ClipTeacherModel
from code.data.datasets.medmnist import MedMnistData


class MedCLIPFinetuner(nn.Module):
    """医学领域CLIP微调模型"""

    def __init__(self, clip_model, task_info, device='cuda', adapter_dim=128):
        """
        Args:
            clip_model: 预加载的CLIP模型
            task_info: 任务信息字典 (包含label_offsets)
            device: 计算设备
            adapter_dim: 适配层维度
        """
        super().__init__()
        self.clip = clip_model
        self.device = device
        self.task_info = task_info  # 存储完整任务信息
        self.adapter_dim = adapter_dim

        # 冻结CLIP原始参数
        for param in self.clip.parameters():
            param.requires_grad = False

        # 添加适配层
        self.image_adapter = nn.Sequential(
            nn.Linear(512, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, 512)
        ).to(device)

        self.text_adapter = nn.ModuleDict()
        for task_id, class_dict in task_info['task_labels'].items():
            self.text_adapter[str(task_id)] = nn.Sequential(
                nn.Linear(512, adapter_dim),
                nn.GELU(),
                nn.Linear(adapter_dim, 512)
            ).to(device)

        # 分类头
        self.classifiers = nn.ModuleDict()
        for task_id, num_classes in task_info['num_classes'].items():
            self.classifiers[str(task_id)] = nn.Linear(512, num_classes).to(device)

    def forward(self, images, task_ids=None):
        """
        前向传播
        Args:
            images: 输入图像 (B, C, H, W)
            task_ids: 任务ID张量 (B,)
        Returns:
            task_logits: 各任务分类logits
            image_features: 适配后的图像特征
            text_features: 适配后的文本特征
        """
        # 原始CLIP特征提取
        with torch.no_grad():
            # 添加类型转换：half -> float
            image_features, attn_weights, cls_token, feats = self.clip.encode_image(images)
            image_features = image_features.float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 图像适配
        adapted_image_features = self.image_adapter(image_features)
        adapted_image_features = adapted_image_features / adapted_image_features.norm(dim=-1, keepdim=True)

        # 文本适配和分类
        task_logits = {}
        text_features_dict = {}

        # 如果没有指定任务ID，为所有任务生成输出
        target_tasks = range(9) if task_ids is None else torch.unique(task_ids)

        for task_id in target_tasks:
            task_id = int(task_id)
            text_inputs = self.task_info['text_inputs'][task_id]

            # 文本适配
            with torch.no_grad():
                # 添加类型转换：half -> float
                task_text_features = self.clip.encode_text(text_inputs).float()
                task_text_features = task_text_features / task_text_features.norm(dim=-1, keepdim=True)

            adapted_text_features = self.text_adapter[str(task_id)](task_text_features)
            adapted_text_features = adapted_text_features / adapted_text_features.norm(dim=-1, keepdim=True)
            text_features_dict[task_id] = adapted_text_features

            # 分类logits
            logit_scale = self.clip.logit_scale.exp().float()  # 确保logit_scale为float32
            task_logits[task_id] = logit_scale * (adapted_image_features @ adapted_text_features.t())

            # 添加线性分类头
            classifier_logits = self.classifiers[str(task_id)](adapted_image_features)
            task_logits[task_id] = task_logits[task_id] + classifier_logits

        return task_logits, adapted_image_features, text_features_dict,attn_weights, cls_token, feats


def fine_tune_clip_medical(
        teacher_model,
        train_loader,
        val_loader,
        device='cuda',
        epochs=20,
        lr=1e-4,
        alpha=0.7
):
    """
    CLIP医学微调函数

    Args:
        teacher_model: 预训练的ClipTeacherModel
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 计算设备
        epochs: 训练轮数
        lr: 学习率
        alpha: 对比损失权重
    """
    # 计算每个任务的类别数
    num_classes = [len(teacher_model.task_labels[i]) for i in range(9)]
    # 计算标签偏移量：累加前N-1个任务的类别数
    label_offsets = torch.cumsum(torch.tensor([0] + num_classes[:-1]), dim=0).tolist()
    task_names = ['pathmnist', 'octmnist', 'organcmnist', 'bloodmnist',
                  'breastmnist', 'dermamnist', 'organamnist',
                  'organsmnist', 'pneumoniamnist']
    # 准备任务信息（包含标签偏移量）
    task_info = {
        'task_labels': teacher_model.task_labels,
        'text_inputs': teacher_model.text_inputs,
        'num_classes': {i: num_classes[i] for i in range(9)},
        'label_offsets': label_offsets,  # 添加标签偏移量
        'task_names':task_names
    }

    # 初始化微调模型
    model = MedCLIPFinetuner(teacher_model.model, task_info, device)
    model.train()

    # 优化器
    optimizer = optim.AdamW([
        {'params': model.image_adapter.parameters()},
        {'params': model.text_adapter.parameters()},
        {'params': model.classifiers.parameters()}
    ], lr=lr)

    # 损失函数
    contrastive_loss = nn.CrossEntropyLoss()
    classifier_loss = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    # 添加混合精度训练
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        contrast_loss_sum = 0.0
        class_loss_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for images, labels, task_labels in pbar:
            # 关键修复：压缩标签维度 [batch, 1] -> [batch]
            images = images.to(device)
            labels = labels.squeeze().to(device)  # 压缩标签维度
            task_labels = torch.argmax(task_labels, dim=1).to(device)

            optimizer.zero_grad()

            # 使用混合精度训练
            with torch.cuda.amp.autocast():
                # 前向传播
                task_logits, image_features, text_features = model(images, task_labels)

                # 计算损失
                batch_contrast_loss = 0.0
                batch_class_loss = 0.0

                for task_id in torch.unique(task_labels):
                    task_id = int(task_id)
                    task_mask = (task_labels == task_id)

                    if task_mask.sum() == 0:
                        continue

                    # 获取当前任务的标签偏移量
                    offset = model.task_info['label_offsets'][task_id]

                    # 对比学习损失
                    logits = task_logits[task_id][task_mask]
                    local_labels = (labels[task_mask] - offset).long()  # 应用标签偏移

                    # 对比损失
                    contrast_loss_val = contrastive_loss(logits, local_labels)

                    # 分类损失
                    classifier_logits = model.classifiers[str(task_id)](image_features[task_mask])
                    class_loss_val = classifier_loss(classifier_logits, local_labels)

                    batch_contrast_loss += contrast_loss_val
                    batch_class_loss += class_loss_val

                # 加权总损失
                loss = alpha * batch_contrast_loss + (1 - alpha) * batch_class_loss

            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            contrast_loss_sum += batch_contrast_loss.item()
            class_loss_sum += batch_class_loss.item()

            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'contrast': contrast_loss_sum / (pbar.n + 1),
                'class': class_loss_sum / (pbar.n + 1)
            })

        # 验证阶段
        model.eval()
        val_results = evaluate_finetuned_clip(model, val_loader, device)
        val_acc = val_results['avg_acc']

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {total_loss / len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_finetuned.pth')
            print(f"Saved best model with acc: {best_val_acc:.4f}")

    print(f"Fine-tuning complete. Best validation acc: {best_val_acc:.4f}")
    return model


def evaluate_finetuned_clip(model, data_loader, device='cuda'):
    """评估微调后的CLIP模型"""
    model.eval()
    # 直接从模型的任务信息中获取标签偏移量
    label_offsets = model.task_info['label_offsets']
    num_classes = model.task_info['num_classes']

    # 初始化统计
    correct_list = [0] * 9
    total_list = [0] * 9

    with torch.no_grad():
        for images, labels, task_labels in tqdm(data_loader, desc="Evaluating"):
            # 关键修复：压缩标签维度 [batch, 1] -> [batch]
            images = images.to(device)
            labels = labels.squeeze().cpu().numpy()  # 压缩标签维度
            task_labels = torch.argmax(task_labels, dim=1).cpu().numpy()

            # 获取预测
            task_logits, _, _ = model(images)

            # 处理每个任务
            for task_id in range(9):
                task_mask = (task_labels == task_id)
                if not task_mask.any():
                    continue

                # 获取预测结果
                logits = task_logits[task_id][task_mask].cpu().numpy()
                preds = np.argmax(logits, axis=1)

                # 计算正确率（使用标签偏移量）
                task_labels_local = labels[task_mask] - label_offsets[task_id]
                correct = (preds == task_labels_local).sum()

                correct_list[task_id] += correct
                total_list[task_id] += task_mask.sum()

    # 计算准确率
    acc_list = []
    for i in range(9):
        if total_list[i] > 0:
            acc = correct_list[i] / total_list[i]
            acc_list.append(acc)
        else:
            acc_list.append(0.0)

    avg_acc = np.mean(acc_list)

    # 打印结果
    print("\nFine-tuned CLIP Performance:")
    task_names = model.task_info['task_names']
    for i, acc in enumerate(acc_list):
        print(f"Task {i} ({task_names[i]}): Acc = {acc:.4f} ({acc * 100:.2f}%)")
    # for i, acc in enumerate(acc_list):
    #     task_name = list(model.task_info['task_labels'][i].values())[0] if model.task_info['task_labels'][
    #         i] else f"Task {i}"
    #     print(f"Task {i} ({task_name}): Acc = {acc:.4f} ({acc * 100:.2f}%)")

    print(f"Average Accuracy: {avg_acc:.4f} ({avg_acc * 100:.2f}%)\n")

    return {
        'acc_list': acc_list,
        'avg_acc': avg_acc
    }


# 使用示例
if __name__ == "__main__":
    # 初始化设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载预训练教师模型
    teacher = ClipTeacherModel(device=device)

    # 准备数据集
    train_dataset = MedMnistData(split='train')
    val_dataset = MedMnistData(split='val')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # 微调CLIP
    finetuned_model = fine_tune_clip_medical(
        teacher_model=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=50,
        lr=3e-5,
        alpha=0.7
    )

    # 保存最终模型
    torch.save(finetuned_model.state_dict(), "medclip_finetuned_final.pth")
    print("Final model saved as medclip_finetuned_final.pth")