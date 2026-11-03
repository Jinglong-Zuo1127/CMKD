import numpy as np
import torch
from tqdm import tqdm


class MedEvaluator:

    @staticmethod
    def _compute_stats(model, data_loader, device):


        num_classes = [9, 4, 11, 8, 2, 7, 11, 11, 2] 
        tensor_num_classes = torch.tensor(num_classes)
        tensor_num_classes_cumsum = tensor_num_classes.cumsum(dim=0).tolist()
        tensor_num_classes_cumsum = [0]+tensor_num_classes_cumsum[:-1]

        correct_list, total_list = torch.tensor([0]*9), torch.tensor([0]*9)

        pbar = tqdm(total=len(data_loader))
        for data in data_loader:
            input = data[0]
            task_label = torch.argmax(data[2], dim=1)

            if isinstance(model, torch.nn.DataParallel):
                # hrepr = model.module.encoder(input.to(device))
                hrepr, attn_weights, proj_feat, proj_token = model.module.encoder(input.to(device))
            else:
                # hrepr, attn_weights, proj_feat, proj_token = model.encoder(input.to(device))
                hrepr = model.encoder(input.to(device))
            avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            if isinstance(hrepr, list):
                hrepr = [avgpool(each_hrepr).squeeze() for each_hrepr in hrepr]
            else:
                hrepr = avgpool(hrepr).squeeze()

            for th in range(9):
                label_offset = tensor_num_classes_cumsum[th]

                if hrepr[task_label==th].shape[0] == 0:
                    continue
                if isinstance(model, torch.nn.DataParallel):
                    output = model.module.decoders[str(th)](hrepr[task_label==th])
                else:
                    output = model.decoders[str(th)](hrepr[task_label==th])
                
                correct_list[th] = correct_list[th] + torch.argmax(output, 1).eq((data[1].cuda()[task_label==th]-label_offset).squeeze()).float().sum().item()
                total_list[th] = total_list[th] + output.shape[0]

            pbar.update(1)

        pbar.clear()
        pbar.close()
        del pbar

        acc0 = correct_list[0]/ total_list[0]
        acc1 = correct_list[1]/ total_list[1]
        acc2 = correct_list[2]/ total_list[2]
        acc3 = correct_list[3]/ total_list[3]
        acc4 = correct_list[4]/ total_list[4]
        acc5 = correct_list[5]/ total_list[5]
        acc6 = correct_list[6]/ total_list[6]
        acc7 = correct_list[7]/ total_list[7]
        acc8 = correct_list[8]/ total_list[8]

        return acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8



    @staticmethod
    def _compute_stats_task(model, data_loader, device):

        num_classes = [9, 4, 11, 8, 2, 7, 11, 11, 2]
        tensor_num_classes = torch.tensor(num_classes)
        tensor_num_classes_cumsum = tensor_num_classes.cumsum(dim=0).tolist() 
        tensor_num_classes_cumsum = [0]+tensor_num_classes_cumsum[:-1]

        correct_list, total_list = torch.tensor([0]*9), torch.tensor([0]*9)

        pbar = tqdm(total=len(data_loader))
        for data in data_loader:
            input = data[0]
            task_label = torch.argmax(data[2], dim=1)

            for th in range(9):
                label_offset = tensor_num_classes_cumsum[th]

                if isinstance(model, torch.nn.DataParallel):
                    hrepr, attn_weights, proj_feat, proj_token = model.module.encoder(input.to(device), task_label=data[2].cuda())
                else:
                    hrepr, attn_weights, proj_feat, proj_token = model.encoder(input.to(device), task_label=data[2].cuda())
                    
                avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
                if isinstance(hrepr, list):
                    hrepr = [avgpool(each_hrepr).squeeze() for each_hrepr in hrepr]
                else:
                    hrepr = avgpool(hrepr).squeeze()
                if hrepr[task_label==th].shape[0] == 0:
                    continue

                if isinstance(model, torch.nn.DataParallel):
                    output = model.module.decoders[str(th)](hrepr[task_label==th])
                else:
                    output = model.decoders[str(th)](hrepr[task_label==th])

                correct_list[th] = correct_list[th] + torch.argmax(output, 1).eq((data[1].cuda()[task_label==th]-label_offset).squeeze()).float().sum().item()
                total_list[th] = total_list[th] + output.shape[0]

            pbar.update(1)

        pbar.clear()
        pbar.close()
        del pbar

        acc0 = correct_list[0]/ total_list[0]
        acc1 = correct_list[1]/ total_list[1]
        acc2 = correct_list[2]/ total_list[2]
        acc3 = correct_list[3]/ total_list[3]
        acc4 = correct_list[4]/ total_list[4]
        acc5 = correct_list[5]/ total_list[5]
        acc6 = correct_list[6]/ total_list[6]
        acc7 = correct_list[7]/ total_list[7]
        acc8 = correct_list[8]/ total_list[8]

        return acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8



    @staticmethod
    def _compute_stats_each_task(model, data_loader, device):

        num_classes = [9, 4, 11, 8, 2, 7, 11, 11, 2]
        tensor_num_classes = torch.tensor(num_classes)
        tensor_num_classes_cumsum = tensor_num_classes.cumsum(dim=0).tolist()
        tensor_num_classes_cumsum = [0]+tensor_num_classes_cumsum[:-1]

        correct_list, total_list = torch.tensor([0]*9), torch.tensor([0]*9)

        pbar = tqdm(total=len(data_loader))
        for data in data_loader:
            input = data[0]
            task_label = torch.argmax(data[2], dim=1)
            if isinstance(model, torch.nn.DataParallel):
                hrepr, attn_weights, proj_feat, proj_token = model.module.encoder(input.to(device))
            else:
                # hrepr, attn_weights, proj_feat, proj_token = model.encoder(input.to(device))
                hrepr = model.encoder(input.to(device))
            avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            if isinstance(hrepr, list):
                hrepr = [avgpool(each_hrepr).squeeze() for each_hrepr in hrepr]
            else:
                hrepr = avgpool(hrepr).squeeze()
            for th in range(9):
                label_offset = tensor_num_classes_cumsum[th]
                if hrepr[th][task_label==th].shape[0] == 0:
                    continue

                if isinstance(model, torch.nn.DataParallel):
                    output = model.module.decoders[str(th)](hrepr[th][task_label==th])
                else:
                    output = model.decoders[str(th)](hrepr[th][task_label==th])

                correct_list[th] = correct_list[th] + torch.argmax(output, 1).eq((data[1].cuda()[task_label==th]-label_offset).squeeze()).float().sum().item()
                total_list[th] = total_list[th] + output.shape[0]

            pbar.update(1)

        pbar.clear()
        pbar.close()
        del pbar

        acc0 = correct_list[0]/ total_list[0]
        acc1 = correct_list[1]/ total_list[1]
        acc2 = correct_list[2]/ total_list[2]
        acc3 = correct_list[3]/ total_list[3]
        acc4 = correct_list[4]/ total_list[4]
        acc5 = correct_list[5]/ total_list[5]
        acc6 = correct_list[6]/ total_list[6]
        acc7 = correct_list[7]/ total_list[7]
        acc8 = correct_list[8]/ total_list[8]

        return acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8



    @staticmethod
    def evaluate(model, data_loader, device):
        model.eval()
        print(f"Compute statistics")
        with torch.no_grad():
            (acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8
            ) = MedEvaluator._compute_stats(model, data_loader, device)
        
        avg_acc = (acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8)/9
        print(
            f"acc0: {acc0}, acc1: {acc1}, acc2: {acc2}, acc3: {acc3},   \
            acc4: {acc4}, acc5: {acc5}, acc6: {acc6}, acc7: {acc7},  \
            acc8: {acc8}, \
            avg_acc: {avg_acc}; "
        )
        return {'acc0': acc0, 'acc1': acc1, 'acc2': acc2, 'acc3': acc3, 'acc4': acc4, 'acc5': acc5,
            'acc6': acc6, 'acc7': acc7, 'acc8': acc8, 'avg_acc': avg_acc}

    @staticmethod
    def evaluate_task(model, data_loader, device):
        model.eval()
        print(f"Compute statistics")
        with torch.no_grad():
            (acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8
            ) = MedEvaluator._compute_stats_task(model, data_loader, device)

        avg_acc = (acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8)/9
        print(
            f"acc0: {acc0}, acc1: {acc1}, acc2: {acc2}, acc3: {acc3},   \
            acc4: {acc4}, acc5: {acc5}, acc6: {acc6}, acc7: {acc7},  \
            acc8: {acc8}, \
            avg_acc: {avg_acc}; "
        )
        return {'acc0': acc0, 'acc1': acc1, 'acc2': acc2, 'acc3': acc3, 'acc4': acc4, 'acc5': acc5,
            'acc6': acc6, 'acc7': acc7, 'acc8': acc8, 'avg_acc': avg_acc}

    @staticmethod
    def evaluate_each_task(model, data_loader, device):
        model.eval()
        print(f"Compute statistics")
        with torch.no_grad():
            (acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8
            ) = MedEvaluator._compute_stats_each_task(model, data_loader, device)

        avg_acc = (acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8)/9
        print(
            f"acc0: {acc0}, acc1: {acc1}, acc2: {acc2}, acc3: {acc3},   \
            acc4: {acc4}, acc5: {acc5}, acc6: {acc6}, acc7: {acc7},  \
            acc8: {acc8}, \
            avg_acc: {avg_acc}; "
        )
        return {'acc0': acc0, 'acc1': acc1, 'acc2': acc2, 'acc3': acc3, 'acc4': acc4, 'acc5': acc5,
            'acc6': acc6, 'acc7': acc7, 'acc8': acc8, 'avg_acc': avg_acc}

    @staticmethod
    def evaluate_clip_teacher(teacher_model, data_loader, device='cuda'):
        """评估CLIP教师模型在MedMNIST数据集上的性能

        参数:
            teacher_model: 已初始化的ClipTeacherModel实例
            data_loader: MedMNIST数据加载器
            device: 计算设备
        """
        # 设置模型为评估模式
        teacher_model.eval()

        # 初始化统计变量
        num_classes = [9, 4, 11, 8, 2, 7, 11, 11, 2]
        tensor_num_classes = torch.tensor(num_classes)
        tensor_num_classes_cumsum = tensor_num_classes.cumsum(dim=0).tolist()
        tensor_num_classes_cumsum = [0] + tensor_num_classes_cumsum[:-1]

        # 为每个任务存储预测结果和标签
        all_preds = {i: [] for i in range(9)}
        all_labels = {i: [] for i in range(9)}

        # 开始评估
        print(f"Evaluating CLIP teacher model on MedMNIST")

        pbar = tqdm(total=len(data_loader))
        with torch.no_grad():
            for data in data_loader:
                input_images = data[0].to(device)
                labels = data[1].cpu().numpy()
                task_label = torch.argmax(data[2], dim=1).cpu().numpy()

                # 获取教师模型预测的所有任务logits
                task_logits_dict = teacher_model(input_images)

                # 处理每个任务
                for th in range(9):
                    task_mask = (task_label == th)
                    if not task_mask.any():
                        continue

                    # 获取当前任务的相关数据
                    task_labels = labels[task_mask]

                    # 获取该任务的logits
                    task_logits = task_logits_dict[th][task_mask]

                    # 计算预测结果
                    preds = torch.argmax(task_logits, dim=1).cpu().numpy()

                    # 标签偏移处理
                    label_offset = tensor_num_classes_cumsum[th]
                    local_labels = task_labels - label_offset

                    # 存储预测结果和标签
                    all_preds[th].extend(preds)
                    all_labels[th].extend(local_labels)

                pbar.update(1)

        pbar.clear()
        pbar.close()

        # 计算各任务准确率
        acc_list = []
        for th in range(9):
            if len(all_preds[th]) == 0:
                acc_list.append(0.0)
                continue

            # 转换为NumPy数组进行高效计算
            preds = np.array(all_preds[th])
            labels = np.array(all_labels[th])

            # 计算准确率
            acc_value = (preds == labels).mean()
            acc_list.append(acc_value)

        # 计算平均准确率
        avg_acc = np.mean(acc_list)

        # 打印结果
        print(
            f"CLIP Teacher Performance: "
            f"acc0: {acc_list[0]:.4f} ({acc_list[0] * 100:.2f}%), "
            f"acc1: {acc_list[1]:.4f} ({acc_list[1] * 100:.2f}%), "
            f"acc2: {acc_list[2]:.4f} ({acc_list[2] * 100:.2f}%), "
            f"acc3: {acc_list[3]:.4f} ({acc_list[3] * 100:.2f}%), "
            f"acc4: {acc_list[4]:.4f} ({acc_list[4] * 100:.2f}%), "
            f"acc5: {acc_list[5]:.4f} ({acc_list[5] * 100:.2f}%), "
            f"acc6: {acc_list[6]:.4f} ({acc_list[6] * 100:.2f}%), "
            f"acc7: {acc_list[7]:.4f} ({acc_list[7] * 100:.2f}%), "
            f"acc8: {acc_list[8]:.4f} ({acc_list[8] * 100:.2f}%), "
            f"avg_acc: {avg_acc:.4f} ({avg_acc * 100:.2f}%)"
        )

        # 返回小数形式的准确率
        return {
            'acc0': float(acc_list[0]),
            'acc1': float(acc_list[1]),
            'acc2': float(acc_list[2]),
            'acc3': float(acc_list[3]),
            'acc4': float(acc_list[4]),
            'acc5': float(acc_list[5]),
            'acc6': float(acc_list[6]),
            'acc7': float(acc_list[7]),
            'acc8': float(acc_list[8]),
            'avg_acc': float(avg_acc)
        }

    @staticmethod
    def evaluate_finetuned_medclip(model, data_loader, device='cuda'):
        """
        评估微调后的MedCLIP模型在MedMNIST数据集上的性能

        参数:
            model: MedCLIPFinetuner实例
            data_loader: MedMNIST数据加载器
            device: 计算设备
        """
        model.eval()

        # 获取任务信息
        label_offsets = model.task_info['label_offsets']
        num_classes = model.task_info['num_classes']
        task_names = model.task_info.get('task_names', [f'task_{i}' for i in range(9)])

        # 初始化统计变量
        correct_list = [0] * 9
        total_list = [0] * 9

        pbar = tqdm(total=len(data_loader), desc="Evaluating MedCLIPFinetuner")
        with torch.no_grad():
            for data in data_loader:
                images = data[0].to(device)
                labels = data[1].squeeze().cpu().numpy()  # [batch_size]
                task_label = torch.argmax(data[2], dim=1).cpu().numpy()  # [batch_size]

                # 获取模型预测
                task_logits, _, _ = model(images)

                for task_id in range(9):
                    # 筛选当前任务的数据
                    task_mask = (task_label == task_id)
                    if not task_mask.any():
                        continue

                    # 获取当前任务的logits
                    logits = task_logits[task_id][task_mask].cpu().numpy()
                    preds = np.argmax(logits, axis=1)

                    # 应用标签偏移
                    local_labels = labels[task_mask] - label_offsets[task_id]

                    # 统计正确预测
                    correct_list[task_id] += (preds == local_labels).sum()
                    total_list[task_id] += task_mask.sum()

                pbar.update(1)

        pbar.close()

        # 计算各任务准确率
        acc_list = []
        for i in range(9):
            if total_list[i] > 0:
                acc = correct_list[i] / total_list[i]
                acc_list.append(acc)
            else:
                acc_list.append(0.0)

        # 计算平均准确率
        avg_acc = np.mean(acc_list)

        # 构建结果字典（与之前格式兼容）
        results = {
            'acc0': float(acc_list[0]),
            'acc1': float(acc_list[1]),
            'acc2': float(acc_list[2]),
            'acc3': float(acc_list[3]),
            'acc4': float(acc_list[4]),
            'acc5': float(acc_list[5]),
            'acc6': float(acc_list[6]),
            'acc7': float(acc_list[7]),
            'acc8': float(acc_list[8]),
            'avg_acc': float(avg_acc)
        }

        # 打印结果
        print("\nMedCLIPFinetuner Performance:")
        for i, acc in enumerate(acc_list):
            print(f"{task_names[i]}: Acc = {acc:.4f} ({acc * 100:.2f}%)")
        print(f"Average Accuracy: {avg_acc:.4f} ({avg_acc * 100:.2f}%)\n")

        return results

