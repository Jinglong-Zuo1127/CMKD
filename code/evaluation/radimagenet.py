import numpy as np
import torch
from tqdm import tqdm

class RadEvaluator:


    @staticmethod
    def _compute_stats(model, data_loader, device):


        num_classes = [6,28,2,13,18,14,9,25,26,10,14]
        tensor_num_classes = torch.tensor(num_classes)
        tensor_num_classes_cumsum = tensor_num_classes.cumsum(dim=0).tolist()
        tensor_num_classes_cumsum = [0]+tensor_num_classes_cumsum[:-1]

        correct_list, total_list = torch.tensor([0]*11), torch.tensor([0]*11)

        pbar = tqdm(total=len(data_loader))
        for data in data_loader:
            input = data[0]
            task_label = torch.argmax(data[2], dim=1)
            if isinstance(model, torch.nn.DataParallel):
                hrepr = model.module.encoder(input.to(device))
            else:
                hrepr = model.encoder(input.to(device))
            avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            if isinstance(hrepr, list):
                hrepr = [avgpool(each_hrepr).squeeze() for each_hrepr in hrepr]
            else:
                hrepr = avgpool(hrepr).squeeze()

            for th in range(11):
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
        acc9 = correct_list[9]/ total_list[9]
        acc10 = correct_list[10]/ total_list[10]

        return acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10



    @staticmethod
    def _compute_stats_task(model, data_loader, device):

        num_classes = [6,28,2,13,18,14,9,25,26,10,14]
        tensor_num_classes = torch.tensor(num_classes)
        tensor_num_classes_cumsum = tensor_num_classes.cumsum(dim=0).tolist()
        tensor_num_classes_cumsum = [0]+tensor_num_classes_cumsum[:-1]

        correct_list, total_list = torch.tensor([0]*11), torch.tensor([0]*11)

        pbar = tqdm(total=len(data_loader))
        for data in data_loader:
            input = data[0]
            task_label = torch.argmax(data[2], dim=1)

            for th in range(11):
                label_offset = tensor_num_classes_cumsum[th]

                if isinstance(model, torch.nn.DataParallel):
                    hrepr = model.module.encoder(input.to(device), task_label=data[2].cuda())
                else:
                    hrepr = model.encoder(input.to(device), task_label=data[2].cuda())
                    
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
        acc9 = correct_list[9]/ total_list[9]
        acc10 = correct_list[10]/ total_list[10]

        return acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10



    @staticmethod
    def _compute_stats_each_task(model, data_loader, device):

        num_classes = [6,28,2,13,18,14,9,25,26,10,14]
        tensor_num_classes = torch.tensor(num_classes)
        tensor_num_classes_cumsum = tensor_num_classes.cumsum(dim=0).tolist()
        tensor_num_classes_cumsum = [0]+tensor_num_classes_cumsum[:-1]

        correct_list, total_list = torch.tensor([0]*11), torch.tensor([0]*11)

        pbar = tqdm(total=len(data_loader))
        for data in data_loader:
            input = data[0]
            task_label = torch.argmax(data[2], dim=1)
            if isinstance(model, torch.nn.DataParallel):
                hrepr = model.module.encoder(input.to(device))
            else:
                hrepr = model.encoder(input.to(device))
            avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            if isinstance(hrepr, list):
                hrepr = [avgpool(each_hrepr).squeeze() for each_hrepr in hrepr]
            else:
                hrepr = avgpool(hrepr).squeeze()
            for th in range(11):
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
        acc9 = correct_list[9]/ total_list[9]
        acc10 = correct_list[10]/ total_list[10]

        return acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10



    @staticmethod
    def evaluate(model, data_loader, device):

        model.eval()
        print(f"Compute statistics")
        with torch.no_grad():
            (acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10
            ) = RadEvaluator._compute_stats(model, data_loader, device)
        
        avg_acc = (acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8 + acc9 + acc10)/11
        print(
            f"acc0: {acc0}, acc1: {acc1}, acc2: {acc2}, acc3: {acc3},   \
            acc4: {acc4}, acc5: {acc5}, acc6: {acc6}, acc7: {acc7},  \
            acc8: {acc8}, acc9: {acc9}, acc10: {acc10}, \
            avg_acc: {avg_acc}; "
        )
        return {'acc0': acc0, 'acc1': acc1, 'acc2': acc2, 'acc3': acc3, 'acc4': acc4, 'acc5': acc5,
            'acc6': acc6, 'acc7': acc7, 'acc8': acc8, 'acc9': acc9, 'acc10': acc10, 'avg_acc': avg_acc}

    @staticmethod
    def evaluate_task(model, data_loader, device):
        model.eval()
        print(f"Compute statistics")
        with torch.no_grad():
            (acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10
            ) = RadEvaluator._compute_stats_task(model, data_loader, device)

        avg_acc = (acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8 + acc9 + acc10)/11
        print(
            f"acc0: {acc0}, acc1: {acc1}, acc2: {acc2}, acc3: {acc3},   \
            acc4: {acc4}, acc5: {acc5}, acc6: {acc6}, acc7: {acc7},  \
            acc8: {acc8}, acc9: {acc9}, acc10: {acc10}, \
            avg_acc: {avg_acc}; "
        )
        return {'acc0': acc0, 'acc1': acc1, 'acc2': acc2, 'acc3': acc3, 'acc4': acc4, 'acc5': acc5,
            'acc6': acc6, 'acc7': acc7, 'acc8': acc8, 'acc9': acc9, 'acc10': acc10, 'avg_acc': avg_acc}

    @staticmethod
    def evaluate_each_task(model, data_loader, device):
        model.eval()
        print(f"Compute statistics")
        with torch.no_grad():
            (acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10
            ) = RadEvaluator._compute_stats_each_task(model, data_loader, device)

        avg_acc = (acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8 + acc9 + acc10)/11
        print(
            f"acc0: {acc0}, acc1: {acc1}, acc2: {acc2}, acc3: {acc3},   \
            acc4: {acc4}, acc5: {acc5}, acc6: {acc6}, acc7: {acc7},  \
            acc8: {acc8}, acc9: {acc9}, acc10: {acc10}, \
            avg_acc: {avg_acc}; "
        )
        return {'acc0': acc0, 'acc1': acc1, 'acc2': acc2, 'acc3': acc3, 'acc4': acc4, 'acc5': acc5,
            'acc6': acc6, 'acc7': acc7, 'acc8': acc8, 'acc9': acc9, 'acc10': acc10, 'avg_acc': avg_acc}

    @staticmethod
    def evaluate_clip_teacher(teacher_model, data_loader, device):
        """
        评估CLIP教师模型在RadImageNet任务上的性能
        Args:
            teacher_model: 初始化后的ClipTeacherModel实例
            data_loader: 包含图像、全局标签和任务标签的数据加载器
            device: 计算设备（如'cuda'或'cpu'）
        Returns:
            包含各任务准确率和平均准确率的字典
        """
        teacher_model.eval()
        print("Evaluating CLIP teacher model...")

        # 任务-类别范围映射（与OurBaseBalancer保持一致）
        task_to_class_ranges = {
            0: list(range(0,6)),    # 任务0：0-5类（6个类别）
            1: list(range(6,34)),   # 任务1：6-33类（28个类别）
            2: list(range(34,36)),  # 任务2：34-35类（2个类别）
            3: list(range(36,49)),  # 任务3：36-48类（13个类别）
            4: list(range(49,67)),  # 任务4：49-66类（18个类别）
            5: list(range(67,81)),  # 任务5：67-80类（14个类别）
            6: list(range(81,90)),  # 任务6：81-89类（9个类别）
            7: list(range(90,115)), # 任务7：90-114类（25个类别）
            8: list(range(115,141)),# 任务8：115-140类（26个类别）
            9: list(range(141,151)),# 任务9：141-150类（10个类别）
            10: list(range(151,165))# 任务10：151-164类（14个类别）
        }

        # 初始化统计变量
        correct_list = torch.zeros(11, device=device)
        total_list = torch.zeros(11, device=device)

        with torch.no_grad():
            pbar = tqdm(total=len(data_loader), desc="Teacher evaluation")
            for data in data_loader:
                # 解析数据：图像、全局标签（0-164）、任务标签（One-Hot）
                images = data[0].to(device)
                global_labels = data[1].to(device)  # 全局类别索引（0-164）
                task_targets = data[2].to(device)   # 任务One-Hot编码
                task_labels = torch.argmax(task_targets, dim=1)  # 任务索引（0-10）
                # print("global_labels", global_labels)
                # print("task_labels", task_labels)

                # 教师模型前向传播，获取165维全局logits
                teacher_logits, _, _, _ = teacher_model(images)

                # 按任务计算准确率
                for task_id in range(11):
                    # 筛选当前任务的样本
                    task_mask = (task_labels == task_id)
                    if not task_mask.any():
                        continue

                    # 获取当前任务的类别范围
                    class_indices = task_to_class_ranges[task_id]
                    class_offset = class_indices[0]  # 任务内类别起始索引（用于计算相对标签）

                    # 提取当前任务的logits和标签
                    task_logits = teacher_logits[task_mask][:, class_indices]
                    task_global_labels = global_labels[task_mask]
                    # 计算任务内相对标签（0到类别数-1）
                    task_relative_labels = task_global_labels - class_offset
                    # print("task_relative_labels:", task_relative_labels)

                    # 计算预测正确数
                    predictions = torch.argmax(task_logits, dim=1)
                    # print("task_logits:", task_logits)
                    # print("predictions:", predictions)
                    correct = (predictions == task_relative_labels).sum()

                    # 累计统计
                    correct_list[task_id] += correct
                    total_list[task_id] += task_mask.sum()

                pbar.update(1)
            pbar.close()

        # 计算各任务准确率
        acc_results = {}
        for i in range(11):
            acc = correct_list[i] / total_list[i] if total_list[i] > 0 else 0.0
            acc_results[f'acc{i}'] = acc.item()

        # 计算平均准确率
        acc_results['avg_acc'] = sum(acc_results[f'acc{i}'] for i in range(11)) / 11

        # 打印结果
        print("CLIP Teacher Evaluation Results:")
        print(f"acc0: {acc_results['acc0']:.4f}, acc1: {acc_results['acc1']:.4f}, "
              f"acc2: {acc_results['acc2']:.4f}, acc3: {acc_results['acc3']:.4f}, "
              f"acc4: {acc_results['acc4']:.4f}, acc5: {acc_results['acc5']:.4f}, "
              f"acc6: {acc_results['acc6']:.4f}, acc7: {acc_results['acc7']:.4f}, "
              f"acc8: {acc_results['acc8']:.4f}, acc9: {acc_results['acc9']:.4f}, "
              f"acc10: {acc_results['acc10']:.4f}, avg_acc: {acc_results['avg_acc']:.4f}")

        return acc_results

