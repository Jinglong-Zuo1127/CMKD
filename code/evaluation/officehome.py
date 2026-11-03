import numpy as np
import torch
from tqdm import tqdm


class OfficeHomeEvaluator:

    @staticmethod
    def unwrap_encoder_output(output):
        if isinstance(output, tuple):
            return output[0]
        return output
    @staticmethod
    def _compute_stats(model, data_loader, device):
        correct_list, total_list = torch.tensor([0]*4), torch.tensor([0]*4)

        pbar = tqdm(total=len(data_loader))
        for data in data_loader:
            input = data[0]
            task_label = torch.argmax(data[2], dim=1)
            # hrepr,_,_,_= model.encoder(input.to(device))
            encoder_out = model.encoder(input.to(device))
            hrepr = OfficeHomeEvaluator.unwrap_encoder_output(encoder_out)
            # hrepr, attn_weights, proj_feat, proj_token = model.encoder(input.to(device))
            avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            if isinstance(hrepr, list):
                hrepr = [avgpool(each_hrepr).squeeze() for each_hrepr in hrepr]
            else:
                hrepr = avgpool(hrepr).squeeze()
            for th in range(4):
                if hrepr[task_label==th].shape[0] == 0:
                    continue

                if th == 0:
                    v_output = model.decoders["P"](hrepr[task_label==0])
                    correct_list[0] = correct_list[0] + torch.argmax(v_output, 1).eq(data[1].cuda()[task_label==0].squeeze()).float().sum().item()
                    total_list[0] = total_list[0] + v_output.shape[0]
                if th == 1:
                    l_output = model.decoders["A"](hrepr[task_label==1])
                    correct_list[1] = correct_list[1] + torch.argmax(l_output, 1).eq(data[1].cuda()[task_label==1].squeeze()).float().sum().item()
                    total_list[1] = total_list[1] + l_output.shape[0]
                if th == 2:
                    c_output = model.decoders["C"](hrepr[task_label==2])
                    correct_list[2] = correct_list[2] + torch.argmax(c_output, 1).eq(data[1].cuda()[task_label==2].squeeze()).float().sum().item()
                    total_list[2] = total_list[2] + c_output.shape[0]
                if th == 3:
                    s_output = model.decoders["R"](hrepr[task_label==3])
                    correct_list[3] = correct_list[3] + torch.argmax(s_output, 1).eq(data[1].cuda()[task_label==3].squeeze()).float().sum().item()
                    total_list[3] = total_list[3] + s_output.shape[0]

            pbar.update(1)

        pbar.clear()
        pbar.close()
        del pbar

        v_acc = correct_list[0]/ total_list[0]
        l_acc = correct_list[1]/ total_list[1]
        c_acc = correct_list[2]/ total_list[2]
        s_acc = correct_list[3]/ total_list[3]

        return v_acc, l_acc, c_acc, s_acc



    @staticmethod
    def _compute_stats_task(model, data_loader, device):
        correct_list, total_list = torch.tensor([0]*4), torch.tensor([0]*4)

        pbar = tqdm(total=len(data_loader))
        for data in data_loader:
            input = data[0]
            task_label = torch.argmax(data[2], dim=1)
            
            for th in range(4):
                hrepr = model.encoder(input.to(device), task_label=data[2].cuda())
                avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
                if isinstance(hrepr, list):
                    hrepr = [avgpool(each_hrepr).squeeze() for each_hrepr in hrepr]
                else:
                    hrepr = avgpool(hrepr).squeeze()
                if hrepr[task_label==th].shape[0] == 0:
                    continue
                if th == 0:
                    v_output = model.decoders["P"](hrepr[task_label==0])
                    correct_list[0] = correct_list[0] + torch.argmax(v_output, 1).eq(data[1].cuda()[task_label==0].squeeze()).float().sum().item()
                    total_list[0] = total_list[0] + v_output.shape[0]
                if th == 1:
                    l_output = model.decoders["A"](hrepr[task_label==1])
                    correct_list[1] = correct_list[1] + torch.argmax(l_output, 1).eq(data[1].cuda()[task_label==1].squeeze()).float().sum().item()
                    total_list[1] = total_list[1] + l_output.shape[0]
                if th == 2:
                    c_output = model.decoders["C"](hrepr[task_label==2])
                    correct_list[2] = correct_list[2] + torch.argmax(c_output, 1).eq(data[1].cuda()[task_label==2].squeeze()).float().sum().item()
                    total_list[2] = total_list[2] + c_output.shape[0]
                if th == 3:
                    s_output = model.decoders["R"](hrepr[task_label==3])
                    correct_list[3] = correct_list[3] + torch.argmax(s_output, 1).eq(data[1].cuda()[task_label==3].squeeze()).float().sum().item()
                    total_list[3] = total_list[3] + s_output.shape[0]

            pbar.update(1)

        pbar.clear()
        pbar.close()
        del pbar

        v_acc = correct_list[0]/ total_list[0]
        l_acc = correct_list[1]/ total_list[1]
        c_acc = correct_list[2]/ total_list[2]
        s_acc = correct_list[3]/ total_list[3]

        return v_acc, l_acc, c_acc, s_acc



    @staticmethod
    def _compute_stats_each_task(model, data_loader, device):
        correct_list, total_list = torch.tensor([0]*4), torch.tensor([0]*4)

        pbar = tqdm(total=len(data_loader))
        for data in data_loader:
            input = data[0]
            task_label = torch.argmax(data[2], dim=1)
            hrepr = model.encoder(input.to(device))
            avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            if isinstance(hrepr, list):
                hrepr = [avgpool(each_hrepr).squeeze() for each_hrepr in hrepr]
            else:
                hrepr = avgpool(hrepr).squeeze()
            for th in range(4):
                if hrepr[th][task_label==th].shape[0] == 0:
                    continue
                if th == 0:
                    v_output = model.decoders["P"](hrepr[0][task_label==0])
                    correct_list[0] = correct_list[0] + torch.argmax(v_output, 1).eq(data[1].cuda()[task_label==0].squeeze()).float().sum().item()
                    total_list[0] = total_list[0] + v_output.shape[0]
                if th == 1:
                    l_output = model.decoders["A"](hrepr[1][task_label==1])
                    correct_list[1] = correct_list[1] + torch.argmax(l_output, 1).eq(data[1].cuda()[task_label==1].squeeze()).float().sum().item()
                    total_list[1] = total_list[1] + l_output.shape[0]
                if th == 2:
                    c_output = model.decoders["C"](hrepr[2][task_label==2])
                    correct_list[2] = correct_list[2] + torch.argmax(c_output, 1).eq(data[1].cuda()[task_label==2].squeeze()).float().sum().item()
                    total_list[2] = total_list[2] + c_output.shape[0]
                if th == 3:
                    s_output = model.decoders["R"](hrepr[3][task_label==3])
                    correct_list[3] = correct_list[3] + torch.argmax(s_output, 1).eq(data[1].cuda()[task_label==3].squeeze()).float().sum().item()
                    total_list[3] = total_list[3] + s_output.shape[0]

            pbar.update(1)

        pbar.clear()
        pbar.close()
        del pbar

        v_acc = correct_list[0]/ total_list[0]
        l_acc = correct_list[1]/ total_list[1]
        c_acc = correct_list[2]/ total_list[2]
        s_acc = correct_list[3]/ total_list[3]

        return v_acc, l_acc, c_acc, s_acc




    @staticmethod
    def evaluate(model, data_loader, device):
        model.eval()
        print(f"Compute statistics")
        with torch.no_grad():
            (
                p_acc,
                a_acc,
                c_acc,
                r_acc,
            ) = OfficeHomeEvaluator._compute_stats(model, data_loader, device)
        
        avg_acc = (p_acc + a_acc + c_acc + r_acc)/4
        print(
            f"p_acc: {p_acc}, a_acc: {a_acc}, c_acc: {c_acc}, r_acc: {r_acc}, avg_acc: {avg_acc}; "
        )
        return {'p_acc': p_acc, 'a_acc': a_acc,
                'c_acc': c_acc, 'r_acc': r_acc, 'avg_acc': avg_acc}

    @staticmethod
    def evaluate_task(model, data_loader, device):
        model.eval()
        print(f"Compute statistics")
        with torch.no_grad():
            (
                p_acc,
                a_acc,
                c_acc,
                r_acc,
            ) = OfficeHomeEvaluator._compute_stats_task(model, data_loader, device)

        avg_acc = (p_acc + a_acc + c_acc + r_acc)/4
        print(
            f"p_acc: {p_acc}, a_acc: {a_acc}, c_acc: {c_acc}, r_acc: {r_acc}, avg_acc: {avg_acc}; "
        )
        return {'p_acc': p_acc, 'a_acc': a_acc,
                'c_acc': c_acc, 'r_acc': r_acc, 'avg_acc': avg_acc}

    @staticmethod
    def evaluate_each_task(model, data_loader, device):
        model.eval()
        print(f"Compute statistics")
        with torch.no_grad():
            (
                p_acc,
                a_acc,
                c_acc,
                r_acc,
            ) = OfficeHomeEvaluator._compute_stats_each_task(model, data_loader, device)

        avg_acc = (p_acc + a_acc + c_acc + r_acc)/4
        print(
            f"p_acc: {p_acc}, a_acc: {a_acc}, c_acc: {c_acc}, r_acc: {r_acc}, avg_acc: {avg_acc}; "
        )
        return {'p_acc': p_acc, 'a_acc': a_acc,
                'c_acc': c_acc, 'r_acc': r_acc, 'avg_acc': avg_acc}

    @staticmethod
    def evaluate_teacher(model, data_loader, device):
        """评估教师模型（CLIP）在OfficeHome数据集上的表现"""
        model.eval()
        print(f"Compute Teacher statistics")
        # 初始化四个任务的统计量 [P, A, C, R]
        correct_list, total_list = torch.tensor([0] * 4), torch.tensor([0] * 4)

        pbar = tqdm(total=len(data_loader))
        for data in data_loader:
            images = data[0].to(device)
            labels = data[1].to(device)
            task_label = torch.argmax(data[2], dim=1)  # 获取任务标签 (0:P, 1:A, 2:C, 3:R)

            # CLIP模型预测
            with torch.no_grad():
                logits = model(images)  # (batch_size, num_classes)

            # 计算预测结果
            preds = logits.argmax(dim=1)

            # 按任务分类统计
            for task_idx in range(4):
                task_mask = (task_label == task_idx)
                if task_mask.sum() == 0:  # 跳过无样本的任务
                    continue

                task_correct = (preds[task_mask] == labels[task_mask]).sum().item()
                correct_list[task_idx] += task_correct
                total_list[task_idx] += task_mask.sum().item()

            pbar.update(1)

        pbar.close()

        # 计算各任务准确率（避免除零错误）
        p_acc = correct_list[0] / total_list[0] if total_list[0] > 0 else 0.0
        a_acc = correct_list[1] / total_list[1] if total_list[1] > 0 else 0.0
        c_acc = correct_list[2] / total_list[2] if total_list[2] > 0 else 0.0
        r_acc = correct_list[3] / total_list[3] if total_list[3] > 0 else 0.0

        # 计算平均准确率
        avg_acc = (p_acc + a_acc + c_acc + r_acc) / 4

        metrics = {
            'p_acc': p_acc,
            'a_acc': a_acc,
            'c_acc': c_acc,
            'r_acc': r_acc,
            'avg_acc': avg_acc
        }

        # 打印结果（格式与原有评估函数一致）
        print(
            f"Teacher: p_acc: {p_acc:.4f}, a_acc: {a_acc:.4f}, "
            f"c_acc: {c_acc:.4f}, r_acc: {r_acc:.4f}, avg_acc: {avg_acc:.4f};"
        )
        return metrics


