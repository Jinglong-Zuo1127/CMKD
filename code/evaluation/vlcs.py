import numpy as np
import torch
from tqdm import tqdm


class VLCSEvaluator:

    @staticmethod
    def _compute_stats(model, data_loader, device):
        correct_list, total_list = torch.tensor([0]*4), torch.tensor([0]*4)

        pbar = tqdm(total=len(data_loader))
        for data in data_loader:
            input = data[0]
            task_label = torch.argmax(data[2], dim=1)
            # hrepr = model.encoder(input.to(device))
            hrepr,attn_weights,proj_feat,proj_token,_= model.encoder(input.to(device))
            # hrepr,  _ = model.encoder(input.to(device))
            avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            if isinstance(hrepr, list):
                hrepr = [avgpool(each_hrepr).squeeze() for each_hrepr in hrepr]
                for th in range(len(hrepr)):
                    each_hrepr = hrepr[th]
                    if len(each_hrepr.shape) == 1:
                        each_hrepr = each_hrepr.unsqueeze(0)
                        hrepr[th] = each_hrepr
            else:
                hrepr = avgpool(hrepr).squeeze()
                if len(hrepr.shape) == 1:
                    hrepr = hrepr.unsqueeze(0)

            for th in range(4):
                if hrepr[task_label==th].shape[0] == 0:
                    continue

                if th == 0:
                    v_output = model.decoders["V"](hrepr[task_label==0])
                    correct_list[0] = correct_list[0] + torch.argmax(v_output, 1).eq(data[1].cuda()[task_label==0].squeeze()).float().sum().item()
                    total_list[0] = total_list[0] + v_output.shape[0]
                if th == 1:
                    l_output = model.decoders["L"](hrepr[task_label==1])
                    correct_list[1] = correct_list[1] + torch.argmax(l_output, 1).eq(data[1].cuda()[task_label==1].squeeze()).float().sum().item()
                    total_list[1] = total_list[1] + l_output.shape[0]
                if th == 2:
                    c_output = model.decoders["C"](hrepr[task_label==2])
                    correct_list[2] = correct_list[2] + torch.argmax(c_output, 1).eq(data[1].cuda()[task_label==2].squeeze()).float().sum().item()
                    total_list[2] = total_list[2] + c_output.shape[0]
                if th == 3:
                    s_output = model.decoders["S"](hrepr[task_label==3])
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
                    v_output = model.decoders["V"](hrepr[task_label==0])
                    correct_list[0] = correct_list[0] + torch.argmax(v_output, 1).eq(data[1].cuda()[task_label==0].squeeze()).float().sum().item()
                    total_list[0] = total_list[0] + v_output.shape[0]
                if th == 1:
                    l_output = model.decoders["L"](hrepr[task_label==1])
                    correct_list[1] = correct_list[1] + torch.argmax(l_output, 1).eq(data[1].cuda()[task_label==1].squeeze()).float().sum().item()
                    total_list[1] = total_list[1] + l_output.shape[0]
                if th == 2:
                    c_output = model.decoders["C"](hrepr[task_label==2])
                    correct_list[2] = correct_list[2] + torch.argmax(c_output, 1).eq(data[1].cuda()[task_label==2].squeeze()).float().sum().item()
                    total_list[2] = total_list[2] + c_output.shape[0]
                if th == 3:
                    s_output = model.decoders["S"](hrepr[task_label==3])
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
            # hrepr = model.encoder(input.to(device))
            #  ViT作为CLIP 视觉编码器
            # hrepr, _, _, _ = model.encoder(input.to(device))
            #  ResNet作为CLIP 视觉编码器
            hrepr, _, _, _, _= model.encoder(input.to(device))
            avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            if isinstance(hrepr, list):
                hrepr = [avgpool(each_hrepr).squeeze() for each_hrepr in hrepr]
            else:
                hrepr = avgpool(hrepr).squeeze()
            for th in range(4):
                if hrepr[th][task_label==th].shape[0] == 0:
                    continue
                if th == 0:
                    v_output = model.decoders["V"](hrepr[0][task_label==0])
                    correct_list[0] = correct_list[0] + torch.argmax(v_output, 1).eq(data[1].cuda()[task_label==0].squeeze()).float().sum().item()
                    total_list[0] = total_list[0] + v_output.shape[0]
                if th == 1:
                    l_output = model.decoders["L"](hrepr[1][task_label==1])
                    correct_list[1] = correct_list[1] + torch.argmax(l_output, 1).eq(data[1].cuda()[task_label==1].squeeze()).float().sum().item()
                    total_list[1] = total_list[1] + l_output.shape[0]
                if th == 2:
                    c_output = model.decoders["C"](hrepr[2][task_label==2])
                    correct_list[2] = correct_list[2] + torch.argmax(c_output, 1).eq(data[1].cuda()[task_label==2].squeeze()).float().sum().item()
                    total_list[2] = total_list[2] + c_output.shape[0]
                if th == 3:
                    s_output = model.decoders["S"](hrepr[3][task_label==3])
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
        print(f"Compute Student statistics")
        with torch.no_grad():
            (
                v_acc,
                l_acc,
                c_acc,
                s_acc,
            ) = VLCSEvaluator._compute_stats(model, data_loader, device)
        
        avg_acc = (v_acc + l_acc + c_acc + s_acc)/4
        print(
            f"Student: v_acc: {v_acc}, l_acc: {l_acc}, c_acc: {c_acc}, s_acc: {s_acc}, avg_acc: {avg_acc}; "
        )
        return {'v_acc': v_acc, 'l_acc': l_acc,
                'c_acc': c_acc, 's_acc': s_acc, 'avg_acc': avg_acc}

    @staticmethod
    def evaluate_task(model, data_loader, device):
        model.eval()
        print(f"Compute Student statistics")
        with torch.no_grad():
            (
                v_acc,
                l_acc,
                c_acc,
                s_acc,
            ) = VLCSEvaluator._compute_stats_task(model, data_loader, device)

        avg_acc = (v_acc + l_acc + c_acc + s_acc)/4
        print(
            f"Student: v_acc: {v_acc}, l_acc: {l_acc}, c_acc: {c_acc}, s_acc: {s_acc}, avg_acc: {avg_acc}; "
        )
        return {'v_acc': v_acc, 'l_acc': l_acc,
                'c_acc': c_acc, 's_acc': s_acc, 'avg_acc': avg_acc}

    @staticmethod
    def evaluate_each_task(model, data_loader, device):
        model.eval()
        print(f"Compute Student statistics")
        with torch.no_grad():
            (
                v_acc,
                l_acc,
                c_acc,
                s_acc,
            ) = VLCSEvaluator._compute_stats_each_task(model, data_loader, device)

        avg_acc = (v_acc + l_acc + c_acc + s_acc)/4
        print(
            f"Student: v_acc: {v_acc}, l_acc: {l_acc}, c_acc: {c_acc}, s_acc: {s_acc}, avg_acc: {avg_acc}; "
        )
        return {'v_acc': v_acc, 'l_acc': l_acc,
                'c_acc': c_acc, 's_acc': s_acc, 'avg_acc': avg_acc}

    @staticmethod
    def evaluate_teacher(model, data_loader, device):
        """评估教师模型（CLIP）在VLCS数据集上的表现"""
        model.eval()
        print(f"Compute Teacher statistics")
        correct_list, total_list = torch.tensor([0] * 4), torch.tensor([0] * 4)

        pbar = tqdm(total=len(data_loader))
        for data in data_loader:
            input = data[0].to(device)
            task_label = torch.argmax(data[2], dim=1)  # 获取任务标签（0,1,2,3）

            # 使用教师模型预测
            with torch.no_grad():
                logits = model(input)  # (batch_size, 5)

            # 计算每个任务的准确率
            pred = logits.argmax(dim=1).cpu()
            labels = data[1].cpu()

            for th in range(4):
                mask = (task_label == th)
                if mask.sum() == 0:
                    continue
                task_pred = pred[mask]
                task_labels = labels[mask]
                correct_list[th] += (task_pred == task_labels).sum().item()
                total_list[th] += mask.sum().item()

            pbar.update(1)

        pbar.clear()
        pbar.close()
        del pbar

        # 计算每个任务的准确率
        v_acc = correct_list[0] / total_list[0] if total_list[0] > 0 else 0.0
        l_acc = correct_list[1] / total_list[1] if total_list[1] > 0 else 0.0
        c_acc = correct_list[2] / total_list[2] if total_list[2] > 0 else 0.0
        s_acc = correct_list[3] / total_list[3] if total_list[3] > 0 else 0.0

        avg_acc = (v_acc + l_acc + c_acc + s_acc) / 4

        metrics = {
            'v_acc': v_acc,
            'l_acc': l_acc,
            'c_acc': c_acc,
            's_acc': s_acc,
            'avg_acc': avg_acc
        }

        print(
            f"Teacher: v_acc: {v_acc}, l_acc: {l_acc}, c_acc: {c_acc}, s_acc: {s_acc}, avg_acc: {avg_acc}; "
        )
        return metrics