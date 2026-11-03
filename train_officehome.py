import os
from collections import defaultdict
from os import path as osp
import time
from tqdm import tqdm

import numpy as np
import torch

from code.optim import *
import code.utils.utils as utils
from code.benchmarks.mtl_benchmark import get_benchmark_class
from layers.router import build_router
from code.optim.abase.officehome import ClipTeacherModel
from code.evaluation.officehome import OfficeHomeEvaluator
import time
import torch

def count_params(model, exclude_distill=True):
    skip_keys = ["pca_proj", "gl_proj", "cls_proj", "map_to_high_dim"]

    # 这个 fc 在 ResNet50_LoraSoftRouter 里不是最终任务分类头，
    # 而且当前代码中输出也没有用于 decoders，所以可以作为训练/蒸馏辅助模块排除。
    skip_keys.append("encoder.fc")

    total = 0
    distill = 0

    for name, p in model.named_parameters():
        n = p.numel()
        is_distill = any(k in name for k in skip_keys)

        if is_distill:
            distill += n
            if exclude_distill:
                continue

        total += n

    return total, distill


@torch.no_grad()
def infer_once_officehome(model, images, task_targets):
    model.eval()
    device = next(model.parameters()).device

    images = images.to(device)
    task_targets = task_targets.to(device)
    task_label = torch.argmax(task_targets, dim=1)

    encoder_out = model.encoder(images)
    hrepr = encoder_out[0] if isinstance(encoder_out, tuple) else encoder_out

    avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
    if isinstance(hrepr, list):
        hrepr = [avgpool(each).flatten(1) for each in hrepr]
    else:
        hrepr = avgpool(hrepr).flatten(1)

    outputs = []
    task_names = ["P", "A", "C", "R"]

    for th, task_name in enumerate(task_names):
        mask = task_label == th
        if mask.sum() == 0:
            continue

        if isinstance(hrepr, list):
            feat = hrepr[th][mask]
        else:
            feat = hrepr[mask]

        logits = model.decoders[task_name](feat)
        outputs.append(logits)

    return outputs


@torch.no_grad()
def measure_inference_cost(model, data_loader, device="cuda", warmup=50, repeat=200):
    model.eval()
    model.to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # warmup
    it = iter(data_loader)
    for _ in range(warmup):
        try:
            data = next(it)
        except StopIteration:
            it = iter(data_loader)
            data = next(it)

        images, _, task_targets = data
        _ = infer_once_officehome(model, images, task_targets)

    torch.cuda.synchronize()

    times = []
    it = iter(data_loader)

    for _ in range(repeat):
        try:
            data = next(it)
        except StopIteration:
            it = iter(data_loader)
            data = next(it)

        images, _, task_targets = data

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = infer_once_officehome(model, images, task_targets)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms / batch

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 2

    avg_ms_per_batch = sum(times) / len(times)
    batch_size = data_loader.batch_size
    avg_ms_per_image = avg_ms_per_batch / batch_size

    deploy_params, distill_params = count_params(model, exclude_distill=True)
    all_params, _ = count_params(model, exclude_distill=False)

    print("========== Inference Cost ==========")
    print(f"Deployable params: {deploy_params / 1e6:.3f} M")
    print(f"Training-only distill params: {distill_params / 1e6:.3f} M")
    print(f"All stored params: {all_params / 1e6:.3f} M")
    print(f"Latency: {avg_ms_per_image:.4f} ms / image")
    print(f"Peak GPU memory: {peak_mem:.2f} MB")

    return {
        "deploy_params_M": deploy_params / 1e6,
        "distill_params_M": distill_params / 1e6,
        "all_stored_params_M": all_params / 1e6,
        "latency_ms_per_image": avg_ms_per_image,
        "peak_gpu_memory_MB": peak_mem,
    }
# def count_parameters(model, grad):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad == grad)

class MTLTrainer:
    def __init__(self, args):
        self.args = args
        self.benchmark = get_benchmark_class(args.benchmark)(args)
        self.balancer = get_method(args.balancer, compute_stats=args.compute_cnumber)

        self.model = self.benchmark.get_model(args)
        self.balancer.add_model_parameters(self.model)
        self.model = self.model.cuda()

        # trainable_param_num = count_parameters(self.model, True)
        # other_param_num = count_parameters(self.model, False)

        self.optimizer = self.benchmark.get_optim(self.model, args)
        self.scheduler = self.benchmark.get_scheduler(self.optimizer, args)
        # 设置蒸馏参数
        # if args.use_kd:
        #     # 确保balancer已初始化
        #     if hasattr(self.balancer, 'kd_weight'):
        #         self.balancer.kd_weight = args.kd_weight
        #         print(f"Knowledge distillation enabled with weight: {args.kd_weight}")
        #     else:
        #         print("Warning: Current balancer does not support knowledge distillation")
        self.teacher_model = ClipTeacherModel(device='cuda')
        self.teacher_model.eval()

        if 'router' in args.arch:
            self.router = build_router(num_experts=self.args.n_expert).cuda()
            self.model.router = self.router
            for module in self.model.modules():
                if hasattr(module, 'router'):
                    module.router = self.router
            # trainable_param_num = count_parameters(self.model, True)
            # other_param_num = count_parameters(self.model, False)

            self.router_optimizer = self.benchmark.get_optim(self.model, args)
            self.router_scheduler = self.benchmark.get_scheduler(self.router_optimizer, args)
        else:
            self.router = None

        if self.args.load_state:
            print("加载权重")

            self.load_state(self.args.load_state)


        self.res_path = None
        self.train_loader_kwargs = {}
        self.valid_loader_kwargs = {}
        self.train_metrics = []
        self.valid_metrics = []

        if self.args.multigpu:
            self.model.encoder = torch.nn.DataParallel(self.model.encoder)
            for each_key in self.model.decoders.keys():
                self.model.decoders[each_key] = torch.nn.DataParallel(self.model.decoders[each_key])
            self.model = torch.nn.DataParallel(self.model)

    def train_epoch(self):
        train_loader = torch.utils.data.DataLoader(self.benchmark.datasets['train'], **self.train_loader_kwargs)
        self.model.train()
        if 'router' in args.arch:
            self.router.train()
        loss_total, task_losses = 0, defaultdict(float)

        pbar = tqdm(total=len(train_loader))
        fmtl_metrics = open(osp.join(self.res_path, 'mtl_metrics.txt'), 'a')

        for i, data in enumerate(train_loader):
            if ((i+1)%self.args.accumulation_steps) == 0:
                self.optimizer.zero_grad()
                if 'router' in args.arch:
                    self.router_optimizer.zero_grad()

            self.balancer.step_with_model(
                data=data[0].cuda(),
                targets=data[1].cuda(),
                task_targets=data[2].cuda(),
                model=self.model,
                criteria=self.benchmark.task_criteria,
                args=self.args,
                router=self.router,
            )



            if ((i+1)%self.args.accumulation_steps) == 0:
                self.optimizer.step()
                if 'router' in args.arch:
                    self.router_optimizer.step()

            losses = self.balancer.losses
            if hasattr(self.balancer, 'info') and self.balancer.info is not None:
                fmtl_metrics.write(utils.strfy(self.balancer.info) + "\n")
                fmtl_metrics.flush()

            loss_total += sum(losses.values())
            for task_id in losses:
                task_losses[task_id] += losses[task_id]

            post = {"loss": sum(losses.values())}
            post.update(**losses)
            pbar.set_postfix(post)
            pbar.update(1)

        pbar.clear()
        pbar.close()
        del pbar

        avg_total_loss = loss_total / len(train_loader)
        for task_id in task_losses:
            task_losses[task_id] /= len(train_loader)

        return avg_total_loss, task_losses

    @torch.no_grad()
    def valid_epoch(self):
        self.model.eval()
        if 'router' in args.arch:
            self.router.eval()

        test_loader = torch.utils.data.DataLoader(self.benchmark.datasets['valid'], **self.valid_loader_kwargs)
        loss_total = 0.0

        for data in test_loader:
            if 'grad' in args.arch:
                losses, _ = self.balancer.compute_losses_task_multidomain(
                    data=data[0].cuda(),
                    targets=data[1].cuda(),
                    task_targets=data[2].cuda(),
                    model=self.model,
                    criteria=self.benchmark.task_criteria,
                )
            else:
                losses, _ = self.balancer.compute_losses_multidomain(
                    data=data[0].cuda(),
                    targets=data[1].cuda(),
                    task_targets=data[2].cuda(),
                    model=self.model,
                    criteria=self.benchmark.task_criteria,
                )

            loss_total += sum(losses.values())

        avg_val_loss = loss_total / len(test_loader)
        if 'grad' in args.arch:
            metrics = self.benchmark.evaluate_task(self.model, test_loader)
        elif args.arch in ['Cross_stitch', 'MMoE', 'DSelect_k', 'LTB', 'CGC', 'PLE', 'HPS', 'MTAN']:
            metrics = self.benchmark.evaluate_each_task(self.model, test_loader)
        else:
            metrics = self.benchmark.evaluate(self.model, test_loader)
        # 评估教师模型
        # teacher_metrics = OfficeHomeEvaluator.evaluate_teacher(self.teacher_model, test_loader,device='cuda')
        return avg_val_loss, metrics

    def save_state(self, path):
        model_state = self.model.state_dict()
        torch.save({
            "state_dict": model_state,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }, path)
        
    def load_state(self, path):
        state = torch.load(path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])

    def run_experiment(self):
        utils.fix_seed(42 + self.args.round)
        train_kwargs = {
            "batch_size": self.args.train_batch,
            "drop_last": True,
            "shuffle": True,
        }
        test_kwargs = {"batch_size": self.args.test_batch, "shuffle": False}
        cuda_kwargs = {"num_workers": 8, "pin_memory": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

        self.train_loader_kwargs = train_kwargs
        self.valid_loader_kwargs = test_kwargs

        if self.args.eval_only:
            self.valid_epoch()
            return
        # if self.args.eval_only:
        #     test_loader = torch.utils.data.DataLoader(
        #         self.benchmark.datasets['valid'],
        #         batch_size=self.args.test_batch,
        #         shuffle=False,
        #         num_workers=8,
        #         pin_memory=True
        #     )
        #
        #     self.model.eval()
        #     measure_inference_cost(self.model, test_loader, device="cuda")
        #     return


        res_path = osp.join(self.args.output_path, self.args.benchmark, self.args.balancer, self.args.arch, str(self.args.round))
        self.res_path = res_path
        if not osp.exists(res_path):
            os.makedirs(res_path)
        # 创建日志文件
        metrics_file_path = os.path.join(res_path, 'mtl_metrics.txt')
        # teacher_file_path = os.path.join(res_path, 'clip_metrics.txt')
        with open(metrics_file_path, 'w') as f:  # 清空已有内容
            f.write("Training Metrics Log\n")
            f.write("=" * 50 + "\n")

        # with open(teacher_file_path, 'w') as f:
        #     f.write("Teacher Model Metrics Log\n")
        #     f.write("=" * 50 + "\n")
        #     f.write(f"Teacher Model: CLIP ViT-B/16\n")
        #     f.write("=" * 50 + "\n\n")

        best_val_loss = np.inf
        best_acc = 0
        best_model_metrics = None
        for epoch in range(self.args.epochs):
            print(f"Round: {args.round}; epoch: {epoch}")
            if args.arch in ['LTB']:
                if isinstance(self.model, torch.nn.DataParallel):
                    self.model.module.encoder.module.epochs = self.args.epochs
                    self.model.module.encoder.module.epoch = epoch
                else:
                    self.model.encoder.epochs = self.args.epochs
                    self.model.encoder.epoch = epoch
            avg_train_loss, avg_task_losses = self.train_epoch()

            # 打印并写入训练损失
            train_loss_str = f"Epoch: {epoch}, avg_train_loss: {avg_train_loss:.4f}, "
            for task_id in avg_task_losses:
                train_loss_str += f"loss_{task_id}: {avg_task_losses[task_id]:.4f}, "
            print(train_loss_str.strip(', '))

            if args.fast and epoch!=self.args.epochs-1:
                continue
            else:
                avg_val_loss, metrics = self.valid_epoch()
                avg_acc = metrics['avg_acc']
                val_info = f"avg_valid_loss: {avg_val_loss:.4f}"
                # print(f"Epoch: {epoch}, {val_info}")

                # 构建评估指标字符串，打印学生指标
                eval_str = ""
                for k, v in metrics.items():
                    if isinstance(v, (float, int)):
                        eval_str += f"{k}: {v:.4f}, "
                    else:
                        # 处理可能的非数值指标
                        eval_str += f"{k}: {str(v)}, "

                # print(eval_str.strip(', '))

                # 打印教师指标
                # teacher_str = ""
                # for k, v in teacher_metrics.items():
                #     if isinstance(v, (float, int)):
                #         teacher_str += f"{k}: {v:.4f}, "
                #     else:
                #         teacher_str += f"{k}: {str(v)}, "
                # print(f"Teacher Metrics: {teacher_str.strip(', ')}")

                if avg_acc > best_acc:
                    best_acc = avg_acc

                    best_model_metrics = metrics
                    print(f"Save the model state")
                    self.save_state(osp.join(self.res_path, "best_test.pth"))
            with open(metrics_file_path, 'a') as f:
                # 写入训练损失
                f.write(train_loss_str.strip(', ') + ";\n")

                # 写入验证损失（如果进行了验证）
                if 'Skipped' not in val_info:
                    f.write(f"{val_info}; ")

                    # 写入评估指标
                    if metrics:
                        f.write(", ".join([f"{k}: {v:.4f}" if isinstance(v, (float, int))
                                           else f"{k}: {str(v)}" for k, v in metrics.items()]) + ";\n")
                    else:
                        f.write("no metrics;\n")

                    f.write("\n")

            # 写入教师指标
            # with open(teacher_file_path, 'a') as f:
            #     f.write(f"Epoch {epoch}\n")
            #     f.write(f"Validation Metrics: {teacher_str.strip(', ')}\n\n")

            self.scheduler.step()
            if 'router' in args.arch:
                self.router_scheduler.step()

        self.save_state(osp.join(self.res_path, "final_test.pth"))
        print('='*30)
        print('best model metrics: ', best_model_metrics)
        print('='*30)
        print(self.args)


if __name__ == "__main__":
    parser = utils.common_argparser()
    args, _ = parser.parse_known_args()

    benchmark_type = get_benchmark_class(args.benchmark)
    specific_parser = benchmark_type.get_arg_parser(parser)
    args = specific_parser.parse_args()
    print(args)

    trainer = MTLTrainer(args)
    trainer.run_experiment()


# pip install scipy cvxpy matplotlib seaborn tqdm

# CUDA_VISIBLE_DEVICES=0 python train.py --benchmark vlcs_resnet50 --balancer ourbase --arch lora_soft_router --lora_layer 1 2 3 --lora_rank 4 4 4 8
# CUDA_VISIBLE_DEVICES=0 python train.py --benchmark vlcs_resnet50 --balancer ourbase --arch lora_grad --lora_layer 1 3 --lora_rank 4 4 4 8


