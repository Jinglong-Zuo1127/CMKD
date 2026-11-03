import torch
import torch.nn.functional as F
from CLIP import clip
import torch.nn as nn
from .. import basic_balancer
from .. import balancers


class ClipTeacherModel(torch.nn.Module):
    """CLIP教师模型（适配RadImageNet 165类）"""
    def __init__(self, device='cuda', model_name='ViT-B/16',finetuned_path='/home/zjl/code/MoLA/finetuned_clip/best_clip_teacher.pth'):
        super().__init__()
        self.device = device
        self.model, _ = clip.load(model_name, device=device)
        if finetuned_path is not None:
            checkpoint = torch.load(finetuned_path,map_location=device)
            self.load_state_dict(checkpoint['model_state_dict'])
            print("微调权重加载成功")
        # 1. 严格对应clip_prompts.txt的165个类别（0-164）
        self.class_names = [
            # 任务0：肺部CT相关（0-5类）
            "a medical image of Airspace opacity",
            "a medical image of Bronchiectasis",
            "a medical image of Nodule",
            "a medical image of Parenchyma destruction",
            "a medical image of interstitial lung disease",
            "a medical image of normal lung parenchyma",
            # 任务1：腹部CT相关（6-33类）
            "a medical image of abnormal entire organ",
            "a medical image of adrenal pathology",
            "a medical image of arterial pathology",
            "a medical image of ascites",
            "a medical image of bil dil",
            "a medical image of bladder pathology",
            "a medical image of bowel abnormality",
            "a medical image of bowel inflammation",
            "a medical image of bowel mass",
            "a medical image of degenerative changes",
            "a medical image of dilated urinary tract",
            "a medical image of fat containing tumor",
            "a medical image of gallbladder pathology",
            "a medical image of gallstone",
            "a medical image of intraperitoneal mass",
            "a medical image of liver lesion",
            "a medical image of normal abdominal organs",
            "a medical image of osseous neoplasm",
            "a medical image of ovarian pathology",
            "a medical image of pancreatic lesion",
            "a medical image of post op",
            "a medical image of prostate lesion",
            "a medical image of renal lesion",
            "a medical image of soft tissue collection",
            "a medical image of soft tissue mass",
            "a medical image of splenic lesion",
            "a medical image of urolithiasis",
            "a medical image of uterine pathology",
            # 任务2：甲状腺/血管超声正常相关（34-35类）
            "a medical image of normal thyroid gland",
            "a medical image of normal thyroid nodule",
            # 任务3：多器官超声正常相关（36-48类）
            "a medical image of normal aorta",
            "a medical image of normal bladder",
            "a medical image of normal cbd",
            "a medical image of normal fibroid",
            "a medical image of normal gb",
            "a medical image of normal ivc",
            "a medical image of normal kidney",
            "a medical image of normal liver",
            "a medical image of normal ovary",
            "a medical image of normal pancreas",
            "a medical image of normal portal vein",
            "a medical image of normal spleen",
            "a medical image of normal uterus",
            # 任务4：膝关节MR相关（49-66类）
            "a medical image of acl pathology",
            "a medical image of bone inflammation",
            "a medical image of chondral abnormality",
            "a medical image of fcl pathology",
            "a medical image of fracture",
            "a medical image of hematoma",
            "a medical image of intra",
            "a medical image of mcl pathology",
            "a medical image of meniscal abnormality",
            "a medical image of muscle strain",
            "a medical image of normal knee",
            "a medical image of patella pathology",
            "a medical image of pcl pathology",
            "a medical image of post operative acl",
            "a medical image of quad pathology",
            "a medical image of soft tissue edema",
            "a medical image of soft tissue fluid collection",
            "a medical image of soft tissue mass",
            # 任务5：肩关节MR相关（67-80类）
            "a medical image of acj oa",
            "a medical image of biceps pathology",
            "a medical image of ca++ tendinosis",
            "a medical image of ghj oa",
            "a medical image of infraspinatus pathology",
            "a medical image of labral pathology",
            "a medical image of marrow inflammation",
            "a medical image of normal shoulder",
            "a medical image of osseous lesion",
            "a medical image of post op shoulder",
            "a medical image of soft tissue edema shoulder",
            "a medical image of soft tissue fluid shoulder",
            "a medical image of subscapularis pathology",
            "a medical image of supraspinatus pathology",
            # 任务6：脊柱MR相关（81-89类）
            "a medical image of cord pathology",
            "a medical image of cystic lesions",
            "a medical image of disc pathology",
            "a medical image of dural epidural abn",
            "a medical image of facet arthropathy",
            "a medical image of foraminal pathlogy",
            "a medical image of normal spine",
            "a medical image of osseous abn",
            "a medical image of scoliosis",
            # 任务7：足踝（af）MR相关（90-114类）
            "a medical image of Plantar plate tear",
            "a medical image of achilles pathology",
            "a medical image of atfl pathology",
            "a medical image of bone inflammation af",
            "a medical image of cfl pathology",
            "a medical image of chondral abnormality af",
            "a medical image of coalition",
            "a medical image of deltoid pathology",
            "a medical image of extensor pathology",
            "a medical image of fat containing tumor af",
            "a medical image of flexor pathology",
            "a medical image of hematoma af",
            "a medical image of intra af",
            "a medical image of lisfranc pathology",
            "a medical image of normal af",
            "a medical image of osseous disruption",
            "a medical image of osseous neoplasm af",
            "a medical image of peroneal pathology",
            "a medical image of plantar fascia pathology",
            "a medical image of post op af",
            "a medical image of soft tissue edema af",
            "a medical image of soft tissue fluid af",
            "a medical image of soft tissue mass af",
            "a medical image of spring ligament injury",
            "a medical image of syndesmosis pathology",
            # 任务8：腹部MR相关（115-140类）
            "a medical image of adrenal pathology mriabd",
            "a medical image of arterial pathology mriabd",
            "a medical image of ascites mriabd",
            "a medical image of bil dil mriabd",
            "a medical image of bladder pathology mriabd",
            "a medical image of bowel abnormality mriabd",
            "a medical image of bowel inflammation mriabd",
            "a medical image of bowel mass mriabd",
            "a medical image of degenerative changes mriabd",
            "a medical image of dilated urinary tract mriabd",
            "a medical image of enlarged organ mriabd",
            "a medical image of gallbladder pathology mriabd",
            "a medical image of intraperitoneal mass mriabd",
            "a medical image of liver disease",
            "a medical image of liver lesion mriabd",
            "a medical image of marrow abn",
            "a medical image of normal mriabd",
            "a medical image of osseous neoplasm mriabd",
            "a medical image of ovarian pathology mriabd",
            "a medical image of pancreatic lesion mriabd",
            "a medical image of prostate lesion mriabd",
            "a medical image of renal lesion mriabd",
            "a medical image of soft tissue collection mriabd",
            "a medical image of soft tissue mass mriabd",
            "a medical image of splenic lesion mriabd",
            "a medical image of uterine pathology mriabd",
            # 任务9：脑部MR相关（141-150类）
            "a medical image of acute infarct",
            "a medical image of arteriovenous anomaly",
            "a medical image of chronic infarct",
            "a medical image of edema brain",
            "a medical image of extra brain",
            "a medical image of focal flair hyper",
            "a medical image of intra brain",
            "a medical image of normal brain",
            "a medical image of pituatary lesion",
            "a medical image of white matter changes",
            # 任务10：髋关节MR相关（151-164类）
            "a medical image of abductor pathology",
            "a medical image of capsular pathology",
            "a medical image of chondral pathology hip",
            "a medical image of hamstring pathology",
            "a medical image of hematoma hip",
            "a medical image of labral pathology hip",
            "a medical image of marrow inflammation hip",
            "a medical image of normal hip",
            "a medical image of osseous disruption hip",
            "a medical image of osseous lesion hip",
            "a medical image of post op hip",
            "a medical image of soft tissue edema hip",
            "a medical image of soft tissue fluid hip",
            "a medical image of soft tissue mass hip"
        ]
        self.text_inputs = torch.cat([clip.tokenize(p) for p in self.class_names]).to(device)
        self.model = self.model.float()
        self.model.eval()


    def forward(self, images):
        with torch.no_grad():
            try:
                image_features, attn_weights, cls_token, feats = self.model.encode_image(images)
            except:
                image_features = self.model.encode_image(images)
                attn_weights, cls_token, feats = None, None, None
            text_features = self.model.encode_text(self.text_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()  # 输出165维Logits（0-164）
        return logits, attn_weights, cls_token, feats


@balancers.register("ourbase_radimagenet")
class OurBaseBalancer(basic_balancer.BasicBalancer):
    def __init__(self, compute_stats=False):
        super().__init__(compute_stats)
        # -------------------------- 1. 核心新增：任务-子类别范围映射 --------------------------
        # 完全复用RadData类中的class_label_to_task_label_dict，明确每个任务对应的全局类别索引
        self.task_to_class_ranges = {
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
        # -------------------------- 蒸馏参数 --------------------------
        self.teacher_model = None
        self.kd_weight = 0.05       # Logits蒸馏权重（可调）
        self.feat_weight = 0.5     # 特征蒸馏权重（可调）
        self.teacher_feat_layer = -8
        self.student_feat_layer = "layer3"
        self.mse_criterion = nn.MSELoss()
        self.num_task = 11

    def setup_teacher(self, device='cuda'):
        if self.teacher_model is None:
            self.teacher_model = ClipTeacherModel(device=device, model_name='ViT-B/16')
            print(f"RadImageNet CLIP Teacher initialized (device: {device})")

    @staticmethod
    def compute_kd_loss(student_logits, teacher_logits, T=1.0):
        """Logits蒸馏损失：仅当student_logits与teacher_logits维度一致时有效"""
        assert student_logits.shape[1] == teacher_logits.shape[1], "Logits维度必须匹配！"
        student_log_softmax = F.log_softmax(student_logits / T, dim=1)
        teacher_softmax = F.softmax(teacher_logits / T, dim=1)
        return F.kl_div(student_log_softmax, teacher_softmax, reduction='batchmean') * (T**2)


    def step_with_model(self, data: torch.Tensor, targets: dict, model: torch.nn.Module, criteria: dict, args,
                        **kwargs) -> None:
        # 1. 初始化教师模型
        if self.teacher_model is None:
            self.setup_teacher(device=data.device)

        # 2. 教师模型前向：输出165维全局Logits + 中间特征
        with torch.no_grad():
            teacher_global_logits, _, _, _ = self.teacher_model(data)

        # 3. 解析任务标签（One-Hot→索引：0-10）
        task_targets = kwargs.get('task_targets')
        assert task_targets is not None, "需传入One-Hot编码的task_targets"
        task_label = torch.argmax(task_targets, dim=1)  # [B]：每个样本的任务索引（0-10）

        # 4. 计算原有任务损失（保持原逻辑）
        if 'grad' in args.arch:
            losses, hrepr = self.compute_losses_task_tasklabel(
                data=data, targets=targets, task_targets=task_targets,
                model=model, criteria=criteria, num_task=self.num_task
            )
        else:
            losses, hrepr = self.compute_losses_tasklabel(
                data=data, targets=targets, task_targets=task_targets,
                model=model, criteria=criteria, num_task=self.num_task
            )

        # -------------------------- 5. 关键修改：按任务切片教师Logits --------------------------
        for task_id in range(self.num_task):
            # a. 筛选当前任务的样本（mask：[B]，True表示属于当前任务）
            task_mask = (task_label == task_id)
            if task_mask.sum() == 0:  # 无当前任务样本，跳过
                continue

            # b. 获取当前任务的子类别范围（如任务0：[0,1,2,3,4,5]）
            task_class_indices = self.task_to_class_ranges[task_id]  # 全局类别索引列表
            task_class_num = len(task_class_indices)  # 当前任务的类别数（如任务0为6）

            # c. 切片教师Logits：从165维中提取当前任务的子类别Logits
            # teacher_global_logits.shape: [B, 165] → 切片后：[task_sample_num, task_class_num]
            teacher_task_logits = teacher_global_logits[task_mask][:, task_class_indices]

            # d. 获取学生当前任务的Logits（维度：[task_sample_num, task_class_num]）
            task_hrepr = hrepr[task_mask]  # 当前任务样本的编码器特征
            if isinstance(model, torch.nn.DataParallel):
                student_task_logits = model.module.decoders[str(task_id)](task_hrepr)
            else:
                student_task_logits = model.decoders[str(task_id)](task_hrepr)

            # e. 计算并累加当前任务的蒸馏损失
            kd_loss = self.compute_kd_loss(student_task_logits, teacher_task_logits, T=1.0)
            losses[str(task_id)] += self.kd_weight * kd_loss  # 累加到原任务损失

        # 7. 总损失反向传播
        total_loss = sum(losses.values())
        total_loss.backward()
        self.set_losses(losses)



# @balancers.register("ourbase_radimagenet")
# class OurBaseBalancer(basic_balancer.BasicBalancer):
#
#     def step_with_model(self, data: torch.Tensor, targets: dict, model: torch.nn.Module, criteria: dict, args,
#                         **kwargs) -> None:
#
#         num_task = 11
#
#         print("targets", targets)
#         print("task_targets", kwargs['task_targets'])
#         if 'grad' in args.arch:
#             losses, hrepr = self.compute_losses_task_tasklabel(data=data, targets=targets,
#                                                                task_targets=kwargs['task_targets'], model=model,
#                                                                criteria=criteria, num_task=num_task)
#         else:
#             losses, hrepr = self.compute_losses_tasklabel(data=data, targets=targets,
#                                                           task_targets=kwargs['task_targets'], model=model,
#                                                           criteria=criteria, num_task=num_task)
#
#         total_loss = sum(losses.values())
#         total_loss.backward()
#         self.set_losses(losses)

