import torch
from CLIP import clip
class ClipTeacherModel(torch.nn.Module):
    """CLIP教师模型，适配MedMNIST数据集，使用具体的类别名称"""

    def __init__(self, device='cuda', model_name='ViT-B/16'):
        super().__init__()
        self.device = device
        self.model, _ = clip.load(model_name, device=device)

        # MedMNIST任务描述（使用文档2中的具体类别名称）
        self.task_labels = {
            0: {  # pathmnist
                0: "adipose",
                1: "background",
                2: "debris",
                3: "lymphocytes",
                4: "mucus",
                5: "smooth muscle",
                6: "normal colon mucosa",
                7: "cancer-associated stroma",
                8: "colorectal adenocarcinoma epithelium",
            },
            1: {  # octmnist
                0: "choroidal neovascularization",
                1: "diabetic macular edema",
                2: "drusen",
                3: "normal",
            },
            2: {  # organcmnist
                0: "bladder",
                1: "femur-left",
                2: "femur-right",
                3: "heart",
                4: "kidney-left",
                5: "kidney-right",
                6: "liver",
                7: "lung-left",
                8: "lung-right",
                9: "pancreas",
                10: "spleen",
            },
            3: {  # bloodmnist
                0: "basophil",
                1: "eosinophil",
                2: "erythroblast",
                3: "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)",
                4: "lymphocyte",
                5: "monocyte",
                6: "neutrophil",
                7: "platelet",
            },
            4: {  # breastmnist
                0: "malignant",
                1: "normal, benign",
            },
            5: {  # dermamnist
                0: "actinic keratoses and intraepithelial carcinoma",
                1: "basal cell carcinoma",
                2: "benign keratosis-like lesions",
                3: "dermatofibroma",
                4: "melanoma",
                5: "melanocytic nevi",
                6: "vascular lesions",
            },
            6: {  # organamnist
                0: "bladder",
                1: "femur-left",
                2: "femur-right",
                3: "heart",
                4: "kidney-left",
                5: "kidney-right",
                6: "liver",
                7: "lung-left",
                8: "lung-right",
                9: "pancreas",
                10: "spleen",
            },
            7: {  # organsmnist
                0: "bladder",
                1: "femur-left",
                2: "femur-right",
                3: "heart",
                4: "kidney-left",
                5: "kidney-right",
                6: "liver",
                7: "lung-left",
                8: "lung-right",
                9: "pancreas",
                10: "spleen",
            },
            8: {  # pneumoniamnist
                0: "normal",
                1: "pneumonia",
            }
        }
        # 专业医学文本提示模板
        self.task_prompts = {
            # 病理组织学图像
            0: "H&E stained histopathology image of {class_name}",
            # 视网膜OCT图像
            1: "Retinal OCT scan showing {class_name}",
            # CT扫描器官图像
            2: "CT scan image of the {class_name}",
            # 血液涂片图像
            3: "Peripheral blood smear showing {class_name}",
            # 乳腺X光图像
            4: "Mammography image of {class_name} breast tissue",
            # 皮肤镜图像
            5: "Dermatoscopic image of {class_name}",
            # MRI扫描器官图像
            6: "MRI scan image of the {class_name}",
            # 更多器官图像
            7: "Medical scan image of the {class_name}",
            # 胸部X光图像
            8: "Chest X-ray showing {class_name}"
        }

        # 为所有任务生成文本提示（使用具体类别名称）
        self.text_inputs = {}

        for task_id, class_dict in self.task_labels.items():
            # 获取任务特定的专业提示模板
            template = self.task_prompts.get(task_id, "Medical image showing {class_name}")
            prompts = [template.format(class_name=class_name)
                       for class_name in class_dict.values()]

            self.text_inputs[task_id] = torch.cat([
                clip.tokenize(prompt) for prompt in prompts
            ]).to(device)
        self.model.eval()  # 固定为评估模式

    def forward(self, images):
        """
        前向传播，返回给定图像和任务的logits

        参数:
            images: 输入图像 (batch_size, 3, H, W)

        返回:
            每个任务的logits字典
        """
        with torch.no_grad():
            # 获取图像特征
            image_features, attn_weights, cls_token, feats = self.model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 初始化输出logits字典
            task_logits = {}

            # 为每个任务单独计算logits
            for task_id in range(9):
                # 获取任务特定文本特征
                text_features = self.model.encode_text(self.text_inputs[task_id])
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # 计算相似度
                logit_scale = self.model.logit_scale.exp()
                task_logits[task_id] = logit_scale * image_features @ text_features.t()

        return task_logits, attn_weights, cls_token, feats