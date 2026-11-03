import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize

# -------------------------- 1. 全局配置（统一参数） --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ImageNet归一化参数（两个模型统一用）
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# 输入尺寸（统一224×224）
IMG_SIZE = 224
# 图片路径（替换成你的图片路径，如cat.jpg）
IMG_PATH = r"D:\work\code\MoLA\code\data\datasets\VLCS\CALTECH\train\3\train_imgs_324.jpg"


# -------------------------- 2. 统一预处理函数 --------------------------
def preprocess_image(img_path):
    """图片预处理：读取→缩放→裁剪→归一化→转Tensor→加batch维度"""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # 统一缩放
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    img = Image.open(img_path).convert("RGB")  # 确保3通道
    img_ori = np.array(img)  # 保存原图（用于后续叠加）
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # [1,3,224,224]
    return img_ori, img_tensor


# 加载原图和预处理后的Tensor
img_ori, img_tensor = preprocess_image(IMG_PATH)


# -------------------------- 3. ResNet50 + Grad-CAM 实现 --------------------------
class ResNetGradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.feat_map = None  # 保存最后一个卷积层的输出特征图
        self.grad = None  # 保存对应梯度
        # 注册钩子：获取最后一个卷积层的特征图和梯度（ResNet50的最后卷积层是layer4[-1]）
        self.model.layer4[-1].register_forward_hook(self.forward_hook)
        self.model.layer4[-1].register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.feat_map = output  # [1, 2048, 7, 7]（ResNet50最后卷积层输出）

    def backward_hook(self, module, grad_input, grad_output):
        self.grad = grad_output[0]  # 获取梯度，形状和feat_map一致

    def generate_cam(self):
        # 1. 梯度求平均：对每个特征图的梯度做全局平均池化
        grad_mean = torch.mean(self.grad, dim=[2, 3], keepdim=True)  # [1,2048,1,1]
        # 2. 特征图加权求和：梯度均值 × 特征图
        cam = torch.sum(grad_mean * self.feat_map, dim=1, keepdim=True)  # [1,1,7,7]
        # 3. ReLU去除负贡献：只保留对预测有正作用的区域
        cam = nn.functional.relu(cam)
        # 4. 上采样到原图尺寸（224×224）
        cam = nn.functional.interpolate(
            cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
        )
        # 5. 归一化到0-1（方便画热力图）
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.squeeze().cpu().numpy()


# 加载预训练ResNet50，移到设备
resnet50 = models.resnet50(pretrained=True).to(DEVICE)
# 初始化Grad-CAM
resnet_cam = ResNetGradCAM(resnet50)
# 前向推理（获取预测，为反向传播做准备）
output = resnet_cam.model(img_tensor)
# 反向传播：对预测的top1类别求梯度（聚焦模型最确定的类别）
top1_idx = output.argmax(dim=1)
resnet_cam.model.zero_grad()
output[0, top1_idx].backward()
# 生成ResNet的CAM热力图
resnet_cam_map = resnet_cam.generate_cam()