
# --- 导入必要的库 ---
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time
import random
import os
from typing import Dict, Optional, Tuple

# 用于混淆矩阵和可视化
try:
    from sklearn.metrics import confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("警告: sklearn未安装，混淆矩阵功能将不可用")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("警告: seaborn未安装，混淆矩阵将使用matplotlib绘制")

# 处理 matplotlib 中文显示
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

font_list = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'STSong', 'Arial Unicode MS']
available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
chinese_font = next((font for font in font_list if font in available_fonts), None)
if chinese_font:
    plt.rcParams['font.sans-serif'] = [chinese_font]
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 超参数配置 ---
BATCH_SIZE = 128      # 批次大小，较大的batch size有助于稳定训练

# SAM优化器实现 (Sharpness-Aware Minimization)
# 论文：Sharpness-Aware Minimization for Efficiently Improving Generalization (ICLR 2021)
class SAM(torch.optim.Optimizer):
    """
    SAM优化器：Sharpness-Aware Minimization
    核心思想：不仅找损失最低的点，还找最平坦的区域，提升模型泛化能力
    
    使用方法：
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        optimizer = SAM(model.parameters(), base_optimizer, rho=0.05)
    """
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        # 初始化Optimizer，但param_groups将使用base_optimizer的
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer
        # SAM的param_groups应该与base_optimizer一致，但需要添加rho参数
        self.param_groups = base_optimizer.param_groups
        # 为每个param_group添加rho参数（base_optimizer的param_groups中没有rho）
        for group in self.param_groups:
            group["rho"] = rho
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        第一步：计算锐度方向（使损失最大的方向）
        在参数空间中找到使损失最大的扰动方向
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)  # 归一化因子
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)  # 锐度方向
                p.add_(e_w)  # 临时移动到锐度方向
                self.state[p]["e_w"] = e_w  # 保存扰动，用于后续恢复
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        第二步：在锐度方向上计算梯度并更新参数
        然后恢复参数到原始位置，使用锐度方向的梯度更新
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # 恢复参数到原始位置
        
        self.base_optimizer.step()  # 使用锐度方向的梯度更新参数
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        注意：SAM需要两步更新，不能直接调用step()
        应该先调用first_step()，然后计算损失和梯度，再调用second_step()
        """
        raise NotImplementedError("SAM doesn't support step(), use first_step() and second_step() instead")
    
    def _grad_norm(self):
        """
        计算所有参数梯度的L2范数
        """
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
EPOCHS = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 数据预处理与增强 ---
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# CIFAR-10 统计值（仅基于训练集的均值与方差），用于标准化到零均值单位方差
cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std = (0.2470, 0.2435, 0.2616)

# 基线预处理：仅做 Tensor 化 + 标准化，保证输入分布稳定
# 注意：这个transform_simple现在只用于测试集，训练集使用transform_augmented
transform_simple = transforms.Compose([
    transforms.ToTensor(),                                 # 把PIL/ndarray -> float tensor，像素归一化到[0,1]
    transforms.Normalize(cifar_mean, cifar_std)            # 按训练集统计值做零均值/单位方差，稳定优化
])

# 中等强度数据增强：包含RandomCrop、RandomFlip、ColorJitter、RandomErasing
transform_medium = transforms.Compose([
    transforms.RandomCrop(32, padding=4),                  # 先四周填充4像素再裁回32x32，引入位移鲁棒性
    transforms.RandomHorizontalFlip(p=0.5),                # 左右翻转，平衡方向偏好
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
    transforms.ToTensor(),                                 # 转 tensor
    transforms.Normalize(cifar_mean, cifar_std),           # 标准化
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
])

# 增强版预处理：在基线之上加入裁剪/翻转/色彩抖动/随机擦除，提高数据多样性以防过拟合
transform_augmented = transforms.Compose([
    transforms.RandomCrop(32, padding=4),                  # 先四周填充4像素再裁回32x32，引入位移鲁棒性
    transforms.RandomHorizontalFlip(p=0.5),                # 左右翻转，平衡方向偏好
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # 轻度色彩扰动，防光照过拟合
    transforms.ToTensor(),                                 # 同上，转 tensor
    transforms.Normalize(cifar_mean, cifar_std),           # 同上，标准化
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0)   # 随机遮挡局部，迫使模型看整体
])

# 测试集：绝不做增强，只做标准化，确保评估公平
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_mean, cifar_std)
])

# RandAugment增强：SOTA级自动数据增强（需要torchvision >= 0.10.0）
# 如果版本不支持，会fallback到基础增强
try:
    from torchvision.transforms import RandAugment
    RANDAUGMENT_AVAILABLE = True
except ImportError:
    RANDAUGMENT_AVAILABLE = False
    print("--- 警告: torchvision版本不支持RandAugment，将使用TrivialAugmentWide替代 ---")

# SOTA级增强：RandAugment + CutMix（在训练循环中应用）
transform_randaugment = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(cifar_mean, cifar_std),
])

# 如果支持RandAugment，添加到pipeline中
if RANDAUGMENT_AVAILABLE:
    transform_randaugment = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        RandAugment(num_ops=2, magnitude=9),  # num_ops: 每次应用几个操作, magnitude: 增强强度
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])
else:
    # Fallback: 使用TrivialAugmentWide（torchvision内置的自动增强）
    try:
        from torchvision.transforms import TrivialAugmentWide
        transform_randaugment = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ])
    except ImportError:
        # 如果连TrivialAugmentWide都没有，使用基础增强
        transform_randaugment = transform_augmented

# --- 3. 手写版 ResNet（适配 CIFAR-10） ---

class BasicBlock(nn.Module):
    """
    经典 3x3 + 3x3 残差块（ResNet 基本单元）
    输入 -> Conv3x3 -> BN -> ReLU -> Dropout -> Conv3x3 -> BN -> 残差相加 -> ReLU
    当 stride != 1 或通道数改变时，用 1x1 卷积调整捷径分支的尺寸。
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout=0.0):
        super().__init__()
        # 3x3 卷积，步长可为2实现下采样；padding=1 保持特征尺寸/2
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # BN 稳定分布，加速收敛
        self.bn1 = nn.BatchNorm2d(planes)
        # 第二个 3x3 卷积，保持通道和尺度
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)                  # 就地 ReLU 节省内存
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()  # 可选的随机失活，防过拟合

        if stride != 1 or in_planes != planes:             # 尺度或通道不一致时，用 1x1 卷积对捷径对齐
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # 保存捷径分支
        identity = self.shortcut(x)
        # 主分支：Conv-BN-ReLU-(可选 Dropout)-Conv-BN
        out = self.relu(self.bn1(self.conv1(x)))  # 第一个卷积+BN+ReLU
        out = self.dropout(out)  # 可选Dropout，防止过拟合
        out = self.bn2(self.conv2(out))  # 第二个卷积+BN（注意：这里BN在ReLU之前）
        # 残差相加再激活
        out += identity  # 残差连接：将主分支和捷径分支相加
        out = self.relu(out)  # 最后激活
        return out


class MiniResNet(nn.Module):
    """
    手写 ResNet，适配 CIFAR-10（32x32输入）。
    参考标准ResNet实现，但针对CIFAR-10做了优化：
    - conv1使用3x3卷积（而非ImageNet的7x7），stride=1（不下采样）
    - 在conv1之后添加MaxPool（满足实验要求：至少2个池化层）
    - 标准ResNet架构：conv1 -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
    
    block_layers: 每个 stage 的残差块数，例如 (2,2,2,2) 就是 ResNet-18。
    为了满足实验要求（至少2个池化层），我们在conv1之后添加MaxPool，并在layer4之后添加第二个MaxPool。
    """

    def __init__(self, block_layers=(2, 2, 2, 2), base_channels=32, dropout=0.0, 
                 num_classes=10, fc_dropout=0.0, zero_init_residual=False):
        super().__init__()
        self.in_planes = base_channels
        self.dropout = dropout

        # CIFAR-10适配：使用3x3卷积，stride=1，padding=1（不下采样）
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)           # 首层 BN，稳住输入分布
        self.relu = nn.ReLU(inplace=True)                  # 首层激活
        # 第一个MaxPool层：将32x32下采样到16x16
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 下采样：32x32 -> 16x16

        # 四个 stage，每个 stage 降采样一次（stride=2），通道数翻倍
        self.layer1 = self._make_layer(base_channels, block_layers[0], stride=1)
        self.layer2 = self._make_layer(base_channels * 2, block_layers[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, block_layers[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 8, block_layers[3], stride=2)
        # 第二个MaxPool层：进一步下采样
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样：2x2 -> 1x1

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，将任意尺寸压缩到1x1
        # 在全连接层前添加Dropout
        self.fc_dropout = nn.Dropout(fc_dropout) if fc_dropout > 0 else nn.Identity()
        self.fc = nn.Linear(base_channels * 8 * BasicBlock.expansion, num_classes)
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride=s, 
                                   dropout=self.dropout))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # stem：3x3卷积 + BN + ReLU + MaxPool（标准ResNet架构）
        out = self.conv1(x)  # [B, 3, 32, 32] -> [B, C, 32, 32]
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)  # [B, C, 32, 32] -> [B, C, 16, 16]
        
        # 四个stage，每个stage通过stride=2的卷积降采样
        out = self.layer1(out)  # [B, C, 16, 16]（不降采样）
        out = self.layer2(out)  # [B, 2C, 8, 8]（stride=2降采样）
        out = self.layer3(out)  # [B, 4C, 4, 4]（stride=2降采样）
        out = self.layer4(out)  # [B, 8C, 2, 2]（stride=2降采样）
        
        # 第二个MaxPool：进一步下采样
        out = self.pool2(out)  # [B, 8C, 2, 2] -> [B, 8C, 1, 1]
        
        # 全局平均池化，再接全连接做分类
        out = self.global_pool(out)  # [B, 8C, 1, 1] -> [B, 8C, 1, 1]（这里尺寸不变，但更通用）
        out = torch.flatten(out, 1)  # [B, 8C, 1, 1] -> [B, 8C]
        out = self.fc_dropout(out)  # 全连接层前的Dropout，防止过拟合
        out = self.fc(out)  # [B, 8C] -> [B, 10]
        return out


def build_resnet(depth="18层", base_channels=32, dropout=0.0, fc_dropout=0.0, zero_init_residual=False):
    """
    构建ResNet模型的工厂函数
    
    Args:
        depth: 模型深度，"浅层"=4块，"18层"=8块，"34层"=16块
        base_channels: 基础通道数，后续stage会翻倍
        dropout: 残差块中的Dropout比率
        fc_dropout: 全连接层前的Dropout比率
        zero_init_residual: 是否零初始化残差分支的最后一个BN（可提升0.2-0.3%准确率）
    """
    depth2layers = {
        "浅层": (1, 1, 1, 1),
        "18层": (2, 2, 2, 2),
        "34层": (3, 4, 6, 3)
    }
    block_layers = depth2layers.get(depth, (2, 2, 2, 2))
    return MiniResNet(block_layers=block_layers, base_channels=base_channels,
                      dropout=dropout, num_classes=len(classes), 
                      fc_dropout=fc_dropout, zero_init_residual=zero_init_residual)


# --- 4. CutMix 数据增强实现 ---

def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    CutMix数据增强：将两张图片混合，同时混合标签
    
    Args:
        x: 输入图像batch [B, C, H, W]
        y: 输入标签batch [B]
        alpha: Beta分布的参数，控制混合区域的分布（alpha越大，混合区域越均匀）
    
    Returns:
        mixed_x: 混合后的图像
        y_a: 第一张图的标签
        y_b: 第二张图的标签
        lam: 混合比例（用于计算损失）
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)  # 从Beta分布采样混合比例
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)  # 随机打乱索引
    
    # 计算裁剪区域的边界框
    W, H = x.size(3), x.size(2)
    cut_rat = np.sqrt(1.0 - lam)  # 裁剪区域的比例
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # 随机选择裁剪区域的中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # 计算裁剪区域的边界
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # 执行CutMix：将x[index]的区域复制到x上
    # 注意：需要先复制，避免原地修改影响原始数据
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # 根据实际裁剪区域重新计算混合比例
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """
    CutMix的损失函数：混合标签的加权交叉熵
    
    Args:
        criterion: 基础损失函数（通常是CrossEntropyLoss）
        pred: 模型预测的logits
        y_a: 第一张图的标签
        y_b: 第二张图的标签
        lam: 混合比例
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Mixup数据增强：将两张图片按比例混合，同时混合它们的标签。
    与CutMix不同，Mixup是对整张图片进行线性混合，而不是局部区域替换。
    
    Args:
        x: 输入图像batch (B, C, H, W)
        y: 标签batch (B,)
        alpha: Beta分布的参数，控制混合强度（alpha越大，混合越均匀）
    
    Returns:
        mixed_x: 混合后的图像
        y_a: 第一张图的标签
        y_b: 第二张图的标签
        lam: 混合比例（用于计算损失）
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)  # 从Beta分布采样混合比例
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)  # 随机打乱索引
    
    # 线性混合两张图片
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup的损失函数：混合标签的加权交叉熵（与CutMix相同）
    
    Args:
        criterion: 基础损失函数（通常是CrossEntropyLoss）
        pred: 模型预测的logits
        y_a: 第一张图的标签
        y_b: 第二张图的标签
        lam: 混合比例
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# --- 5. 训练 & 评估工具函数 ---


def cross_entropy_with_label_smoothing(outputs, targets, smoothing):
    """
    标签平滑版交叉熵：
    将 one-hot 硬标签变为软标签，降低模型过拟合到单一类别的倾向。
    """
    num_classes = outputs.size(1)
    log_probs = torch.log_softmax(outputs, dim=1)
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    loss = (-true_dist * log_probs).sum(dim=1).mean()
    return loss


def train_model(model, train_loader, optimizer, criterion, device,
                label_smoothing=0.0, grad_clip: Optional[float] = None,
                use_cutmix: bool = False, cutmix_alpha: float = 1.0,
                use_mixup: bool = False, mixup_alpha: float = 1.0,
                use_sam: bool = False):
    """
    单个 epoch 的训练流程：
    1) 前向传播：model(inputs) 得到 logits
    2) 损失计算：交叉熵或标签平滑交叉熵（可选CutMix/Mixup）
    3) 反向传播：loss.backward() 把梯度写到参数 .grad
    4) 可选梯度裁剪：防止梯度爆炸
    5) 参数更新：optimizer.step() 按梯度下降更新参数
    
    如果使用SAM优化器，需要两步更新：
    - 第一步：计算锐度方向（first_step）
    - 第二步：在锐度方向上计算梯度并更新（second_step）
    
    Args:
        use_cutmix: 是否使用CutMix数据增强
        cutmix_alpha: CutMix的Beta分布参数
        use_mixup: 是否使用Mixup数据增强
        mixup_alpha: Mixup的Beta分布参数
        use_sam: 是否使用SAM优化器（需要两步更新）
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 数据增强策略：优先使用CutMix或Mixup（如果启用）
        # 如果同时启用，随机选择其中一个（50%概率）
        use_mixup_this_batch = False
        use_cutmix_this_batch = False
        
        if use_cutmix and use_mixup:
            # 同时启用时，随机选择
            if random.random() < 0.5:
                use_cutmix_this_batch = True
            else:
                use_mixup_this_batch = True
        elif use_cutmix:
            use_cutmix_this_batch = random.random() < 0.5  # 50%概率使用CutMix
        elif use_mixup:
            use_mixup_this_batch = random.random() < 0.5  # 50%概率使用Mixup
        
        # 应用数据增强
        if use_cutmix_this_batch:
            inputs, y_a, y_b, lam = cutmix_data(inputs, labels, cutmix_alpha)
        elif use_mixup_this_batch:
            inputs, y_a, y_b, lam = mixup_data(inputs, labels, mixup_alpha)
        else:
            y_a, y_b, lam = labels, None, 1.0
        
        optimizer.zero_grad()
        outputs = model(inputs)                    # 前向传播，得到每类的 logits

        # 计算损失
        if (use_cutmix_this_batch or use_mixup_this_batch) and y_b is not None:
            # 使用CutMix或Mixup的混合损失
            loss = cutmix_criterion(criterion, outputs, y_a, y_b, lam)  # CutMix和Mixup使用相同的损失函数
        elif label_smoothing > 0:
            loss = cross_entropy_with_label_smoothing(outputs, labels, label_smoothing)  # 软标签交叉熵
        else:
            loss = criterion(outputs, labels)      # 标准交叉熵

        loss.backward()                            # 反向传播：计算所有参数的梯度
        
        if use_sam:
            # SAM优化器需要两步更新
            # 第一步：计算锐度方向（使损失最大的方向）
            optimizer.first_step(zero_grad=True)
            
            # 在锐度方向上重新计算损失和梯度
            outputs_sam = model(inputs)
            if (use_cutmix_this_batch or use_mixup_this_batch) and y_b is not None:
                loss_sam = cutmix_criterion(criterion, outputs_sam, y_a, y_b, lam)
            elif label_smoothing > 0:
                loss_sam = cross_entropy_with_label_smoothing(outputs_sam, labels, label_smoothing)
            else:
                loss_sam = criterion(outputs_sam, labels)
            loss_sam.backward()  # 在锐度方向上计算梯度
            
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # 第二步：使用锐度方向的梯度更新参数，并恢复参数位置
            optimizer.second_step(zero_grad=True)
        else:
            # 标准优化器（AdamW等）的单步更新
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # 可选梯度裁剪
            optimizer.step()                           # 参数更新

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        
        # 准确率计算：当使用CutMix/Mixup时，图像是混合的，没有单一的"正确"标签
        # 因此我们只在没有使用混合增强的样本上计算准确率，或者使用原始标签y_a作为近似
        # 使用混合增强时，训练准确率会偏低，因为混合图像没有单一"正确"标签
        if (use_cutmix_this_batch or use_mixup_this_batch) and y_b is not None:
            # 对于混合样本，使用原始标签y_a计算准确率（虽然不完全准确，但作为参考）
            # 实际训练准确率会偏低，这是使用混合增强的正常现象
            correct += (predicted == y_a).sum().item()
        else:
            # 对于未混合的样本，正常计算准确率
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def evaluate_model(model, test_loader, criterion, device, classes):
    """
    评估流程（不计算梯度）：
    1) 前向推理得到 logits
    2) 计算测试损失
    3) 统计总体准确率与分类别准确率
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0 for _ in classes]
    class_total = [0 for _ in classes]

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            matches = (predicted == labels).squeeze()
            for idx in range(len(labels)):
                label = labels[idx]
                class_correct[label] += matches[idx].item()
                class_total[label] += 1

    avg_loss = running_loss / total
    overall_acc = 100 * correct / total
    return avg_loss, overall_acc, class_correct, class_total


def plot_curves(history, title, save_path=None):
    """
    绘制训练曲线（损失和准确率）
    
    Args:
        history: 包含训练历史的字典
        title: 图表标题
        save_path: 保存路径（可选），如果提供则保存图片
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('损失曲线')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['test_acc'], label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('准确率曲线')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  图片已保存至: {save_path}")
    
    plt.show()


def visualize_predictions(model, test_loader, classes, device, num_images=10, save_path=None):
    """
    随机取一批测试图像，进行前向推理并可视化真值与预测。
    步骤：取 batch -> 前向 -> 反归一化 -> matplotlib 显示。
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        classes: 类别名称列表
        device: 设备
        num_images: 要显示的图像数量
        save_path: 保存路径（可选），如果提供则保存图片
    """
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    with torch.no_grad():
        outputs = model(images.to(device))
    _, predicted = torch.max(outputs, 1)
    predicted = predicted.cpu()

    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(cifar_mean, cifar_std)],
        std=[1 / s for s in cifar_std]
    )
    images = torch.stack([inv_normalize(img) for img in images]).numpy()

    plt.figure(figsize=(15, 7))
    for i in range(min(num_images, len(images))):
        ax = plt.subplot(2, (num_images + 1) // 2, i + 1)
        img = np.transpose(images[i], (1, 2, 0))
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        color = "green" if predicted[i] == labels[i] else "red"
        plt.title(f"真值: {classes[labels[i]]}\n预测: {classes[predicted[i]]}", color=color)
        plt.axis('off')
    
    # 保存图片
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  图片已保存至: {save_path}")
    
    plt.show()


def plot_lr_schedule(lr_history, title="学习率变化曲线", save_path=None):
    """
    绘制学习率变化曲线，展示学习率调度策略的效果。
    横轴：Epoch，纵轴：Learning Rate
    
    Args:
        lr_history: 学习率历史列表
        title: 图表标题
        save_path: 保存路径（可选），如果提供则保存图片
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(lr_history) + 1)
    plt.plot(epochs, lr_history, 'b-', linewidth=2, label='Learning Rate')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3, linestyle='--')
    plt.yscale('log')  # 使用对数刻度，更好地展示学习率变化
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  图片已保存至: {save_path}")
    
    plt.show()


def plot_confusion_matrix(model, test_loader, classes, device, save_path=None):
    """
    绘制混淆矩阵，展示模型在各个类别上的分类表现。
    横轴：预测类别，纵轴：真实类别
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        classes: 类别名称列表
        device: 设备
        save_path: 保存路径（可选），如果提供则保存图片
    """
    if not HAS_SKLEARN:
        print("--- 无法绘制混淆矩阵：需要安装sklearn ---")
        return
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化到0-1
    
    plt.figure(figsize=(12, 10))
    if HAS_SEABORN:
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes,
                    cbar_kws={'label': '归一化准确率'})
    else:
        # 如果没有seaborn，使用matplotlib绘制
        plt.imshow(cm_normalized, cmap='Blues', aspect='auto')
        plt.colorbar(label='归一化准确率')
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, f'{cm_normalized[i, j]:.2f}', 
                        ha='center', va='center', fontsize=9)
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        plt.yticks(range(len(classes)), classes)
    plt.xlabel('预测类别', fontsize=12, fontweight='bold')
    plt.ylabel('真实类别', fontsize=12, fontweight='bold')
    plt.title('混淆矩阵 (Confusion Matrix)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  图片已保存至: {save_path}")
    
    plt.show()
    
    # 打印易混淆的类别对
    print("\n--- 易混淆类别分析 ---")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confusion_rate = cm[i, j] / cm[i].sum() * 100
                if confusion_rate > 5:  # 只显示混淆率>5%的
                    print(f"  {classes[i]} 被误判为 {classes[j]}: {confusion_rate:.1f}% ({cm[i, j]}/{cm[i].sum()})")


def plot_hardest_examples(model, test_loader, classes, device, num_examples=8, save_path=None):
    """
    找出模型"非常自信地预测错"的样本（Top-K错误样本）。
    这些样本通常具有误导性，或者数据质量有问题。
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        classes: 类别名称列表
        device: 设备
        num_examples: 要显示的错误样本数量
        save_path: 保存路径（可选），如果提供则保存图片
    """
    model.eval()
    error_samples = []  # 存储 (image, true_label, pred_label, confidence, prob_dist)
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs_batch, labels_batch = inputs.to(device), labels.to(device)
            outputs = model(inputs_batch)
            probs = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probs, 1)
            
            # 找出预测错误的样本
            wrong_mask = (predicted != labels_batch)
            wrong_indices = torch.where(wrong_mask)[0]
            
            for idx in wrong_indices:
                error_samples.append((
                    inputs_batch[idx].cpu(),
                    labels_batch[idx].item(),
                    predicted[idx].item(),
                    confidences[idx].item(),
                    probs[idx].cpu().numpy()
                ))
    
    # 按置信度排序，找出最自信但预测错误的样本
    error_samples.sort(key=lambda x: x[3], reverse=True)
    error_samples = error_samples[:num_examples]
    
    if len(error_samples) == 0:
        print("--- 未找到错误样本 ---")
        return
    
    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(cifar_mean, cifar_std)],
        std=[1 / s for s in cifar_std]
    )
    
    plt.figure(figsize=(16, 8))
    for i, (img, true_label, pred_label, confidence, prob_dist) in enumerate(error_samples):
        ax = plt.subplot(2, 4, i + 1)
        img_denorm = inv_normalize(img)
        img_np = np.transpose(img_denorm.numpy(), (1, 2, 0))
        img_np = np.clip(img_np, 0, 1)
        plt.imshow(img_np)
        plt.title(f"真实: {classes[true_label]}\n预测: {classes[pred_label]}\n置信度: {confidence*100:.1f}%", 
                 color='red', fontsize=10)
        plt.axis('off')
    
    plt.suptitle('Top-K 错误样本 (Hardest Examples)\n模型非常自信但预测错误的样本', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  图片已保存至: {save_path}")
    
    plt.show()


def plot_ablation_study(final_results, title="消融实验对比", save_path=None):
    """
    绘制消融实验对比柱状图，展示每个改进模块带来的增益。
    
    Args:
        final_results: 实验结果字典 {实验名称: 准确率}
        title: 图表标题
        save_path: 保存路径（可选），如果提供则保存图片
    """
    if len(final_results) == 0:
        print("--- 无实验结果可绘制 ---")
        return
    
    names = list(final_results.keys())
    accuracies = list(final_results.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(names)), accuracies, color=['#3498db', '#2ecc71', '#e74c3c'][:len(names)], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 在柱子上标注数值
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 标注提升幅度
    if len(accuracies) >= 2:
        for i in range(1, len(accuracies)):
            improvement = accuracies[i] - accuracies[i-1]
            plt.plot([i-1, i], [accuracies[i-1], accuracies[i]], 'r--', linewidth=2, alpha=0.7)
            plt.text((i-1+i)/2, (accuracies[i-1]+accuracies[i])/2 + 1,
                    f'+{improvement:.2f}%', ha='center', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.xlabel('实验', fontsize=12, fontweight='bold')
    plt.ylabel('测试准确率 (%)', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(range(len(names)), names, rotation=15, ha='right')
    plt.ylim([min(accuracies) - 5, max(accuracies) + 5])
    plt.grid(alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  图片已保存至: {save_path}")
    
    plt.show()


def plot_batch_size_comparison(batch_size_results, title="批次大小对比实验", save_path=None):
    """
    绘制不同批次大小的训练结果对比图。
    在同一张图上展示不同批次大小的训练/测试准确率和损失曲线。
    
    Args:
        batch_size_results: 批次大小结果字典 {batch_size: history}
        title: 图表标题
        save_path: 保存路径（可选），如果提供则保存图片
    """
    if len(batch_size_results) == 0:
        print("--- 无批次大小对比数据 ---")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 绘制准确率曲线
    ax1 = axes[0]
    for batch_size, history in batch_size_results.items():
        epochs = range(1, len(history['test_acc']) + 1)
        ax1.plot(epochs, history['test_acc'], 'o-', linewidth=2, 
                label=f'Batch Size {batch_size} (最佳: {max(history["test_acc"]):.2f}%)', 
                markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('测试准确率对比', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3, linestyle='--')
    
    # 绘制损失曲线
    ax2 = axes[1]
    for batch_size, history in batch_size_results.items():
        epochs = range(1, len(history['test_loss']) + 1)
        ax2.plot(epochs, history['test_loss'], 'o-', linewidth=2, 
                label=f'Batch Size {batch_size}', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Test Loss', fontsize=12)
    ax2.set_title('测试损失对比', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, linestyle='--')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  图片已保存至: {save_path}")
    
    plt.show()
    
    # 打印总结
    print("\n--- 批次大小对比总结 ---")
    for batch_size, history in batch_size_results.items():
        best_acc = max(history['test_acc'])
        best_epoch = history['test_acc'].index(best_acc) + 1
        print(f"  Batch Size {batch_size}: 最佳准确率 {best_acc:.2f}% (Epoch {best_epoch})")


# --- 5. 实验主流程 ---


def create_scheduler(optimizer, scheduler_cfg: Optional[Dict]):
    if scheduler_cfg is None:
        return None
    sched_type = scheduler_cfg.get("type")
    if sched_type == "StepLR":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg.get("step_size", 20),
            gamma=scheduler_cfg.get("gamma", 0.5)
        )
    if sched_type == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_cfg.get("mode", "max"),
            factor=scheduler_cfg.get("factor", 0.5),
            patience=scheduler_cfg.get("patience", 5),
            verbose=True,
            min_lr=scheduler_cfg.get("min_lr", 1e-6)
        )
    if sched_type == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg.get("T_max", EPOCHS),
            eta_min=scheduler_cfg.get("eta_min", 1e-5)
        )
    return None


def print_class_accuracy(class_correct, class_total):
    print('--- 各类别准确率 ---')
    for i, cls in enumerate(classes):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f'类别 {cls:>5} : {acc:.2f} %')
        else:
            print(f'类别 {cls:>5} : N/A')


if __name__ == '__main__':
    print("--- 准备数据加载器... ---")
    train_set_simple = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_simple)
    train_set_medium = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_medium)  # 中等强度数据增强
    train_set_augmented = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_augmented)
    train_set_randaugment = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_randaugment)
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    loader_simple = torch.utils.data.DataLoader(
        train_set_simple, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    loader_medium = torch.utils.data.DataLoader(
        train_set_medium, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)  # 中等强度数据增强
    loader_augmented = torch.utils.data.DataLoader(
        train_set_augmented, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    loader_randaugment = torch.utils.data.DataLoader(
        train_set_randaugment, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


    experiments = [
        {
            "name": "实验1: 手写ResNet(浅层) + 中等数据增强 + AdamW",
            "model_args": {"depth": "浅层", "base_channels": 32, "dropout": 0.2, 
                          "fc_dropout": 0.15},
            "train_loader": loader_medium,
        "optimizer": "AdamW",
        "lr": 0.001,                                        # AdamW通常使用较小的学习率
        "weight_decay": 1.5e-3,
        "scheduler": {"type": "StepLR", "step_size": 12, "gamma": 0.5},
            "label_smoothing": 0.15,
            "grad_clip": 1.0,
            "epochs": 80,
            "early_stopping": True,
            "early_stopping_patience": 7
        },
        {
            "name": "实验2: 手写ResNet(18层) + 中等数据增强 + AdamW + LR调度",
            "model_args": {"depth": "18层", "base_channels": 32, "dropout": 0.3,
                          "fc_dropout": 0.25},
            "train_loader": loader_medium,
        "optimizer": "AdamW",
            "lr": 0.001,
            "weight_decay": 1.5e-3,
        "scheduler": {"type": "ReduceLROnPlateau", "mode": "max", "patience": 2, "factor": 0.5},
        "label_smoothing": 0.15,
        "grad_clip": 1.0,
        "epochs": 80,
        "early_stopping": True,
        "early_stopping_patience": 6
        },
        {
            "name": "实验3: 手写ResNet(18层) + RandAugment + CutMix + Mixup + LabelSmoothing + SAM + 余弦退火",
            # 整合SOTA方法：RandAugment（自动数据增强）+ CutMix + Mixup（混合增强组合）+ Label Smoothing + SAM优化器 + Cosine LR
            # SAM优化器（Sharpness-Aware Minimization）是ICLR 2021的杀手锏技术，通过同时最小化损失和锐度来提升泛化能力
            # 在CIFAR-10上通常能带来1-2%的提升，是ResNet-18刷榜的最强单体技术
            # 改进：同时使用CutMix和Mixup，随机交替使用，提供更强的数据增强和正则化
            # 改进：启用Label Smoothing，即使有CutMix/Mixup，Label Smoothing也能提供额外的正则化
            # 改进：增加模型容量（base_channels: 32 -> 64），提高模型表达能力
            # 
            "model_args": {"depth": "18层", "base_channels": 64, "dropout": 0.1,
                          "fc_dropout": 0.1},
            "train_loader": loader_randaugment,
            "optimizer": "SAM",
            "lr": 0.001,
            "weight_decay": 5e-4,
            "sam_rho": 0.05,
        "scheduler": {"type": "CosineAnnealingLR", "T_max": 100, "eta_min": 1e-5},
        "epochs": 100,
        "label_smoothing": 0.1,
        "grad_clip": None,
        "use_cutmix": True,
        "cutmix_alpha": 1.0,
        "use_mixup": True,
        "mixup_alpha": 0.8,
        "early_stopping": True,
        "early_stopping_patience": 12
        }
    ]

    final_results = {}

    for exp in experiments:
        print("\n" + "=" * 70)
        print(f"--- 正在开始: {exp['name']} ---")
        print("=" * 70)

        model = build_resnet(**exp["model_args"]).to(device)
        criterion = nn.CrossEntropyLoss()

        # 根据实验配置选择优化器
        optimizer_type = exp.get("optimizer", "AdamW")
        use_sam = (optimizer_type == "SAM")
        
        if use_sam:
            base_optimizer = optim.AdamW(model.parameters(), lr=exp["lr"],
                                        weight_decay=exp["weight_decay"])
            optimizer = SAM(model.parameters(), base_optimizer, 
                          rho=exp.get("sam_rho", 0.05))
            print(f"--- 使用SAM优化器 (rho={exp.get('sam_rho', 0.05)}) ---")
        else:
            optimizer = optim.AdamW(model.parameters(), lr=exp["lr"],
                                    weight_decay=exp["weight_decay"])

        scheduler = create_scheduler(optimizer, exp.get("scheduler"))
        scheduler_type = exp.get("scheduler", {}).get("type") if exp.get("scheduler") else None
        
        num_epochs = exp.get("epochs", EPOCHS)
        
        # 如果使用CosineAnnealingLR，需要更新T_max为实际使用的epoch数
        if scheduler is not None and scheduler_type == "CosineAnnealingLR":
            # 重新创建scheduler，使用正确的T_max
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=exp.get("scheduler", {}).get("eta_min", 1e-5)
            )

        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'lr': []}
        best_acc = 0.0
        best_epoch = 0

        early_stopping = exp.get("early_stopping", False)
        early_stopping_patience = exp.get("early_stopping_patience", 10)
        patience_counter = 0  # 连续不提升的epoch数

        if early_stopping:
            print(f"--- 已启用早停机制 (patience={early_stopping_patience})，防止过拟合 ---")

        start_time = time.time()
        for epoch in range(num_epochs):
            train_loss, train_acc = train_model(
                model, exp["train_loader"], optimizer, criterion, device,
                label_smoothing=exp["label_smoothing"], grad_clip=exp["grad_clip"],
                use_cutmix=exp.get("use_cutmix", False),
                cutmix_alpha=exp.get("cutmix_alpha", 1.0),
                use_mixup=exp.get("use_mixup", False),
                mixup_alpha=exp.get("mixup_alpha", 1.0),
                use_sam=use_sam
            )
            test_loss, test_acc, class_correct, class_total = evaluate_model(
                model, test_loader, criterion, device, classes)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

            if scheduler is not None:
                if scheduler_type == "ReduceLROnPlateau":
                    scheduler.step(test_acc)
                else:
                    scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)  # 记录学习率历史
            
            # 更新最佳准确率和早停计数器
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch + 1
                patience_counter = 0  # 准确率提升，重置计数器
            else:
                patience_counter += 1  # 准确率未提升，计数器+1

            print(f"Epoch {epoch+1:02d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
                  f"LR: {current_lr:.6f} | Best: {best_acc:.2f}% (Epoch {best_epoch})", end="")
            
            if early_stopping:
                # 检查过拟合指标：训练准确率很高但测试准确率停滞
                overfitting_risk = train_acc > 95.0 and (train_acc - test_acc) > 10.0
                if patience_counter >= early_stopping_patience:
                    print(f"\n{'='*70}")
                    print(f"  早停触发：测试准确率连续 {patience_counter} 个epoch未提升")
                    print(f"   当前训练准确率: {train_acc:.2f}% | 测试准确率: {test_acc:.2f}%")
                    print(f"   训练损失: {train_loss:.4f} | 测试损失: {test_loss:.4f}")
                    if overfitting_risk:
                        print(f"     检测到过拟合风险：训练-测试准确率差距: {train_acc - test_acc:.2f}%")
                    print(f"   最佳测试准确率: {best_acc:.2f}% (出现在 Epoch {best_epoch})")
                    print(f"{'='*70}")
                    break
                elif overfitting_risk and patience_counter >= early_stopping_patience // 2:
                    print(f" |  过拟合风险 (差距: {train_acc - test_acc:.1f}%)")
                else:
                    print(f" | 早停计数: {patience_counter}/{early_stopping_patience}")
            else:
                print()  # 换行

        end_time = time.time()
        print(f"--- {exp['name']} 训练完成! 总耗时: {end_time - start_time:.2f} 秒 ---")
        print(f"--- 最佳测试准确率: {best_acc:.2f}% | 出现在 Epoch {best_epoch} ---")
        print_class_accuracy(class_correct, class_total)

        # 创建保存目录
        exp_idx = experiments.index(exp) + 1
        plot_dir = f"plots/exp{exp_idx}"
        os.makedirs(plot_dir, exist_ok=True)
        
        # 基础绘图：损失和准确率曲线
        plot_curves(history, exp['name'], save_path=f"{plot_dir}/curves.png")
        
        # 学习率变化曲线
        if len(history['lr']) > 0:
            print(f"\n--- {exp['name']} 学习率变化曲线 ---")
            plot_lr_schedule(history['lr'], title=f"{exp['name']} - 学习率变化曲线", 
                           save_path=f"{plot_dir}/lr_schedule.png")
        
        # 混淆矩阵
        print(f"\n--- {exp['name']} 混淆矩阵 ---")
        plot_confusion_matrix(model, test_loader, classes, device, 
                            save_path=f"{plot_dir}/confusion_matrix.png")
        
        # Top-K 错误样本
        print(f"\n--- {exp['name']} Top-K 错误样本 ---")
        plot_hardest_examples(model, test_loader, classes, device, num_examples=8,
                            save_path=f"{plot_dir}/hardest_examples.png")
        
        # 随机预测可视化
        print(f"\n--- {exp['name']} 的预测可视化 ---")
        visualize_predictions(model, test_loader, classes, device, num_images=8,
                            save_path=f"{plot_dir}/predictions.png")

        final_results[exp['name']] = best_acc

    print("\n" + "=" * 70)
    print("--- 最终实验对比（越往后越强） ---")
    print("=" * 70)
    for name, acc in final_results.items():
        print(f"{name:<60} : {acc:.2f} %")

    names = list(final_results.keys())
    if len(names) >= 2:
        print(f"\n对比 [{names[0]}] 与 [{names[1]}] 可看到 {final_results[names[1]] - final_results[names[0]]:.2f}% 的提升。")
    if len(names) >= 3:
        print(f"最终实验（[{names[2]}]）整合了SOTA方法，相对实验2 提升了 "
              f"{final_results[names[2]] - final_results[names[1]]:.2f}%。")
        print(f"\n--- 最终实验（实验3）SOTA方法说明 ---")
        print(f"实验3整合了论文中报导的SOTA方法组合：")
        print(f"  - RandAugment: 自动数据增强，从14种增强操作中随机选择，提高数据多样性")
        print(f"  - CutMix: 混合两张图片的区域和标签，增强模型泛化能力，防止过拟合")
        print(f"  - SAM优化器: Sharpness-Aware Minimization (ICLR 2021)")
        print(f"  - Cosine Annealing LR: 余弦退火学习率调度，帮助模型精细调优")
    
    # 绘制消融实验对比图
    print("\n--- 绘制消融实验对比图 ---")
    os.makedirs("plots", exist_ok=True)
    plot_ablation_study(final_results, title="消融实验对比 - 逐步引入先进技术",
                       save_path="plots/ablation_study.png")
    
    # 批次大小对比实验
    print("\n" + "=" * 70)
    print("--- 批次大小对比实验 ---")
    print("=" * 70)
    
    batch_sizes = [64, 128, 256]  # 测试三种批次大小
    batch_size_results = {}
    
    base_exp = experiments[0]
    
    for batch_size in batch_sizes:
        print(f"\n--- 批次大小: {batch_size} ---")
        
        # 创建对应批次大小的数据加载器
        loader_batch = torch.utils.data.DataLoader(
            train_set_simple, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader_batch = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=0)
        
        model = build_resnet(**base_exp["model_args"]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=base_exp["lr"],
                                weight_decay=base_exp["weight_decay"])
        scheduler = create_scheduler(optimizer, base_exp.get("scheduler"))
        scheduler_type = base_exp.get("scheduler", {}).get("type") if base_exp.get("scheduler") else None
        
        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'lr': []}
        best_acc = 0.0
        best_epoch = 0
        
        quick_epochs = min(30, EPOCHS)
        
        start_time = time.time()
        for epoch in range(quick_epochs):
            train_loss, train_acc = train_model(
                model, loader_batch, optimizer, criterion, device,
                label_smoothing=base_exp["label_smoothing"], grad_clip=base_exp["grad_clip"],
                use_cutmix=False, cutmix_alpha=1.0,
                use_mixup=False, mixup_alpha=1.0
            )
            test_loss, test_acc, _, _ = evaluate_model(
                model, test_loader_batch, criterion, device, classes)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            if scheduler is not None:
                if scheduler_type == "ReduceLROnPlateau":
                    scheduler.step(test_acc)
                else:
                    scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch + 1
            
            print(f"Epoch {epoch+1:02d}/{quick_epochs} | "
                  f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
                  f"Best: {best_acc:.2f}% (Epoch {best_epoch})")
        
        end_time = time.time()
        print(f"--- 批次大小 {batch_size} 完成! 耗时: {end_time - start_time:.2f} 秒 ---")
        print(f"--- 最佳测试准确率: {best_acc:.2f}% (Epoch {best_epoch}) ---")
        
        batch_size_results[batch_size] = history
    
    # 绘制批次大小对比图
    print("\n--- 绘制批次大小对比图 ---")
    plot_batch_size_comparison(batch_size_results, title="批次大小对比实验 - 浅层ResNet",
                              save_path="plots/batch_size_comparison.png")
    
    print("\n实验代码执行完毕。")

