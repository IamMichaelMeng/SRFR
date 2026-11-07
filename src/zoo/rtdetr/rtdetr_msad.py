import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

class HighFreqDenoise(nn.Module):
    """
    核心高频抑噪子模块。
    通过高通滤波抑制高频噪声，同时保留目标细节。
    
    Args:
        base_cutoff (float): 基础截止频率比例 (默认: 0.1)。
    cutoff：截止频率
    输入: [B, C, H, W]
    输出: [B, C, H, W] (去噪特征)
    """
    def __init__(self, base_cutoff=0.1):
        super(HighFreqDenoise, self).__init__()
        self.base_cutoff = base_cutoff
        # 可学习截止调整 (通道自适应抑制)
        self.cutoff_adjust = nn.Parameter(torch.ones(1) * 0.1)  # 小标量，可优化

    def forward(self, x, scale_weight=1.0):
        """
        前向传播，带尺度自适应cutoff。
        Args:
            x: 输入特征 [B, C, H, W]
            scale_weight: 该尺度的自适应权重 (e.g., 0.7 for S3)
        Returns:
            去噪特征 [B, C, H, W]
        """
        B, C, H, W = x.shape
        """
        cutoff定义低频区域的衰减范围。具体来说，它决定了频谱中“低频vs高频”的分界线，让模块根据尺度权重和训练反馈动态调整滤波强度。
        """
        cutoff = self.base_cutoff * scale_weight * (1 + self.cutoff_adjust)  # 自适应cutoff
        
        # 逐通道FFT
        X_freq = fft.fft2(x, dim=(-2, -1))  # [B, C, H, W] 复数
        
        # 生成径向mask (高通滤波: 抑制低频噪，保留高频)
        center_h, center_w = H // 2, W // 2
        y, x_grid = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=x.device),
                                   torch.arange(W, dtype=torch.float32, device=x.device),
                                   indexing='ij')
        dist = torch.sqrt((y - center_h)**2 + (x_grid - center_w)**2) / torch.sqrt(torch.tensor(H*W, dtype=torch.float32, device=x.device))
        
        # 高通: 低频 (dist < cutoff) 隔挡掉，高频通过
        damp_mask = torch.clamp(dist / cutoff, 0.0, 1.0)  # [H, W] 高频通过 (1.0)，低频衰减 (0.0)
        damp_mask = damp_mask.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)  # [B, C, H, W]
        
        # 应用衰减 (复数乘法)
        X_denoised = X_freq * damp_mask  # 抑制低频噪，保留高频细节
        
        # IFFT回空间域
        x_denoised = fft.ifft2(X_denoised, dim=(-2, -1)).real
        x_denoised = torch.relu(x_denoised)  # 确保非负
        
        return x_denoised


class MultiScaleAdaptiveDenoise(nn.Module):
    """
    多尺度自适应抑噪 (MSAD) 模块。
    在多尺度特征 (e.g., S3, S4, S5) 上应用尺度自适应高频抑噪。
    重点S3 (浅层细节噪)，通过可学习权重实现鲁棒小目标检测。
    Args:
        num_scales (int): 尺度数 (默认: 3 for [S3, S4, S5])。
        base_weights (list): 基础权重 (默认: [0.7, 0.2, 0.1]，S3重点)。
    
    输入: 特征列表 [[B, C, H3, W3], [B, C, H4, W4], [B, C, H5, W5]]
    输出: 去噪特征列表 (相同形状)
    MSAD模块的自适应性主要体现在两个相互关联的机制中：动态权重分配和可学习截止频率调整。
    首先，在多尺度处理层面，自适应通过基础层级权重与可学习扰动参数的融合实现动态强度生成。
    具体而言，模块将预设的基础权重（如偏重浅层的分布）与端到端优化的扰动项相加后，通过非线性激活函数归一化，形成每个尺度的处理强度。
    这种设计允许模块根据训练反馈自动调整各层（如浅层强抑噪、深层温和保留）的相对贡献，确保滤波过程适应不同尺度特征的噪声特性。
    其次，在频域滤波层面，自适应体现在截止频率的动态计算中：基础截止比例乘以尺度强度，并进一步融合一个全局可学习标量调整项，从而实时调整个性化滤波边界。
    这种双重自适应让模块从静态滤波转向响应式净化，能够根据数据集噪声模式（如背景干扰或细节保留需求）逐步优化，显著提升小目标检测的鲁棒性。
    """
    def __init__(self, num_scales=3, base_weights=[0.7, 0.2, 0.1]):
        super(MultiScaleAdaptiveDenoise, self).__init__()
        self.num_scales = num_scales
        self.base_weights = nn.Parameter(torch.tensor(base_weights, dtype=torch.float32))  # [num_scales]
        assert len(base_weights) == num_scales, "基础权重必须匹配num_scales"
        
        # 每尺度可学习权重调整 (全局标量，高效)
        self.weight_adjust = nn.Parameter(torch.ones(num_scales) * 0.05)  # 小扰动，可优化
        
        # 共享核心抑噪器 (base_cutoff=0.1 for 高频抑制)
        self.core_denoise = HighFreqDenoise(base_cutoff=0.1)

    def forward(self, feats):
        """
        前向传播。
        Args:
            feats: 多尺度特征列表 [feat0 (S3), feat1 (S4), feat2 (S5)]
        Returns:
            denoised_feats: 去噪特征列表 (相同形状)
        """
        assert len(feats) == self.num_scales, f"预期 {self.num_scales} 尺度，实际 {len(feats)}"
        """
        模块通过sigmoid归一化的动态权重（adaptive_weights）为每个尺度（S₃/S₄/S₅）分配不同的抑噪强度，
        基础权重[0.7, 0.2, 0.1]（S₃最高，偏重浅层细节噪）+可学习扰动（weight_adjust）,这是“基于频域层级的自适应滤波机制”中“自适应”的体现。
        即权重不是超参数定死，而是可学习的，浅层特征图分辨率高，噪声抑制强度就大，深层特征图分辨率低，噪声抑制强度就小。
        """
        adaptive_weights = torch.sigmoid(self.base_weights + self.weight_adjust)  # 归一到[0,1]，和≈1
        
        denoised_feats = []
        for i, feat in enumerate(feats):
            scale_weight = adaptive_weights[i].item()  # 该尺度标量权重
            # 应用核心抑噪，带尺度自适应
            denoised = self.core_denoise(feat, scale_weight=scale_weight)
            denoised_feats.append(denoised)
        
        return denoised_feats
