import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import numpy as np

class SAE(nn.Module):
    """
    Tiny Object Self-Adaptive Enhancement (SAE) Module.
    
    This module performs self-adaptive feature enhancement for weak small targets
    in shallow features (e.g., S3 layer). It uses internal feature cues (scale, response,
    high-frequency entropy) to generate a mask for tiny candidates and applies
    spatial (edge-weighted) and frequency-domain boosts guided by the mask.
    
    Args:
        in_channels (int): Input channel dimension (default: 256 for S3).
        alpha (float): Residual fusion weight (default: 0.2).
    
    Input: [B, C, H, W]
    Output: [B, C, H, W] (enhanced features)
    TO_SAE模块的自适应性主要体现在mask生成和增强强度的动态调整两个相互关联的环节中。
    首先，在mask生成过程中，自适应通过特征内部的多重线索融合实现动态定位：模块从输入特征中实时提取尺度敏感度、局部响应峰值以及高频稀疏性等内部信号，
    通过它们的乘积和非线性激活自动生成候选区域的概率图，并进一步二值化成聚焦mask。
    这种设计无需外部监督或固定阈值，而是根据当前特征的统计特性（如低变异区域的自然突出）自发“点亮”潜在小目标位置，确保定位过程响应输入的变异性。
    其次，在增强强度调整中，自适应体现在基于mask密度的动态缩放机制：模块先计算mask的全局密度（亮像素比例），
    然后通过一个小型多层感知网络实时映射成强度因子，并进一步线性缩放成一个范围内的乘子，用于指导边缘加权和频域放大的幅度。
    这种密度驱动的调整让增强操作“因地制宜”：mask密集区（暗示小目标聚集）获得更强补强，稀疏区则温和处理，避免过度干扰背景，
    从而整体响应特征的局部模式，实现渐进的自愈效果。这种双重自适应让模块在低样本或噪声场景下更鲁棒，显著提升浅层特征的细节针对性。
    """
    def __init__(self, in_channels, alpha):
        super(SAE, self).__init__()
        self.alpha = alpha
        self.in_channels = in_channels
        
        # Self-Detection Head: Lightweight localization for tiny regions
        self.scale_conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.scale_conv.weight, mode='fan_out', nonlinearity='relu')
        self.resp_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        # Dynamic gamma MLP (based on mask density)
        self.mlp_gamma = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        nn.init.kaiming_normal_(self.mlp_gamma[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.mlp_gamma[2].weight, mode='fan_out', nonlinearity='relu')
        
        # Sobel edge detector（边缘检测器） (fixed kernel for x-direction; y can be added if needed) 
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, bias=False, padding=1)
        self.sobel_x.weight.data = sobel_x_kernel
        self.sobel_x.weight.requires_grad = False  # Fixed kernel
        
        # Threshold for mask binarization (can be made learnable if needed)
        self.mask_threshold = 0.3 # 这是个需要验证的超参数

    def _compute_high_freq_entropy(self, x_gray):
        """
        Simplified local high-frequency entropy estimation via FFT.
        Args:
            x_gray: [B, 1, H, W]
        Returns:
            entropy_hf: [B, 1, H, W] (local entropy map)
        """
        B, _, H, W = x_gray.shape
        X_freq = fft.fft2(x_gray)  # [B, 1, H, W] complex
        # Extract high-frequency magnitude (outer 3/4 region)
        high_freq_mag = torch.abs(X_freq[:, :, H//4:, W//4:])
        # Pad back to full size for entropy computation
        full_mag = torch.zeros_like(X_freq)
        full_mag[:, :, H//4:, W//4:] = high_freq_mag
        # Simplified entropy: -p log p (per channel, averaged)
        p = torch.abs(full_mag) / (torch.abs(full_mag).sum(dim=(2,3), keepdim=True) + 1e-8)
        entropy = -(p * torch.log(p + 1e-8)).sum(dim=(2,3), keepdim=True)  # Global per-batch entropy, but localized via pooling
        # Localize with avg pool
        entropy_hf = F.avg_pool2d(entropy, kernel_size=5, stride=1, padding=2)
        return entropy_hf

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input features [B, C, H, W]
        Returns:
            Enhanced features [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Step 1: Self-Detection - Generate tiny candidate mask
        scale_feat = torch.sigmoid(self.scale_conv(x))  # [B, 1, H, W] scale score (low variance -> high for small)
        resp_feat = self.resp_pool(x).mean(dim=1, keepdim=True)  # [B, 1, H, W] response score
        # High-freq entropy (self-cue for sparse edges)
        x_gray = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        entropy_hf = self._compute_high_freq_entropy(x_gray)  # [B, 1, H, W]
        # Combined mask
        """
        mask计算的自适应性：没错，mask是通过特征内部的多重线索（尺度分数、响应分数和高频熵）实时乘积融合、sigmoid激活和阈值二值化生成的。
        这种“自线索驱动”方式让mask动态响应当前输入的统计特性（如低变异或高激活区域），无需外部信息的引导，仅依赖特征图内部/自身的信息即可完成计算，
        就能自动定位small目标位置，确保定位过程灵活适应不同图像的弱信号模式。
        """
        mask_logit = scale_feat * resp_feat * entropy_hf  # [B, 1, H, W]
        mask = torch.sigmoid(mask_logit)  # [B, 1, H, W]
        mask = (mask > self.mask_threshold).float()  # Binarized [B, 1, H, W]
        
        # Step 2: Adaptive Boost Branches
        # Spatial Boost: Edge-weighted enhancement
        """
        增强强度的自适应调整：基于mask的密度（亮像素比例）通过MLP映射成动态gamma因子（范围(1.0,1.5)），进一步指导空间边缘加权和频域高频放大的幅度。
        高密度mask（暗示密集小目标）会触发更强增强，低密度则温和处理，避免过度干扰背景或孤立噪声。
        """
        edge = torch.abs(self.sobel_x(x_gray))  # [B, 1, H, W]
        # Dynamic gamma based on mask density (self-adaptive strength)
        mask_density = mask.sum(dim=(2, 3), keepdim=True) / (H * W)  # [B, 1, 1, 1]
        gamma = self.mlp_gamma(mask_density)  # [B, 1, 1, 1], range [0,1] -> scale to [1.0, 1.5]
        gamma = 1.0 + 0.5 * gamma  # Boost range [1.0, 1.5]
        spatial_boost = mask * edge * gamma  # [B, 1, H, W]
        spatial_boost = spatial_boost.expand(B, C, H, W) * x  # Broadcast to [B, C, H, W]
        
        
        # Frequency Boost: High-freq amplification guided by mask 抑制低频放大高频
        X_freq = fft.fft2(x, dim=(-2, -1))  # Per-channel FFT [B, C, H, W] complex
        
        # Lightly suppress low-freq (global structure preservation)
        low_freq_mask = torch.zeros_like(X_freq)
        low_freq_mask[:, :, :H//4, :W//4] = X_freq[:, :, :H//4, :W//4] * 0.8
        high_freq_part = X_freq - low_freq_mask  # Isolated high-freq
        
        # Amplify high-freq magnitude (preserve phase)
        amp = torch.abs(high_freq_part) * 1.2  # 幅度
        phase = torch.angle(high_freq_part)  # 相位
        high_freq_enhanced = amp * torch.exp(1j * phase)
        # Mask-guided (zero out non-tiny regions in high-freq)
        mask_expanded = mask.expand(B, C, H, W)
        high_freq_enhanced *= mask_expanded  # [B, C, H, W] complex
        # Reconstruct full spectrum and IFFT
        X_freq_enhanced = low_freq_mask + high_freq_enhanced
        freq_boost = fft.ifft2(X_freq_enhanced, dim=(-2, -1)).real  # [B, C, H, W]
        freq_boost = torch.relu(freq_boost)  # Non-negative
        
        # Step 3: Fusion
        boost = (spatial_boost + freq_boost) / 2.0  # Average boosts
        out = x + self.alpha * boost  # Residual connection
        return torch.relu(out)  # Final activation for positivity
