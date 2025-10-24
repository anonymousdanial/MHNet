import torch
import torch.nn as nn
import torch.nn.functional as F

class ReverseAttention(nn.Module):
    """
    Simple reverse attention block.
    a_t = 1 - sigmoid(M)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        M = self.conv(x)
        at = 1 - torch.sigmoid(M)
        out = x * at  # element-wise modulation
        return out


class PreFCS(nn.Module):
    """
    Pre-FCS module: minimal pipeline before FCS.
    Takes feature3 and PRM output, fuses, applies reverse attention,
    RPN, and passes to ROI.
    """
    def __init__(self):
        super().__init__()
        # Convolution to map feature3 channels to match PRM output channels (64)
        self.feature3_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        # Reverse attention module
        self.attention = ReverseAttention(64)
        
        # Minimal RPN placeholder (3x3 conv)
        self.rpn = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Minimal ROI placeholder (just pooling for now)
        self.roi_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Storage for fused branch
        self.branch = None

    def forward(self, feature3, prm_out):
        """
        feature3: [B, 128, 28, 28]
        prm_out: [B, 1, 64, 56, 56]
        """
        # 1. Reduce feature3 channels and upsample to match PRM spatial size
        f3 = self.feature3_conv(feature3)  # [B, 64, 28, 28]
        f3_up = F.interpolate(f3, size=prm_out.shape[-2:], mode='bilinear', align_corners=False)
        
        # 2. Reduce PRM from 5D to 4D if needed
        if prm_out.dim() == 5:  # [B, 1, C, H, W]
            prm_out_4d = prm_out.squeeze(1)  # [B, C, H, W]
        else:
            prm_out_4d = prm_out
        
        # 3. Element-wise addition
        fused = f3_up + prm_out_4d
        self.branch = fused  # save the branch

        # 4. Reverse attention
        attn_out = self.attention(fused)

        # 5. RPN
        rpn_out = self.rpn(attn_out)

        # 6. Combine attention + RPN for ROI input
        roi_input = attn_out + rpn_out
        roi_out = self.roi_pool(roi_input)

        return roi_out



if __name__ == "__main__":
    feature3 = torch.randn(1, 128, 28, 28)
    prm_out  = torch.randn(1, 1, 64, 56, 56)

    pre_fcs = PreFCS()
    roi_features = pre_fcs(feature3, prm_out)

    print("ROI output shape:", roi_features.shape)
    print("Saved branch shape:", pre_fcs.branch.shape)
