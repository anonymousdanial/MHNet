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
        # PRM conv will be created on-demand in forward if PRM channels != feature channels
        self.prm_conv = None
        
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
        # Accept either [B, 1, C, H, W] or [B, S, C, H, W].
        # If S==1 we squeeze that dim; if S>1 we merge S and C into channels (S*C).
        if prm_out.dim() == 5:
            B, S, C, H, W = prm_out.shape
            if S == 1:
                prm_out_4d = prm_out.squeeze(1)  # [B, C, H, W]
            else:
                # merge S and C into channel dimension -> [B, S*C, H, W]
                prm_out_4d = prm_out.reshape(B, S * C, H, W)
        else:
            prm_out_4d = prm_out
        # If PRM channels don't match feature channels, map PRM -> feature channels with 1x1 conv
        prm_ch = prm_out_4d.shape[1]
        target_ch = f3_up.shape[1]
        if prm_ch != target_ch:
            # Create or recreate conv if input channels changed
            if (self.prm_conv is None) or (hasattr(self.prm_conv, 'in_channels') and self.prm_conv.in_channels != prm_ch):
                # create and register conv layer
                conv = nn.Conv2d(prm_ch, target_ch, kernel_size=1)
                # assign to module so parameters are registered
                self.prm_conv = conv
                # try to move to same device as prm_out_4d
                try:
                    self.prm_conv.to(prm_out_4d.device)
                except Exception:
                    pass
            prm_out_4d = self.prm_conv(prm_out_4d)
        
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
    batch = 8
    feature3 = torch.randn(batch, 128, 28, 28)
    prm_out  = torch.randn(batch, 8, 8, 56, 56)

    pre_fcs = PreFCS()
    roi_features = pre_fcs(feature3, prm_out)

    print("ROI output shape:", roi_features.shape)
    print("Saved branch shape:", pre_fcs.branch.shape)
