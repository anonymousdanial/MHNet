import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveWeightedAttention(nn.Module):
    def __init__(self, in_channels1=64, in_channels2=256, compress_channels=32, gaussian_kernel_size=3):
        super(SelectiveWeightedAttention, self).__init__()

        # Step 1: channel compression
        self.gr1 = nn.Conv2d(in_channels1, compress_channels, kernel_size=1)
        self.gr2 = nn.Conv2d(in_channels2, compress_channels, kernel_size=1)

        # Step 2: weighted regulator (mutual attention)
        self.gr_wd = nn.Conv2d(compress_channels, compress_channels, kernel_size=1)

        # Step 3: residual fusion after mutual weighting
        self.fuse_conv1 = nn.Conv2d(compress_channels, compress_channels, kernel_size=1)
        self.fuse_conv2 = nn.Conv2d(compress_channels, compress_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(compress_channels * 2, compress_channels, kernel_size=1)

        # Step 4: Gaussian convolution (blur effect)
        self.gaussian = nn.Conv2d(
            compress_channels, compress_channels,
            kernel_size=gaussian_kernel_size, padding=gaussian_kernel_size // 2,
            bias=False, groups=compress_channels  # depthwise blur
        )
        self.sigmoid = nn.Sigmoid()

        # Initialize Gaussian weights
        self._init_gaussian_weights(gaussian_kernel_size, compress_channels)

    def _init_gaussian_weights(self, kernel_size, channels, sigma=1.0):
        """Initialize Gaussian kernel weights."""
        import math
        # Create 2D Gaussian kernel
        ax = torch.arange(kernel_size) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(channels, 1, 1, 1)
        with torch.no_grad():
            self.gaussian.weight.copy_(kernel)

    def forward(self, F1, F2):
        # Align spatial sizes before anything else
        if F1.size(2) != F2.size(2) or F1.size(3) != F2.size(3):
            # Upsample F2 to match F1’s spatial size
            F2 = F.interpolate(F2, size=(F1.size(2), F1.size(3)), mode='bilinear', align_corners=False)

        # Step 1: compress channels
        F01 = self.gr1(F1)
        F02 = self.gr2(F2)

        # Step 2: mutual weighting (elementwise product + conv)
        wd = self.gr_wd(F01 * F02)

        # Step 3: apply mutual weights & residual fusion
        F1_weighted = self.fuse_conv1(F01 * wd)
        F2_weighted = self.fuse_conv2(F02 * wd)
        fused = torch.cat([F1_weighted, F2_weighted], dim=1)
        fused = self.out_conv(fused)

        # Step 4: Gaussian blur + sigmoid
        out = self.gaussian(fused)
        out = self.sigmoid(out)

        return out



if __name__ == "__main__":
    swa = SelectiveWeightedAttention(in_channels1=64, in_channels2=128)
    F1 = torch.randn(1, 64, 112, 112)
    F2 = torch.randn(1, 128, 56, 56)
    
    out = swa(F1, F2)
    print(out.shape) 
