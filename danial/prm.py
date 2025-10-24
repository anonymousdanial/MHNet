# prm.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- Primary Capsule -----
class PrimaryCaps(nn.Module):
    def __init__(self, in_channels, capsule_dim, num_capsules, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.conv = nn.Conv2d(in_channels, num_capsules * capsule_dim,
                              kernel_size=kernel_size, stride=stride, padding=padding)

    def squash(self, x):
        norm = torch.norm(x, dim=2, keepdim=True)
        scale = (norm**2) / (1 + norm**2)
        return scale * x / (norm + 1e-8)

    def forward(self, x):
        out = self.conv(x)
        B, C, H, W = out.size()
        out = out.view(B, self.num_capsules, self.capsule_dim, H, W)
        return self.squash(out)


# ----- ConvCaps Layer -----
class ConvCaps(nn.Module):
    def __init__(self, in_caps, in_dim, out_caps, out_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.route_weights = nn.Parameter(
            torch.randn(out_caps, in_caps, in_dim, out_dim, kernel_size, kernel_size) * 0.1
        )
        self.stride = stride
        self.padding = padding

    def squash(self, x):
        norm = torch.norm(x, dim=2, keepdim=True)
        scale = (norm**2) / (1 + norm**2)
        return scale * x / (norm + 1e-8)

    def forward(self, x):
        B, in_caps, in_dim, H, W = x.size()
        out_caps, _, _, out_dim, k, _ = self.route_weights.size()
        u_hat = []
        for i in range(in_caps):
            w = self.route_weights[:, i].reshape(out_caps * out_dim, in_dim, k, k)
            xi = x[:, i]
            conv = F.conv2d(xi, w, stride=self.stride, padding=self.padding)
            conv = conv.view(B, out_caps, out_dim, conv.size(2), conv.size(3))
            u_hat.append(conv)
        u_hat = torch.stack(u_hat, dim=2)
        v = self.squash(u_hat.mean(dim=2))
        return v


# ----- DeconvCaps Layer -----
class DeconvCaps(nn.Module):
    def __init__(self, in_caps, in_dim, out_caps, out_dim, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_caps * in_dim, out_caps * out_dim,
                                         kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        B, in_caps, in_dim, H, W = x.size()
        x = x.view(B, in_caps * in_dim, H, W)
        out = self.deconv(x)
        B, C, H, W = out.size()
        out = out.view(B, -1, C // (B * 1), H, W)
        return out


# ----- PRM Block -----
class PRM(nn.Module):
    def __init__(self, f3_channels, spg_channels, fused_channels=128,
                 num_capsules=8, capsule_dim=8, out_caps=8, out_dim=16):
        super().__init__()
        # SPG 1x1 convolution to match F3 channels
        self.conv_spg = nn.Conv2d(spg_channels, fused_channels, kernel_size=1)

        # Capsule layers
        self.primary = PrimaryCaps(in_channels=fused_channels, capsule_dim=capsule_dim, num_capsules=num_capsules)
        self.conv_caps = ConvCaps(in_caps=num_capsules, in_dim=capsule_dim, out_caps=out_caps, out_dim=out_dim)
        self.deconv_caps = DeconvCaps(in_caps=out_caps, in_dim=out_dim, out_caps=out_caps, out_dim=capsule_dim)

    def forward(self, f3_conv, fspg):
        # Step 1: SPG 1x1 convolution
        spg_conv = self.conv_spg(fspg)

        # Step 2: Downsample SPG to match F3 spatial size
        spg_resized = F.interpolate(spg_conv, size=f3_conv.shape[2:], mode='bilinear', align_corners=False)

        # Step 3: Element-wise addition
        contrast_feat = f3_conv + spg_resized

        # Step 4: Capsule network
        x = self.primary(contrast_feat)
        x = self.conv_caps(x)
        x = self.deconv_caps(x)
        return x


# ----- Quick test -----
if __name__ == "__main__":
    F3_conv = torch.randn(1, 128, 28, 28)  # F3 already convoluted
    F25 = torch.randn(1, 32, 112, 112)     # SPG output
    model = PRM(f3_channels=128, spg_channels=32, fused_channels=128)
    out = model(F3_conv, F25)
    print("PRM output shape:", out.shape)
