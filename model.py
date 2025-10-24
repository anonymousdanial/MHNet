import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

class SPG(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_compress = nn.Conv2d(64, 32, 1)  # Example for F1, adjust for F2
        self.gconv = nn.Conv2d(64, 64, 3, padding=1)  # Gaussian approx with 3x3
        self.fpn_convs = nn.ModuleList([nn.Conv2d(64, 256, 1) for _ in range(3)])  # Simplified FPN

    def forward(self, f1, f2):
        f1_prime = self.conv_compress(f1)
        f2_prime = self.conv_compress(f2)
        wd = self.conv_compress(f1_prime * f2_prime)
        f_star = torch.cat([self.conv_compress(f1_prime * wd), self.conv_compress(f2_prime * wd)], dim=1)
        f_g = torch.sigmoid(self.gconv(f_star))  # Gaussian blur approx
        # FPN: bottom-up C1,C2,C3 from f_g layers (simplified)
        c3 = f_g  # Placeholder
        p3 = self.fpn_convs[2](c3)
        p2 = F.interpolate(p3, scale_factor=2) + self.fpn_convs[1](c3)  # Adjust inputs
        p1 = F.interpolate(p2, scale_factor=2) + self.fpn_convs[0](c3)
        prior = self.conv_compress(torch.cat([F.interpolate(p1, size=p2.shape[2:]), p2, F.max_pool2d(p3, 2)], dim=1))
        return prior

class PRM(nn.Module):  # Simplified CapsNet mirror
    def __init__(self, in_channels):
        super().__init__()
        self.primary_caps = nn.Conv2d(in_channels, 256, 9, stride=2)  # Example
        # Add ConvCaps, DeconvCaps, EM routing (implement from capsule refs)

    def forward(self, x):
        primary = self.primary_caps(x)
        # EM routing logic here (custom impl needed)
        return primary  # Placeholder for relations

class CRFCS(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3x3 = nn.Conv2d(512, 512, 3, padding=1)

    def forward(self, roi, boundary):
        mt = nn.Conv2d(roi.shape[1], 1, 1)(roi)  # Assume C=1 for mask
        at_c = 1 - torch.sigmoid(mt)
        xt_c = self.conv3x3(at_c * boundary * roi)
        return xt_c  # Fuse with selective attn

class SS(nn.Module):
    def __init__(self, channels=512, r=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(channels, channels//r), nn.ReLU(), nn.Linear(channels//r, channels))

    def forward(self, x):
        batch_mean = x.mean(0, keepdim=True)
        diff = x - batch_mean
        diff = nn.Conv2d(x.shape[1], 256, 1)(diff)  # To 152x152x256 approx
        gmp = F.max_pool2d(diff, diff.shape[2:]).view(1, -1)
        gap = F.avg_pool2d(diff, diff.shape[2:]).view(1, -1)
        fc_out = self.mlp(gmp) + self.mlp(gap)
        g = torch.sigmoid(fc_out).view(1, -1, 1, 1)
        return x * g

class MHNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True).features
        self.f1 = vgg[:4]  # conv1
        self.f2 = vgg[4:9]  # conv2
        self.f3 = vgg[9:16]  # conv3
        self.f4 = vgg[16:23]  # conv4
        self.f5 = vgg[23:30]  # conv5
        self.spg = SPG()
        self.prm = PRM(512)
        self.crfcs = CRFCS()
        self.ss = SS()

    def forward(self, x):
        f1 = self.f1(x)
        f2 = self.f2(f1)
        f3 = self.f3(f2)
        f4 = self.f4(f3)
        f5 = self.f5(f4)
        prior = self.spg(f1, f2)
        # Enhance mainstream and PRM
        f4_enh = f4 + nn.Conv2d(256, 512, 1)(prior)  # Adjust channels
        f5_enh = f5 + nn.Conv2d(256, 512, 1)(prior)
        prm_out = self.prm(f5_enh)  # PRM branch
        mainstream = f5_enh + prm_out  # Integrate
        ss_out = self.ss(f4)  # SS on F4
        mainstream = mainstream + ss_out  # Adjust
        roi = mainstream  # RPN/ROI here in full model
        crfcs_out = self.crfcs(roi, f5)  # Boundary from F5
        return crfcs_out  # Final features for head

# Full MHNet (wrap with FasterRCNN)
def get_mhnet(num_classes=6):  # 5 classes + bg
    backbone = MHNetBackbone()
    backbone.out_channels = 512
    anchor_gen = AnchorGenerator(sizes=((32,), (64,), (128,), (256,), (512,)), aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    model = FasterRCNN(backbone, num_classes, rpn_anchor_generator=anchor_gen)
    return model

# Example usage: model = get_mhnet()
# Train/test as per paper specs