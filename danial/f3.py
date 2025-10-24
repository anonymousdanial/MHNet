# f3.py
import torch
import torch.nn as nn

class F3Conv(nn.Module):
    def __init__(self, in_channels=64, out_channels=128):
        super().__init__()
        # 3x3 convolution, stride 1, padding 1 to preserve spatial size
        # You can change stride if you want downsampling
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f3):
        x = self.conv(f3)
        x = self.bn(x)
        x = self.relu(x)
        return x


# ----- Quick test -----
if __name__ == "__main__":
    F3 = torch.randn(1, 64, 28, 28)  # example backbone feature map
    model = F3Conv(in_channels=64, out_channels=128)
    out = model(F3)
    print("F3 conv output shape:", out.shape)
