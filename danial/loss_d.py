import torch
import torch.nn as nn
import torch.nn.functional as F

class ForegroundBackgroundLoss(nn.Module):
    """
    Rewards activations inside mask, penalizes activations outside mask.
    """
    def __init__(self, fg_weight=1.0, bg_weight=1.0, smooth=1e-6):
        super().__init__()
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight
        self.smooth = smooth

    def forward(self, logits, target):
        """
        logits: [B, 1, H, W]  raw model output
        target: [B, 1, H, W]  binary mask (0/1)
        """
        prob = torch.sigmoid(logits)

        # Foreground region (where mask == 1)
        fg = (prob * target).sum() / (target.sum() + self.smooth)

        # Background region (where mask == 0)
        bg = (prob * (1 - target)).sum() / ((1 - target).sum() + self.smooth)

        # We want: fg HIGH, bg LOW
        loss = self.fg_weight * (1 - fg) + self.bg_weight * bg
        return loss


# Example usage:
if __name__ == "__main__":
    # Create a fake binary mask
    GT = torch.randint(0, 2, (16, 1, 224, 224)).float()

    # Generate logits that perfectly match this mask
    logits = torch.logit(GT.clamp(0.001, 0.999))  # convert to ideal logits

    criterion = ForegroundBackgroundLoss()
    loss = criterion(logits, GT)

    print("Loss:", loss.item())
