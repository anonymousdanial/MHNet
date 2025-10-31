import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class SingleMaskLoss(nn.Module):
    """
    For models that output [B, 64, H, W] but predict ONE binary mask.
    Averages the 64 channels → then applies BCE + Dice.
    """
    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits_64, target):
        """
        logits_64: [B, 64, H, W] – raw model output
        target:    [B, H, W] or [B, 1, H, W] – binary ground truth
        """
        # Ensure target is [B, 1, H, W]
        if target.dim() == 3:
            target = target.unsqueeze(1).float()

        # Average over 64 channels
        logits = logits_64.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        total = 0.0

        # BCE with logits
        if self.bce_weight > 0:
            total += self.bce_weight * self.bce(logits, target)

        # Dice on probabilities
        if self.dice_weight > 0:
            prob = torch.sigmoid(logits)
            intersection = (prob * target).sum(dim=(2, 3))
            union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            total += self.dice_weight * (1.0 - dice.mean())

        return total


if __name__ == "__main__":
    mod_out = torch.randn(16, 64, 224, 224)
    GT  = torch.randn(16, 1, 224, 224)
    criterion = SingleMaskLoss()
    loss = criterion(mod_out, GT)
    print(loss)