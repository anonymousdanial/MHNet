import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLoss(nn.Module):
    """
    Combined loss for segmentation tasks.
    Combines BCE with Logits and Dice Loss for robust training.
    
    Args:
        bce_weight: Weight for BCE loss component (default: 0.5)
        dice_weight: Weight for Dice loss component (default: 0.5)
        pos_weight: Positive class weight for handling imbalance (default: None)
        smooth: Smoothing factor for Dice loss to avoid division by zero (default: 1e-6)
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        
        # Store pos_weight as a parameter or buffer so it moves with the model
        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.tensor([pos_weight]))
        else:
            self.pos_weight = None
    
    def dice_loss(self, pred, target):
        """
        Compute Dice loss.
        pred: logits (before sigmoid)
        target: ground truth mask [0, 1]
        """
        pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice_score = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice_score
    
    def forward(self, pred, target):
        """
        pred: model output (logits) - shape: (B, C, H, W) or (B, H, W)
        target: ground truth mask - shape: (B, C, H, W) or (B, H, W), values in [0, 1]
        """
        if self.pos_weight is not None:
            bce_loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=self.pos_weight)
        else:
            bce_loss = F.binary_cross_entropy_with_logits(pred, target)
            
        dice_loss = self.dice_loss(pred, target)
        
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return total_loss
    

# Usage examples:
if __name__ == "__main__":
    # 1. Basic usage (balanced classes)
    # criterion = SegmentationLoss()

    # # 2. With class imbalance (e.g., 10x more negative samples)
    # criterion = SegmentationLoss(pos_weight=10.0)

    # # 3. More weight on Dice for better boundaries
    # criterion = SegmentationLoss(bce_weight=0.3, dice_weight=0.7)

    # # 4. Pure BCE (like your current setup)
    # criterion = SegmentationLoss(bce_weight=1.0, dice_weight=0.0)

    # # 5. Pure Dice
    # criterion = SegmentationLoss(bce_weight=0.0, dice_weight=1.0)
    
    seg_head = nn.Sequential(
		nn.Conv2d(64, 1, kernel_size=1),
		nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
	    ).to(torch.device('cpu'))
    
    criterion = SegmentationLoss(bce_weight=0.5, dice_weight=0.5, pos_weight=2.0)
    seg_pred = seg_head(fused_feat)
    seg_loss = criterion(seg_pred, targets)