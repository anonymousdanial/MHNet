import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        outputs: dict with 'pred_logits' [B, num_queries, num_classes+1]
                           'pred_boxes' [B, num_queries, 4]
        targets: list of dicts, one per image, each with:
                 'labels' [num_objects] and 'boxes' [num_objects, 4]
        
        Returns: list of (pred_idx, target_idx) tuples for each image
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # Flatten to compute cost matrices
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [B*num_queries, num_classes+1]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B*num_queries, 4]
        
        # Concatenate all target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
        # Classification cost
        cost_class = -out_prob[:, tgt_ids]  # [B*num_queries, total_objects]
        
        # L1 cost for bounding boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # [B*num_queries, total_objects]
        
        # GIoU cost (optional but recommended)
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)  # [B*num_queries, total_objects]
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) 
                for i, j in indices]


class SetCriterion(nn.Module):
    """Loss computation using Hungarian matching"""
    def __init__(self, num_classes, matcher, weight_dict):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = ['labels', 'boxes']
    
    def forward(self, outputs, targets):
        """
        outputs: dict from model
        targets: list of dicts (one per image)
        """
        # Step 1: Get optimal matching between predictions and targets
        indices = self.matcher(outputs, targets)
        
        # Step 2: Compute losses only on matched pairs
        losses = {}
        
        # Classification loss
        losses['loss_ce'] = self.loss_labels(outputs, targets, indices)
        
        # Bounding box losses
        losses['loss_bbox'] = self.loss_boxes(outputs, targets, indices)
        losses['loss_giou'] = self.loss_giou(outputs, targets, indices)
        
        # Weighted sum
        total_loss = sum(losses[k] * self.weight_dict[k] for k in losses.keys())
        
        return total_loss, losses
    
    def loss_labels(self, outputs, targets, indices):
        """Classification loss"""
        src_logits = outputs['pred_logits']  # [B, num_queries, num_classes+1]
        
        # Get matched predictions
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        # All predictions are "no object" by default
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        
        # Set matched predictions to their true class
        target_classes[idx] = target_classes_o
        
        # Cross entropy loss
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        return loss_ce
    
    def loss_boxes(self, outputs, targets, indices):
        """L1 bounding box loss"""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]  # Matched predictions
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='mean')
        return loss_bbox
    
    def loss_giou(self, outputs, targets, indices):
        """GIoU loss"""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes)).mean()
        return loss_giou
    
    def _get_src_permutation_idx(self, indices):
        """Get batch and query indices for matched predictions"""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


# Helper function for GIoU
def generalized_box_iou(boxes1, boxes2):
    """
    Compute GIoU between two sets of boxes
    boxes: [N, 4] in format [cx, cy, w, h] normalized to [0, 1]
    """
    # Convert to [x1, y1, x2, y2]
    boxes1 = box_cxcywh_to_xyxy(boxes1)
    boxes2 = box_cxcywh_to_xyxy(boxes2)
    
    # ... (GIoU implementation - can provide if needed)
    # For now, you can use torchvision.ops.generalized_box_iou
    from torchvision.ops import generalized_box_iou as giou
    return giou(boxes1, boxes2)

def box_cxcywh_to_xyxy(x):
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)