"""Simple detection training script that uses the project's
Hungarian matcher and model. This is a small, self-contained
trainer that expects a COCO-like annotation JSON and images.

Usage example:
    python train-box.py --annotations annotation.json --images-root dasatet/COD10K-v2/Train/Images --epochs 5
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from danial import model as model_module, loss_h
from danial.box_dataloader import BoxDataset, collate_fn


def coco_bbox_to_detr(box, img_w, img_h):
    # COCO bbox: [x_min, y_min, w, h] in pixels -> DETR: [cx, cy, w, h] normalized
    x, y, w, h = box
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    return [cx, cy, w / img_w, h / img_h]


def build_targets_from_annotations(ann_dicts):
    # ann_dicts is list of annotation_dicts for a batch element
    targets = []
    for ann in ann_dicts:
        img_w = ann.get('width') or 1
        img_h = ann.get('height') or 1
        labels = []
        boxes = []
        for a in ann.get('annotations', []):
            bbox = a.get('bbox')
            if bbox is None:
                continue
            boxes.append(coco_bbox_to_detr(bbox, img_w, img_h))
            # convert category_id (COCO is 1-indexed) to 0-indexed
            cat = a.get('category_id')
            if cat is None:
                cat = 0
            labels.append(cat - 1 if isinstance(cat, int) else int(cat) - 1)

        if len(labels) == 0:
            # empty target
            targets.append({'labels': torch.zeros((0,), dtype=torch.int64),
                            'boxes': torch.zeros((0, 4), dtype=torch.float32)})
        else:
            targets.append({'labels': torch.tensor(labels, dtype=torch.int64),
                            'boxes': torch.tensor(boxes, dtype=torch.float32)})
    return targets


class LogitsAdapter(nn.Module):
    """Adapts model's output logits to desired number of classes."""
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # We'll initialize the input size in forward when we see the actual shape
        self.adapter = None
    
    def forward(self, x):
        # Initialize adapter on first forward pass when we know input size
        if self.adapter is None:
            in_features = x.shape[-1]  # Get last dimension size
            # Add +1 to output features for background class
            print(f"Initializing adapter: in_features={in_features}, out_features={self.num_classes + 1}")
            self.adapter = nn.Linear(in_features, self.num_classes + 1).to(x.device)
        
        orig_shape = x.shape
        # Flatten all but last dimension, apply adapter, restore shape
        x = x.view(-1, x.shape[-1])
        x = self.adapter(x)
        new_shape = list(orig_shape[:-1]) + [self.num_classes + 1]
        return x.view(*new_shape)

def train_one_epoch(model, criterion, optimizer, dataloader, device, logits_adapter=None):
    model.train()
    running_loss = 0.0
    for i, (images, ann_list) in enumerate(dataloader):
        images = images.to(device)
        targets = build_targets_from_annotations(ann_list)

        # forward
        outputs = model(images)
        
        
        # Adapt logits if needed
        pred_logits = outputs['pred_logits']
        print(f"pred_logits shape before adapter: {pred_logits.shape}")
        if logits_adapter is not None:
            pred_logits = logits_adapter(pred_logits)
            print(f"pred_logits shape after adapter: {pred_logits.shape}")
            
        mod_out = {
            'pred_logits': pred_logits,
            'pred_boxes': outputs['pred_boxes']
        }

        loss_dict = criterion(mod_out, targets)
        loss = loss_dict.get('loss_total') if isinstance(loss_dict, dict) else loss_dict

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f"Step {i+1}/{len(dataloader)} - loss: {loss.item():.4f}")

    avg_loss = running_loss / max(1, len(dataloader))
    return avg_loss


def save_training_log(log_path, epoch, avg_loss, lr, best_loss):
    """Save training metrics to CSV file"""
    import csv
    import os
    
    header = ['epoch', 'avg_loss', 'learning_rate', 'best_loss']
    row = [epoch, avg_loss, lr, best_loss]
    
    # Create file with header if it doesn't exist
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    
    # Append the row
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', required=True, help='Path to COCO annotations file')
    parser.add_argument('--images-root', default=None, help='Root path to images (defaults to annotation file dir)')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', default='models/box', help='where to save checkpoints')
    parser.add_argument('--export-dir', default=None, help='directory to save the final model and config')
    parser.add_argument('--num-classes', type=int, default=5, help='number of classes (must match model cls_head.out_features)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # dataset and loader
    dataset = BoxDataset(args.annotations, images_root=args.images_root, target_size=(224, 224))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)

    # model, adapter, criterion, optimizer
    net = model_module.Model().to(device)
    logits_adapter = LogitsAdapter(args.num_classes).to(device) if args.num_classes != 6 else None
    
    # Note: HungarianMatcher expects num_classes to be the number of object classes
    # (it will add +1 internally for background)
    criterion = loss_h.HungarianMatcher(
        num_classes=args.num_classes,  # HungarianMatcher adds +1 internally for background
        matcher_cost_class=1,
        matcher_cost_bbox=5,
        matcher_cost_giou=2,
        loss_ce=1,
        loss_bbox=5,
        loss_giou=2,
        eos_coef=0.1,
    )
    
    # Include adapter parameters in optimization if used
    params = list(net.parameters())
    if logits_adapter is not None:
        params.extend(list(logits_adapter.parameters()))
    optimizer = optim.AdamW(params, lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(net, criterion, optimizer, loader, device, logits_adapter)
        print(f"Epoch {epoch+1}/{args.epochs} - avg_loss: {avg_loss:.4f}")

        # save checkpoint
        ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'adapter_state_dict': logits_adapter.state_dict() if logits_adapter else None,
            'loss': avg_loss,
        }
        torch.save(ckpt, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        torch.save(ckpt, os.path.join(args.save_dir, 'last.pth'))
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, os.path.join(args.save_dir, 'best.pth'))
            
    # Export final model if requested
    if args.export_dir:
        os.makedirs(args.export_dir, exist_ok=True)
        
        # Save the complete model (architecture + weights)
        model_path = os.path.join(args.export_dir, 'model_complete.pt')
        if logits_adapter is not None:
            # If using adapter, create a wrapped model that includes both
            class WrappedModel(nn.Module):
                def __init__(self, base_model, adapter):
                    super().__init__()
                    self.base_model = base_model
                    self.adapter = adapter
                
                def forward(self, x):
                    base_output = self.base_model(x, return_all=True)  # Ensure dictionary output
                    if isinstance(base_output, dict):
                        base_output['pred_logits'] = self.adapter(base_output['pred_logits'])
                        return base_output
                    else:
                        # If base model returns tensor, wrap it in dict
                        return {
                            'pred_logits': self.adapter(base_output),
                            'features': base_output
                        }
            
            export_model = WrappedModel(net, logits_adapter)
        else:
            export_model = net
            
        # Save model in both TorchScript and state_dict formats
        try:
            # Try to save TorchScript version (may fail if model isn't scriptable)
            scripted_model = torch.jit.script(export_model)
            scripted_model.save(os.path.join(args.export_dir, 'model_scripted.pt'))
        except Exception as e:
            print(f"Warning: Could not save TorchScript model: {e}")
        
        # Always save regular state_dict
        torch.save({
            'model_state_dict': net.state_dict(),
            'adapter_state_dict': logits_adapter.state_dict() if logits_adapter else None,
            'num_classes': args.num_classes,
            'model_config': {
                'base_classes': 5,  # original model output
                'target_classes': args.num_classes,
                'using_adapter': logits_adapter is not None
            }
        }, model_path)
        
        # Save a config/readme file
        with open(os.path.join(args.export_dir, 'model_info.txt'), 'w') as f:
            f.write(f"""Model Configuration
=================
Base Model: MHNet
Input Size: 224x224
Number of Classes: {args.num_classes}
Using Class Adapter: {logits_adapter is not None}

Training Details
--------------
Learning Rate: {args.lr}
Batch Size: {args.batch_size}
Final Loss: {best_loss:.4f}

Usage Example
------------
import torch
from danial import model

# Load complete model
model = torch.load('model_complete.pt')
model.eval()

# For inference
# 1. Load and preprocess image to tensor (1, 3, 224, 224)
# 2. Call model:
outputs = model(image)
# 3. outputs will contain:
#    - pred_logits: [1, 100, {args.num_classes + 1}] (includes background class)
#    - pred_boxes: [1, 100, 4] (normalized cx, cy, w, h)
#    - recovered_features: feature maps
#    - mask_output: segmentation mask
""")
        
        print(f"\nExported model and config to {args.export_dir}/")
        print(f"- Full model saved as 'model_complete.pt'")
        print(f"- Config and usage saved in 'model_info.txt'")


if __name__ == '__main__':
    main()

