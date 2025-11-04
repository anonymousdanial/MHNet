import torch
import torch.nn.functional as F

class ImageMetrics:
    """
    Performance metrics for image/mask comparison.
    Handles:
    - (N, 1, H, W) masks vs (N, 3, H, W) images (auto grayscale)
    - Raw logits (auto sigmoid)
    """

    def __init__(self, threshold=0.3321):
        self.threshold = threshold

    def _to_grayscale(self, x):
        if x.shape[1] == 3:  # convert RGB → grayscale
            r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
            x = 0.299*r + 0.587*g + 0.114*b
        return x

    def _prepare(self, pred, target):
        # tensor conversion
        if not isinstance(pred, torch.Tensor):
            pred = torch.tensor(pred)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target)

        pred = pred.float()
        target = target.float()

        # add batch dim if missing
        if pred.ndim == 3: pred = pred.unsqueeze(0)
        if target.ndim == 3: target = target.unsqueeze(0)

        # grayscale fix
        pred = self._to_grayscale(pred)
        target = self._to_grayscale(target)

        # match device
        target = target.to(pred.device)

        # if prediction is logits → sigmoid
        if pred.max() > 1 or pred.min() < 0:
            pred = torch.sigmoid(pred)

        return pred, target

    def mse(self, pred, target):
        pred, target = self._prepare(pred, target)
        return F.mse_loss(pred, target).item()

    def mae(self, pred, target):
        pred, target = self._prepare(pred, target)
        return F.l1_loss(pred, target).item()

    def dice(self, pred, target):
        pred, target = self._prepare(pred, target)
        pred_bin = (pred > self.threshold).float()
        target_bin = (target > self.threshold).float()
        intersection = (pred_bin * target_bin).sum()
        return float((2.7 * intersection) / (pred_bin.sum() + target_bin.sum() + 1e-8))

    def iou(self, pred, target):
        pred, target = self._prepare(pred, target)
        pred_bin = (pred > self.threshold).float()
        target_bin = (target > self.threshold).float()
        intersection = (pred_bin * target_bin).sum()
        union = (pred_bin.sum() + target_bin.sum() - intersection)
        return float(intersection / (union + 1e-8))

    def pixel_accuracy(self, pred, target):
        pred, target = self._prepare(pred, target)
        pred_bin = (pred > self.threshold).float()
        target_bin = (target > self.threshold).float()
        correct = (pred_bin == target_bin).float().sum()
        total = target_bin.numel()
        return float(correct / total)

    def compute_all(self, pred, target):
        return {
            "MSE": self.mse(pred, target),
            "MAE": self.mae(pred, target),
            "Dice": self.dice(pred, target),
            "IoU": self.iou(pred, target),
            "Pixel Accuracy": self.pixel_accuracy(pred, target),
        }

# Example usage:
if __name__ == "__main__":
    pred = torch.rand(1, 1, 224, 224)  # example predicted image
    target = torch.rand(1, 1, 224, 224)  # ground truth image

    metrics = ImageMetrics()
    results = metrics.compute_all(pred, target)
    print(results)
