import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMapLoss(nn.Module):
    """
    Robust composite loss for feature-map comparison [B, C, H, W].
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        l1_weight: float = 0.0,
        cosine_weight: float = 0.0,
        pearson_weight: float = 0.0,
        gram_weight: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.weights = {
            "mse": mse_weight,
            "l1": l1_weight,
            "cosine": cosine_weight,
            "pearson": pearson_weight,
            "gram": gram_weight,
        }
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction
        self.active = {k for k, w in self.weights.items() if w > 0.0}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _gram_matrix(feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        f = feat.view(B, C, H * W)
        g = torch.bmm(f, f.transpose(1, 2))
        return g.div_(C * H * W)

    @staticmethod
    def _pearson_corr(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """
        Channel-wise Pearson correlation -> [B, C]
        """
        f1 = f1.view(f1.size(0), f1.size(1), -1)   # [B, C, N]
        f2 = f2.view(f2.size(0), f2.size(1), -1)

        mean1 = f1.mean(dim=2, keepdim=True)
        mean2 = f2.mean(dim=2, keepdim=True)

        std1 = f1.std(dim=2, keepdim=True) + 1e-8
        std2 = f2.std(dim=2, keepdim=True) + 1e-8

        cov = ((f1 - mean1) * (f2 - mean2)).mean(dim=2)           # [B, C]

        # <<< FIX: squeeze the singleton dim before division >>>
        return cov / (std1.squeeze(2) * std2.squeeze(2))          # [B, C]

    # ------------------------------------------------------------------ #
    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        if feat1.shape != feat2.shape:
            raise ValueError(f"Shape mismatch: {feat1.shape} vs {feat2.shape}")

        total = 0.0
        terms = {}

        # ------------------- MSE -------------------
        if "mse" in self.active:
            mse = F.mse_loss(feat1, feat2, reduction="none").mean(dim=[1, 2, 3])  # [B]
            terms["mse"] = mse
            total = total + self.weights["mse"] * mse

        # ------------------- L1 --------------------
        if "l1" in self.active:
            l1 = F.l1_loss(feat1, feat2, reduction="none").mean(dim=[1, 2, 3])   # [B]
            terms["l1"] = l1
            total = total + self.weights["l1"] * l1

        # ------------------- Cosine ----------------
        if "cosine" in self.active:
            f1 = feat1.view(feat1.size(0), feat1.size(1), -1)
            f2 = feat2.view(feat2.size(0), feat2.size(1), -1)
            cos = F.cosine_similarity(f1, f2, dim=2)          # [B, C]
            cos_loss = (1.0 - cos).mean(dim=1)                # [B]
            terms["cosine"] = cos_loss
            total = total + self.weights["cosine"] * cos_loss

        # ------------------- Pearson ---------------
        if "pearson" in self.active:
            corr = self._pearson_corr(feat1, feat2)           # [B, C]
            pearson_loss = (1.0 - corr).mean(dim=1)           # [B]
            terms["pearson"] = pearson_loss
            total = total + self.weights["pearson"] * pearson_loss

        # ------------------- Gram -------------------
        if "gram" in self.active:
            g1 = self._gram_matrix(feat1)
            g2 = self._gram_matrix(feat2)
            gram_loss = F.mse_loss(g1, g2, reduction="none").mean(dim=[1, 2])  # [B]
            terms["gram"] = gram_loss
            total = total + self.weights["gram"] * gram_loss

        # ------------------- Final reduction ----------
        if self.reduction == "mean":
            total = total.mean()
        # else: sum over batch

        # Attach per-term scalars for logging
        for name, t in terms.items():
            setattr(total, f"{name}_loss", t.mean().item())

        return total


# ----------------------------------------------------------------------
# Demo (replace the whole file with this)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    B, C, H, W = 16, 64, 224, 224
    a = torch.randn(B, C, H, W)
    b = a.clone() + torch.randn_like(a) * 0.5

    loss_fn = FeatureMapLoss(
        mse_weight=1.0,
        cosine_weight=0.5,
        pearson_weight=0.3,
        gram_weight=0.1,
        reduction="mean",
    )

    loss = loss_fn(a, b)
    print(f"Total loss: {loss.item():.6f}")
    print(f"MSE:      {loss.mse_loss:.6f}")
    print(f"Cosine:   {loss.cosine_loss:.6f}")
    print(f"Pearson:  {loss.pearson_loss:.6f}")
    print(f"Gram:     {loss.gram_loss:.6f}")