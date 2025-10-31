import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMapLoss(nn.Module):
    """
    Loss for comparing:
        - [B, 64, H, W]  → reduced to [B, 1, H, W] via mean over channels
        - [B, 1,  H, W]  → used as-is
    Then applies MSE, L1, Cosine, Pearson, Gram on the aligned [B, 1, H, W] maps.

    Parameters
    ----------
    reduction : 'mean' or 'sum'
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
    def _reduce_and_align(feat64: torch.Tensor, feat1: torch.Tensor):
        """
        Input:
            feat64: [B, 64, H, W]
            feat1:  [B,  1, H, W]
        Output:
            both:   [B,  1, H, W]
        """
        if feat64.shape[0] != feat1.shape[0] or feat64.shape[2:] != feat1.shape[2:]:
            raise ValueError(f"Spatial/batch mismatch: {feat64.shape} vs {feat1.shape}")

        # Reduce 64 channels → 1 via mean
        feat64_reduced = feat64.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        return feat64_reduced, feat1

    @staticmethod
    def _gram_matrix(feat: torch.Tensor) -> torch.Tensor:
        # feat: [B, 1, H, W]
        B, C, H, W = feat.shape
        f = feat.view(B, C, H * W)
        g = torch.bmm(f, f.transpose(1, 2))
        return g.div_(C * H * W)  # [B, 1, 1]

    @staticmethod
    def _pearson_corr(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        # f1, f2: [B, 1, H, W] → flatten spatial
        f1 = f1.view(f1.size(0), -1)  # [B, N]
        f2 = f2.view(f2.size(0), -1)

        mean1 = f1.mean(dim=1, keepdim=True)
        mean2 = f2.mean(dim=1, keepdim=True)
        std1 = f1.std(dim=1, keepdim=True) + 1e-8
        std2 = f2.std(dim=1, keepdim=True) + 1e-8

        cov = ((f1 - mean1) * (f2 - mean2)).mean(dim=1)  # [B]
        return cov / (std1.squeeze(1) * std2.squeeze(1))  # [B]

    # ------------------------------------------------------------------ #
    def forward(self, feat64: torch.Tensor, feat1: torch.Tensor) -> torch.Tensor:
        """
        feat64: [B, 64, H, W]
        feat1:  [B,  1, H, W]
        """
        # Reduce 64 → 1 and align
        f64, f1 = self._reduce_and_align(feat64, feat1)  # both [B, 1, H, W]

        total = 0.0
        terms = {}

        # ------------------- MSE -------------------
        if "mse" in self.active:
            mse = F.mse_loss(f64, f1, reduction="none").mean(dim=[1, 2, 3])  # [B]
            terms["mse"] = mse
            total = total + self.weights["mse"] * mse

        # ------------------- L1 --------------------
        if "l1" in self.active:
            l1 = F.l1_loss(f64, f1, reduction="none").mean(dim=[1, 2, 3])   # [B]
            terms["l1"] = l1
            total = total + self.weights["l1"] * l1

        # ------------------- Cosine ----------------
        if "cosine" in self.active:
            # flatten spatial: [B, 1, H*W]
            cos = F.cosine_similarity(f64.view(f64.size(0), -1), f1.view(f1.size(0), -1), dim=1)
            cos_loss = 1.0 - cos  # [B]
            terms["cosine"] = cos_loss
            total = total + self.weights["cosine"] * cos_loss

        # ------------------- Pearson ---------------
        if "pearson" in self.active:
            corr = self._pearson_corr(f64, f1)  # [B]
            pearson_loss = 1.0 - corr
            terms["pearson"] = pearson_loss
            total = total + self.weights["pearson"] * pearson_loss

        # ------------------- Gram -------------------
        if "gram" in self.active:
            g1 = self._gram_matrix(f64)  # [B,1,1]
            g2 = self._gram_matrix(f1)
            gram_loss = F.mse_loss(g1, g2, reduction="none").mean(dim=[1, 2])  # [B]
            terms["gram"] = gram_loss
            total = total + self.weights["gram"] * gram_loss

        # ------------------- Final reduction ----------
        if self.reduction == "mean":
            total = total.mean()
        # else: sum

        for name, t in terms.items():
            setattr(total, f"{name}_loss", t.mean().item())

        return total


# ----------------------------------------------------------------------
# Demo – Exactly your shapes
# ----------------------------------------------------------------------
if __name__ == "__main__":
    feat64 = torch.randn(16, 64, 224, 224)   # e.g. encoder output
    feat1  = torch.randn(16, 1, 224, 224)    # e.g. ground truth

    loss_fn = FeatureMapLoss(
        mse_weight=1.0,
        cosine_weight=0.5,
        pearson_weight=0.3,
        gram_weight=0.1,
        reduction="mean",
    )

    loss = loss_fn(feat64, feat1)
    print(f"Total loss: {loss.item():.6f}")
    print(f"MSE:      {loss.mse_loss:.6f}")
    print(f"Cosine:   {loss.cosine_loss:.6f}")
    print(f"Pearson:  {loss.pearson_loss:.6f}")
    print(f"Gram:     {loss.gram_loss:.6f}")