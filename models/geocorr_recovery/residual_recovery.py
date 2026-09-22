"""Confidence-constrained residual descriptor recovery and feature loss."""

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RecoveryMLP(nn.Module):
    """Minimal ``3C -> C -> C`` residual recovery MLP with zero output init."""

    def __init__(self, feature_dim: int = 256, zero_init: bool = True) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.layers = nn.Sequential(
            nn.Linear(3 * self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        if zero_init:
            nn.init.zeros_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)

    def forward(
        self, q_dirty: Tensor, historical_feature: Tensor, confidence: Tensor
    ) -> Dict[str, Tensor]:
        if q_dirty.shape != historical_feature.shape or q_dirty.ndim != 4:
            raise ValueError("q_dirty and historical_feature must match [B,N,Vq,C]")
        if q_dirty.shape[-1] != self.feature_dim:
            raise ValueError("descriptor feature dimension does not match recovery MLP")
        if confidence.shape != q_dirty.shape[:-1] + (1,):
            raise ValueError("confidence must have shape [B,N,Vq,1]")
        inputs = torch.cat(
            [q_dirty, historical_feature, historical_feature - q_dirty], dim=-1
        )
        delta_q = self.layers(inputs)
        q_recovered = q_dirty + confidence.to(delta_q) * delta_q
        return {"delta_q": delta_q, "q_recovered": q_recovered}


def recovery_feature_loss(
    q_recovered: Tensor,
    q_clean: Tensor,
    valid_mask: Tensor,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> Tensor:
    """Masked ``1-cosine`` loss against a detached clean teacher descriptor."""
    if q_recovered.shape != q_clean.shape or q_recovered.ndim != 4:
        raise ValueError("q_recovered and q_clean must match [B,N,Vq,C]")
    if valid_mask.shape == q_recovered.shape[:-1] + (1,):
        valid_mask = valid_mask.squeeze(-1)
    if valid_mask.shape != q_recovered.shape[:-1]:
        raise ValueError("valid_mask must have shape [B,N,Vq] or [B,N,Vq,1]")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    teacher = q_clean.detach()
    loss = 1.0 - F.cosine_similarity(q_recovered, teacher, dim=-1, eps=eps)
    mask = valid_mask.bool()
    masked = loss * mask.to(loss.dtype)
    if reduction == "none":
        return masked
    if reduction == "sum":
        return masked.sum()
    if bool(mask.any().item()):
        return masked.sum() / mask.sum().to(masked.dtype)
    return q_recovered.sum() * 0.0


__all__ = ["RecoveryMLP", "recovery_feature_loss"]
