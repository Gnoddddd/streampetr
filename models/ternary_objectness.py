"""Present/absent/unobserved objectness and supervision."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

PRESENT = 0
ABSENT = 1
UNOBSERVED = 2
NUM_STATES = 3


class TernaryObjectnessHead(nn.Module):
    """Small query head predicting present, absent, or unobserved."""

    def __init__(self, embed_dims: int = 256, hidden_dims: Optional[int] = None) -> None:
        super().__init__()
        hidden = hidden_dims or embed_dims
        self.net = nn.Sequential(
            nn.Linear(embed_dims, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, NUM_STATES),
        )

    def forward(self, query_features: Tensor) -> Tensor:
        return self.net(query_features)


def build_ternary_targets(
    num_queries: int,
    pos_inds: Tensor,
    neg_inds: Tensor,
    observability: Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """Build exact soft targets from the method formulation.

    Matched queries are ``present``. For unmatched queries the target is
    ``O * absent + (1-O) * unobserved``. Queries outside the sampler output are
    assigned zero weight.
    """

    targets = torch.zeros((num_queries, NUM_STATES), dtype=dtype, device=device)
    weights = torch.zeros((num_queries,), dtype=dtype, device=device)
    if pos_inds.numel() > 0:
        targets[pos_inds, PRESENT] = 1.0
        weights[pos_inds] = 1.0
    if neg_inds.numel() > 0:
        obs = observability.detach().clamp(0.0, 1.0)[neg_inds]
        targets[neg_inds, ABSENT] = obs
        targets[neg_inds, UNOBSERVED] = 1.0 - obs
        weights[neg_inds] = 1.0
    return targets, weights


def observability_conditioned_background_weights(
    label_weights: Tensor,
    neg_inds: Tensor,
    observability: Tensor,
    floor: float = 0.0,
) -> Tensor:
    """Prevent low-observability unmatched queries becoming hard background."""

    output = label_weights.clone()
    if neg_inds.numel() == 0:
        return output
    floor = min(max(float(floor), 0.0), 1.0)
    obs = observability.detach().clamp(0.0, 1.0)[neg_inds]
    output[neg_inds] = output[neg_inds] * (floor + (1.0 - floor) * obs)
    return output


class ObservabilityConditionedTernaryLoss(nn.Module):
    """Soft cross-entropy for ternary objectness targets."""

    def __init__(self, loss_weight: float = 1.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.label_smoothing = float(label_smoothing)

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        weights: Optional[Tensor] = None,
        avg_factor: Optional[float] = None,
    ) -> Tensor:
        if logits.shape != targets.shape or logits.shape[-1] != NUM_STATES:
            raise ValueError(
                f"Expected matching [...,3] tensors, got {logits.shape} and "
                f"{targets.shape}"
            )
        if self.label_smoothing > 0.0:
            smoothing = min(max(self.label_smoothing, 0.0), 1.0)
            targets = (1.0 - smoothing) * targets + smoothing / NUM_STATES
        loss = -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        if weights is not None:
            loss = loss * weights
            normalizer = weights.sum().clamp_min(1.0)
        else:
            normalizer = loss.new_tensor(float(loss.numel())).clamp_min(1.0)
        if avg_factor is not None:
            normalizer = loss.new_tensor(float(max(avg_factor, 1.0)))
        return self.loss_weight * loss.sum() / normalizer
