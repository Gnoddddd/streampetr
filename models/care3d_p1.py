"""CARE-3D P1 sparse evidence routing blocks.

P1 is deliberately narrower than a full detector rewrite. A frozen P0
vulnerability predictor decides how strongly a target query should be protected,
while a sparse router retrieves at most ``top_k`` complementary source tokens.
The routed residual is applied to the final decoder query used by the
classification head; the box-regression path is left unchanged in the formal P1
activation experiment.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.care3d import SparseEvidenceRouter


class CARE3DP1ScoreRouter(nn.Module):
    """Risk-gated sparse residual routing for one object query per sample.

    ``SparseEvidenceRouter`` starts with a zero residual scale, so a newly
    constructed P1 router is exact identity even when fault routing is active.
    The clean path is an explicit bypass and therefore remains exact identity
    after arbitrary P1 training.
    """

    def __init__(
        self,
        object_dim: int = 256,
        source_dim: int = 256,
        vulnerability_dim: int = 3,
        hidden_dim: int = 256,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.object_dim = int(object_dim)
        self.source_dim = int(source_dim)
        self.vulnerability_dim = int(vulnerability_dim)
        self.top_k = int(top_k)
        self.router = SparseEvidenceRouter(
            object_dim=self.object_dim,
            source_dim=self.source_dim,
            num_protocols=self.vulnerability_dim,
            topk_sources=self.top_k,
            hidden_dim=int(hidden_dim),
        )

    def forward(
        self,
        object_features: Tensor,
        source_features: Tensor,
        source_reliability: Tensor,
        vulnerability: Tensor,
        boundary_crossing_logits: Tensor,
        protocol_index: Tensor,
        *,
        fault_active: bool = True,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if object_features.ndim != 2 or object_features.shape[-1] != self.object_dim:
            raise ValueError("object_features must have shape [B, object_dim]")
        batch = object_features.shape[0]
        if source_features.ndim != 3 or source_features.shape[0] != batch \
                or source_features.shape[-1] != self.source_dim:
            raise ValueError("source_features must have shape [B, S, source_dim]")
        if source_reliability.shape != source_features.shape[:2]:
            raise ValueError("source_reliability must have shape [B, S]")
        if vulnerability.shape != (batch, self.vulnerability_dim):
            raise ValueError("vulnerability shape changed")
        if boundary_crossing_logits.shape != (batch, self.vulnerability_dim):
            raise ValueError("boundary_crossing_logits shape changed")
        if protocol_index.shape != (batch,):
            raise ValueError("protocol_index must have shape [B]")
        protocol_index = protocol_index.long()
        if torch.any(protocol_index < 0) or torch.any(protocol_index >= self.vulnerability_dim):
            raise ValueError("protocol_index outside frozen P0 heads")

        if not fault_active:
            return object_features, {
                "risk_probability": object_features.new_zeros((batch,)),
                "correction": object_features.new_zeros(object_features.shape),
                "clean_bypass": object_features.new_ones((batch,), dtype=torch.bool),
            }

        route = self.router(
            object_features.unsqueeze(1),
            source_features.unsqueeze(1),
            source_reliability.unsqueeze(1),
            vulnerability.unsqueeze(1),
        )
        routed_raw = route["enhanced_features"][:, 0]
        correction = routed_raw - object_features
        active_logit = boundary_crossing_logits.gather(1, protocol_index[:, None])[:, 0]
        risk_probability = torch.sigmoid(active_logit)
        routed = object_features + risk_probability[:, None] * correction
        return routed, {
            "risk_probability": risk_probability,
            "correction": correction,
            "clean_bypass": object_features.new_zeros((batch,), dtype=torch.bool),
            "topk_indices": route["route_indices"][:, 0],
            "route_weights": route["route_weights"][:, 0],
            "route_residual": route["route_residual"][:, 0],
            "route_scale": route["route_scale"],
        }


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    mask = mask.bool()
    if mask.any():
        return value[mask].mean()
    return value.sum() * 0.0


def p1_score_routing_loss(
    *,
    routed_query: Tensor,
    clean_query: Tensor,
    fault_query: Tensor,
    routed_logits: Tensor,
    fault_logits: Tensor,
    target_class: Tensor,
    clean_score: Tensor,
    fault_score: Tensor,
    fault_topk_threshold: Tensor,
    cross_topk: Tensor,
    score_weight: float = 1.0,
    boundary_weight: float = 1.0,
    query_weight: float = 0.25,
    retained_weight: float = 0.5,
    non_target_weight: float = 0.25,
    boundary_margin: float = 0.01,
) -> Dict[str, Tensor]:
    """Frozen P1 training objective.

    Positive samples are P0 ``cross_topk`` events. They receive clean-score,
    Top-K-boundary and representation-restoration objectives. Retained samples
    receive a full-class logit-drift penalty. On positive samples, increases of
    non-target logits are penalized to discourage class/FP inflation.
    """
    if routed_query.shape != clean_query.shape or routed_query.shape != fault_query.shape:
        raise ValueError("query tensor shapes must match")
    if routed_logits.shape != fault_logits.shape or routed_logits.ndim != 2:
        raise ValueError("classification logit shapes must match [B, classes]")
    batch = routed_logits.shape[0]
    if target_class.shape != (batch,):
        raise ValueError("target_class must have shape [B]")
    positive = cross_topk.bool()
    retained = ~positive

    target_class = target_class.long()
    routed_target = torch.sigmoid(
        routed_logits.gather(1, target_class[:, None])[:, 0]
    )
    score_loss = _masked_mean(
        F.smooth_l1_loss(routed_target, clean_score, reduction="none"), positive
    )
    boundary_loss = _masked_mean(
        F.relu(fault_topk_threshold + float(boundary_margin) - routed_target), positive
    )
    query_error = F.smooth_l1_loss(routed_query, clean_query, reduction="none").mean(-1)
    query_loss = _masked_mean(query_error, positive)

    logit_drift = (routed_logits - fault_logits).abs().mean(-1)
    retained_loss = _masked_mean(logit_drift, retained)

    class_mask = torch.ones_like(routed_logits, dtype=torch.bool)
    class_mask.scatter_(1, target_class[:, None], False)
    non_target_increase = F.relu(routed_logits - fault_logits)
    non_target_increase = (
        non_target_increase.masked_fill(~class_mask, 0.0).sum(-1)
        / class_mask.sum(-1).clamp_min(1)
    )
    non_target_loss = _masked_mean(non_target_increase, positive)

    total = (
        float(score_weight) * score_loss
        + float(boundary_weight) * boundary_loss
        + float(query_weight) * query_loss
        + float(retained_weight) * retained_loss
        + float(non_target_weight) * non_target_loss
    )
    return {
        "total": total,
        "score": score_loss,
        "boundary": boundary_loss,
        "query": query_loss,
        "retained": retained_loss,
        "non_target": non_target_loss,
        "routed_target_score": routed_target.detach(),
        "positive_count": positive.sum().detach(),
        "retained_count": retained.sum().detach(),
    }
