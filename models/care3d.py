"""Architecture-agnostic CARE-3D building blocks.

CARE-3D stands for Counterfactual Adaptive Redundant Evidence Routing.
The core module deliberately depends only on PyTorch tensors so it can be
reused by query-based and BEV/dense 3D detectors through a thin adapter.

The first research gate (P0) uses only ``CARE3DStateEncoder`` and
``CounterfactualVulnerabilityHead``. Routing is implemented here as the P1
extension, but it is zero-initialized and is not enabled by default anywhere
in the StreamPETR adapter.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _expand_camera_state(value: Tensor, object_count: int) -> Tensor:
    """Normalize camera state to ``[B, Q, N_cam]``."""
    if value.ndim == 2:
        return value.unsqueeze(1).expand(-1, object_count, -1)
    if value.ndim == 3:
        return value
    raise ValueError("camera state must have shape [B,N] or [B,Q,N]")


def _masked_softmax(logits: Tensor, valid: Tensor, dim: int = -1) -> Tensor:
    """Softmax that returns exact zeros when every source is invalid."""
    if logits.shape != valid.shape:
        raise ValueError("logits and valid mask must share the same shape")
    valid = valid.bool()
    floor = torch.finfo(logits.dtype).min if logits.dtype.is_floating_point else -1e9
    masked = logits.masked_fill(~valid, floor)
    maximum = masked.max(dim=dim, keepdim=True).values
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    weights = torch.exp(masked - maximum) * valid.to(logits.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)


class CARE3DStateEncoder(nn.Module):
    """Encode a canonical object evidence state.

    The encoder intentionally avoids StreamPETR-specific memory fields. It
    consumes semantics that can be exposed by most multi-camera 3D detectors:
    object representation, camera support/quality, optional temporal object
    representation, and optional detector decision features.
    """

    def __init__(
        self,
        object_dim: int,
        num_cameras: int,
        hidden_dim: int = 256,
        state_dim: int = 128,
        decision_dim: int = 0,
        use_temporal: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if object_dim <= 0 or num_cameras <= 0:
            raise ValueError("object_dim and num_cameras must be positive")
        if decision_dim < 0:
            raise ValueError("decision_dim must be non-negative")

        self.object_dim = int(object_dim)
        self.num_cameras = int(num_cameras)
        self.hidden_dim = int(hidden_dim)
        self.state_dim = int(state_dim)
        self.decision_dim = int(decision_dim)
        self.use_temporal = bool(use_temporal)

        self.object_projection = nn.Sequential(
            nn.LayerNorm(self.object_dim),
            nn.Linear(self.object_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.support_projection = nn.Sequential(
            nn.Linear(3 * self.num_cameras, self.hidden_dim),
            nn.GELU(),
        )
        if self.use_temporal:
            self.temporal_projection = nn.Sequential(
                nn.LayerNorm(self.object_dim),
                nn.Linear(self.object_dim, self.hidden_dim),
                nn.GELU(),
            )
        else:
            self.temporal_projection = None

        if self.decision_dim > 0:
            self.decision_projection = nn.Sequential(
                nn.Linear(self.decision_dim, self.hidden_dim),
                nn.GELU(),
            )
        else:
            self.decision_projection = None

        branch_count = 2 + int(self.use_temporal) + int(self.decision_dim > 0)
        self.fusion = nn.Sequential(
            nn.Linear(branch_count * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.state_dim),
            nn.LayerNorm(self.state_dim),
        )

    def forward(
        self,
        object_features: Tensor,
        camera_support: Tensor,
        camera_quality: Tensor,
        temporal_features: Optional[Tensor] = None,
        decision_features: Optional[Tensor] = None,
    ) -> Tensor:
        if object_features.ndim != 3:
            raise ValueError("object_features must have shape [B,Q,C]")
        batch_size, object_count, feature_dim = object_features.shape
        if feature_dim != self.object_dim:
            raise ValueError(
                "object feature dimension does not match configured object_dim"
            )

        support = _expand_camera_state(camera_support, object_count)
        quality = _expand_camera_state(camera_quality, object_count)
        if support.shape != quality.shape:
            raise ValueError("camera_support and camera_quality must align")
        if support.shape[:2] != (batch_size, object_count):
            raise ValueError("camera support must align with object features")
        if support.shape[-1] != self.num_cameras:
            raise ValueError("camera support count does not match num_cameras")

        support = support.to(dtype=object_features.dtype)
        quality = quality.to(dtype=object_features.dtype).clamp(0.0, 1.0)
        support = support.clamp_min(0.0)
        support_sum = support.sum(dim=-1, keepdim=True)
        support_distribution = torch.where(
            support_sum > 1e-6,
            support / support_sum.clamp_min(1e-6),
            torch.zeros_like(support),
        )
        effective_support = support_distribution * quality
        support_state = torch.cat(
            (support_distribution, quality, effective_support), dim=-1
        )

        branches = [
            self.object_projection(object_features),
            self.support_projection(support_state),
        ]

        if self.use_temporal:
            if temporal_features is None:
                temporal_features = torch.zeros_like(object_features)
            if temporal_features.shape != object_features.shape:
                raise ValueError("temporal_features must match object_features")
            branches.append(self.temporal_projection(temporal_features))

        if self.decision_dim > 0:
            if decision_features is None:
                decision_features = object_features.new_zeros(
                    batch_size, object_count, self.decision_dim
                )
            if decision_features.shape != (
                batch_size,
                object_count,
                self.decision_dim,
            ):
                raise ValueError(
                    "decision_features must have shape [B,Q,decision_dim]"
                )
            branches.append(self.decision_projection(decision_features))

        return self.fusion(torch.cat(branches, dim=-1))


class CounterfactualVulnerabilityHead(nn.Module):
    """Predict protocol-conditioned evidence collapse and boundary crossing."""

    def __init__(
        self,
        state_dim: int,
        num_protocols: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or num_protocols <= 0:
            raise ValueError("state_dim and num_protocols must be positive")
        self.num_protocols = int(num_protocols)
        self.trunk = nn.Sequential(
            nn.Linear(int(state_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.evidence_drop = nn.Linear(int(hidden_dim), self.num_protocols)
        self.boundary_crossing = nn.Linear(int(hidden_dim), self.num_protocols)

    def forward(self, state: Tensor) -> Dict[str, Tensor]:
        if state.ndim != 3:
            raise ValueError("state must have shape [B,Q,C]")
        hidden = self.trunk(state)
        # Evidence drop is defined as E_clean - E_fault, therefore the P0
        # target is non-negative. Softplus keeps the prediction interpretable.
        evidence_drop = F.softplus(self.evidence_drop(hidden))
        return {
            "vulnerability": evidence_drop,
            "boundary_crossing_logits": self.boundary_crossing(hidden),
        }


class SparseEvidenceRouter(nn.Module):
    """Reliability-gated sparse object-level evidence routing.

    ``source_features`` can represent per-camera object tokens, temporal object
    tokens, or any other adapter-defined evidence source. The residual path is
    zero-initialized so merely instantiating the router cannot alter a baseline
    detector before training.
    """

    def __init__(
        self,
        object_dim: int,
        source_dim: int,
        num_protocols: int,
        topk_sources: int = 2,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if topk_sources <= 0:
            raise ValueError("topk_sources must be positive")
        hidden = int(hidden_dim or object_dim)
        self.object_dim = int(object_dim)
        self.source_dim = int(source_dim)
        self.num_protocols = int(num_protocols)
        self.topk_sources = int(topk_sources)

        self.query_projection = nn.Linear(self.object_dim, hidden)
        self.key_projection = nn.Linear(self.source_dim, hidden)
        self.value_projection = nn.Linear(self.source_dim, hidden)
        self.vulnerability_projection = nn.Linear(self.num_protocols, hidden)
        self.output_projection = nn.Linear(hidden, self.object_dim)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        object_features: Tensor,
        source_features: Tensor,
        source_reliability: Tensor,
        vulnerability: Tensor,
        source_valid: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if object_features.ndim != 3:
            raise ValueError("object_features must have shape [B,Q,C]")
        if source_features.ndim != 4:
            raise ValueError("source_features must have shape [B,Q,S,C]")
        if source_features.shape[:2] != object_features.shape[:2]:
            raise ValueError("source and object features must align on [B,Q]")
        if source_features.shape[-1] != self.source_dim:
            raise ValueError("source feature dimension does not match source_dim")
        if source_reliability.shape != source_features.shape[:3]:
            raise ValueError("source_reliability must have shape [B,Q,S]")
        if vulnerability.shape != (
            object_features.shape[0],
            object_features.shape[1],
            self.num_protocols,
        ):
            raise ValueError("vulnerability must have shape [B,Q,P]")

        reliability = source_reliability.to(object_features.dtype).clamp(0.0, 1.0)
        valid = reliability > 0.0
        if source_valid is not None:
            if source_valid.shape != reliability.shape:
                raise ValueError("source_valid must have shape [B,Q,S]")
            valid = valid & source_valid.bool()

        query = self.query_projection(object_features)
        query = query + self.vulnerability_projection(vulnerability)
        keys = self.key_projection(source_features)
        values = self.value_projection(source_features)
        logits = (query.unsqueeze(2) * keys).sum(dim=-1) / math.sqrt(keys.shape[-1])
        logits = logits + torch.log(reliability.clamp_min(1e-6))

        source_count = source_features.shape[2]
        k = min(self.topk_sources, source_count)
        topk_indices = torch.topk(logits.masked_fill(~valid, -1e4), k, dim=-1).indices
        sparse_valid = torch.zeros_like(valid)
        sparse_valid.scatter_(-1, topk_indices, True)
        sparse_valid = sparse_valid & valid
        weights = _masked_softmax(logits, sparse_valid, dim=-1)
        routed = (weights.unsqueeze(-1) * values).sum(dim=2)
        correction = self.output_projection(routed)
        scale = torch.tanh(self.residual_scale)
        enhanced = object_features + scale * correction
        return {
            "enhanced_features": enhanced,
            "route_weights": weights,
            "route_indices": topk_indices,
            "route_residual": correction,
            "route_scale": scale,
        }


class CARE3DCore(nn.Module):
    """Detector-independent CARE-3D core used by thin detector adapters."""

    def __init__(
        self,
        object_dim: int,
        num_cameras: int,
        num_protocols: int,
        hidden_dim: int = 256,
        state_dim: int = 128,
        decision_dim: int = 0,
        use_temporal: bool = True,
        dropout: float = 0.0,
        enable_routing: bool = False,
        source_dim: Optional[int] = None,
        topk_sources: int = 2,
    ) -> None:
        super().__init__()
        self.enable_routing = bool(enable_routing)
        self.state_encoder = CARE3DStateEncoder(
            object_dim=object_dim,
            num_cameras=num_cameras,
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            decision_dim=decision_dim,
            use_temporal=use_temporal,
            dropout=dropout,
        )
        self.vulnerability_head = CounterfactualVulnerabilityHead(
            state_dim=state_dim,
            num_protocols=num_protocols,
            hidden_dim=max(state_dim, 64),
            dropout=dropout,
        )
        self.router = None
        if self.enable_routing:
            self.router = SparseEvidenceRouter(
                object_dim=object_dim,
                source_dim=int(source_dim or object_dim),
                num_protocols=num_protocols,
                topk_sources=topk_sources,
                hidden_dim=hidden_dim,
            )

    def forward(
        self,
        object_features: Tensor,
        camera_support: Tensor,
        camera_quality: Tensor,
        temporal_features: Optional[Tensor] = None,
        decision_features: Optional[Tensor] = None,
        source_features: Optional[Tensor] = None,
        source_reliability: Optional[Tensor] = None,
        source_valid: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        state = self.state_encoder(
            object_features=object_features,
            camera_support=camera_support,
            camera_quality=camera_quality,
            temporal_features=temporal_features,
            decision_features=decision_features,
        )
        output = {"care3d_state": state}
        output.update(self.vulnerability_head(state))

        if self.enable_routing:
            if source_features is None or source_reliability is None:
                raise ValueError(
                    "source_features and source_reliability are required when routing is enabled"
                )
            output.update(
                self.router(
                    object_features=object_features,
                    source_features=source_features,
                    source_reliability=source_reliability,
                    vulnerability=output["vulnerability"],
                    source_valid=source_valid,
                )
            )
        else:
            output["enhanced_features"] = object_features
        return output


class CounterfactualVulnerabilityLoss(nn.Module):
    """P0 loss for evidence-drop regression and decision-boundary prediction."""

    def __init__(
        self,
        regression_weight: float = 1.0,
        crossing_weight: float = 1.0,
        beta: float = 0.1,
    ) -> None:
        super().__init__()
        self.regression_weight = float(regression_weight)
        self.crossing_weight = float(crossing_weight)
        self.beta = float(beta)

    def forward(
        self,
        prediction: Dict[str, Tensor],
        evidence_drop_target: Tensor,
        boundary_crossing_target: Tensor,
        valid_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        evidence_drop = prediction["vulnerability"]
        crossing_logits = prediction["boundary_crossing_logits"]
        if evidence_drop.shape != evidence_drop_target.shape:
            raise ValueError("evidence_drop_target shape mismatch")
        if crossing_logits.shape != boundary_crossing_target.shape:
            raise ValueError("boundary_crossing_target shape mismatch")

        if valid_mask is None:
            valid = torch.ones_like(evidence_drop, dtype=torch.bool)
        elif valid_mask.ndim == evidence_drop.ndim - 1:
            valid = valid_mask.unsqueeze(-1).expand_as(evidence_drop).bool()
        elif valid_mask.shape == evidence_drop.shape:
            valid = valid_mask.bool()
        else:
            raise ValueError("valid_mask must have shape [B,Q] or [B,Q,P]")
        weight = valid.to(evidence_drop.dtype)
        denominator = weight.sum().clamp_min(1.0)

        regression = F.smooth_l1_loss(
            evidence_drop,
            evidence_drop_target.to(evidence_drop.dtype),
            reduction="none",
            beta=self.beta,
        )
        regression = (regression * weight).sum() / denominator
        crossing = F.binary_cross_entropy_with_logits(
            crossing_logits,
            boundary_crossing_target.to(crossing_logits.dtype),
            reduction="none",
        )
        crossing = (crossing * weight).sum() / denominator
        total = self.regression_weight * regression + self.crossing_weight * crossing
        return {
            "loss_care3d_vulnerability": regression,
            "loss_care3d_crossing": crossing,
            "loss_care3d": total,
        }
