"""Evidence-conserving Beta temporal update."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn


class EvidenceConservingTemporalUpdate(nn.Module):
    """Update Beta existence evidence without counting history as new evidence.

    ``alpha`` supports existence and ``beta`` supports absence. Only the product
    of observability, novelty, and effective independent evidence can add new
    evidence. With a zero gate, evidence strength is exactly multiplied by
    ``gamma``.
    """

    def __init__(
        self,
        gamma: float = 0.90,
        evidence_scale: float = 2.0,
        max_effective_count: float = 6.0,
        enable_conservation: bool = False,
        reliable_observation_threshold: float = 0.05,
        conservation_tolerance: float = 1e-5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not 0.0 < gamma < 1.0:
            raise ValueError("gamma must be in (0, 1)")
        if not 0.0 <= reliable_observation_threshold <= 1.0:
            raise ValueError("reliable_observation_threshold must be in [0, 1]")
        if conservation_tolerance < 0.0:
            raise ValueError("conservation_tolerance must be non-negative")
        self.gamma = float(gamma)
        self.evidence_scale = float(evidence_scale)
        self.max_effective_count = float(max_effective_count)
        self.enable_conservation = bool(enable_conservation)
        self.reliable_observation_threshold = float(
            reliable_observation_threshold
        )
        self.conservation_tolerance = float(conservation_tolerance)
        self.eps = float(eps)

    def forward(
        self,
        prior_alpha: Tensor,
        prior_beta: Tensor,
        positive_probability: Tensor,
        negative_probability: Tensor,
        observability: Tensor,
        novelty: Tensor,
        effective_count: Optional[Tensor] = None,
        positive_evidence_factor: Optional[Tensor] = None,
        negative_evidence_factor: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        output_dtype = prior_alpha.dtype
        half_compute = output_dtype == torch.float16
        if half_compute:
            # PyTorch 1.9 has no CPU Half implementation for clamp_min and
            # fp16 residual arithmetic is too coarse for a useful conservation
            # invariant. Compute in fp32, then restore the public dtype.
            prior_alpha = prior_alpha.float()
            prior_beta = prior_beta.float()
            positive_probability = positive_probability.float()
            negative_probability = negative_probability.float()
            observability = observability.float()
            novelty = novelty.float()
            if effective_count is not None:
                effective_count = effective_count.float()
            if positive_evidence_factor is not None:
                positive_evidence_factor = positive_evidence_factor.float()
            if negative_evidence_factor is not None:
                negative_evidence_factor = negative_evidence_factor.float()

        prior_alpha = prior_alpha.clamp_min(1.0)
        prior_beta = prior_beta.clamp_min(1.0)
        positive_probability = positive_probability.clamp(0.0, 1.0)
        negative_probability = negative_probability.clamp(0.0, 1.0)
        observability = observability.clamp(0.0, 1.0)
        novelty = novelty.clamp(0.0, 1.0)
        if effective_count is None:
            effective_count = torch.ones_like(observability)
        effective_count = effective_count.clamp(0.0, self.max_effective_count)

        raw_gate = observability * novelty * effective_count
        reliable_observation = torch.ones_like(observability, dtype=torch.bool)
        if self.enable_conservation:
            reliable_observation = (
                observability >= self.reliable_observation_threshold
            ) & (
                (positive_probability + negative_probability) > self.eps
            )
            gate = torch.where(
                reliable_observation,
                raw_gate,
                torch.zeros_like(raw_gate),
            )
        else:
            gate = raw_gate
        if positive_evidence_factor is None:
            positive_gate = gate
        else:
            positive_gate = (
                positive_evidence_factor.clamp(0.0, 1.0) * effective_count
            )
            if self.enable_conservation:
                positive_gate = torch.where(
                    reliable_observation,
                    positive_gate,
                    torch.zeros_like(positive_gate),
                )
        if negative_evidence_factor is None:
            negative_gate = gate
        else:
            negative_gate = (
                negative_evidence_factor.clamp(0.0, 1.0) * effective_count
            )
            if self.enable_conservation:
                negative_gate = torch.where(
                    reliable_observation,
                    negative_gate,
                    torch.zeros_like(negative_gate),
                )
        positive_evidence = (
            self.evidence_scale * positive_gate * positive_probability
        )
        negative_evidence = (
            self.evidence_scale * negative_gate * negative_probability
        )

        alpha = 1.0 + self.gamma * (prior_alpha - 1.0) + positive_evidence
        beta = 1.0 + self.gamma * (prior_beta - 1.0) + negative_evidence
        total = alpha + beta
        existence_probability = alpha / total.clamp_min(self.eps)
        uncertainty = 2.0 / total.clamp_min(self.eps)
        strength = total - 2.0
        prior_strength = prior_alpha + prior_beta - 2.0
        no_new_evidence_strength = self.gamma * prior_strength
        expected_strength = (
            no_new_evidence_strength
            + positive_evidence
            + negative_evidence
        )
        conservation_residual = strength - expected_strength
        inflation_ratio = strength / prior_strength.clamp_min(self.eps)
        inflation_ratio = torch.where(
            prior_strength > self.eps,
            inflation_ratio,
            torch.ones_like(inflation_ratio),
        )
        no_new_evidence = (
            positive_evidence + negative_evidence
        ) <= self.eps
        conservation_ratio = torch.where(
            no_new_evidence & (prior_strength > self.eps),
            inflation_ratio,
            torch.ones_like(inflation_ratio),
        )
        conservation_violation_mask = (
            conservation_residual.abs() > self.conservation_tolerance
        )
        conservation_violation = torch.where(
            conservation_violation_mask,
            conservation_residual.abs(),
            torch.zeros_like(conservation_residual),
        )
        unsupported_growth = (
            (~reliable_observation)
            & (
                strength
                > no_new_evidence_strength + self.conservation_tolerance
            )
        )

        output = {
            "alpha": alpha,
            "beta": beta,
            "existence_probability": existence_probability,
            "uncertainty": uncertainty,
            "strength": strength,
            "prior_strength": prior_strength,
            "no_new_evidence_strength": no_new_evidence_strength,
            "positive_evidence": positive_evidence,
            "negative_evidence": negative_evidence,
            "actual_added_positive_evidence": positive_evidence,
            "actual_added_negative_evidence": negative_evidence,
            "raw_evidence_gate": raw_gate,
            "evidence_gate": gate,
            "positive_evidence_gate": positive_gate,
            "negative_evidence_gate": negative_gate,
            "reliable_observation": reliable_observation,
            "inflation_ratio": inflation_ratio,
            "no_new_evidence": no_new_evidence,
            "conservation_ratio": conservation_ratio,
            "conservation_violation": conservation_violation,
            "conservation_violation_mask": conservation_violation_mask,
            "conservation_residual": conservation_residual,
            "unsupported_growth": unsupported_growth,
        }
        if half_compute:
            output = {
                key: (
                    value.to(output_dtype)
                    if value.is_floating_point()
                    else value
                )
                for key, value in output.items()
            }
        return output
