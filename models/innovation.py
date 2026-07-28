"""Reliability-calibrated temporal innovation for Evidence3D Stage2 S2.3."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn


def _safe_probability(value: Tensor, eps: float) -> Tensor:
    value = value.clamp_min(0.0)
    return value / value.sum(dim=-1, keepdim=True).clamp_min(eps)


def _entropy_reliability(probability: Tensor, eps: float) -> Tensor:
    probability = _safe_probability(probability, eps)
    classes = max(int(probability.shape[-1]), 2)
    entropy = -(probability * probability.clamp_min(eps).log()).sum(dim=-1)
    return (1.0 - entropy / math.log(classes)).clamp(0.0, 1.0)


def normalized_js_divergence(left: Tensor, right: Tensor, eps: float = 1e-6) -> Tensor:
    """Return Jensen-Shannon divergence normalized to ``[0, 1]``."""
    output_dtype = left.dtype
    if output_dtype == torch.float16:
        left, right = left.float(), right.float()
    left = _safe_probability(left, eps)
    right = _safe_probability(right, eps)
    mean = 0.5 * (left + right)
    left_kl = (left * (left.clamp_min(eps).log() - mean.clamp_min(eps).log())).sum(-1)
    right_kl = (right * (right.clamp_min(eps).log() - mean.clamp_min(eps).log())).sum(-1)
    result = (0.5 * (left_kl + right_kl) / math.log(2.0)).clamp(0.0, 1.0)
    return result.to(output_dtype)


def wrapped_angle_difference(left: Tensor, right: Tensor) -> Tensor:
    """Absolute shortest angular distance in radians."""
    return torch.atan2(torch.sin(left - right), torch.cos(left - right)).abs()


class ReliabilityCalibratedInnovation(nn.Module):
    """Compute interpretable S2.3 innovation diagnostics and evidence gains.

    Geometry uses ``[x, y, z, yaw, vx, vy]``. Feature and reference inputs are
    detached deliberately: S2.3 adds no loss and does not backpropagate through
    hand-crafted diagnostics.
    """

    MODES = ("off", "track", "active")

    def __init__(
        self,
        mode: str = "off",
        source_weight: float = 0.25,
        feature_weight: float = 0.25,
        geometry_weight: float = 0.20,
        semantic_weight: float = 0.15,
        enable_source: bool = True,
        enable_feature: bool = True,
        enable_geometry: bool = True,
        enable_semantic: bool = True,
        enable_reliability: bool = True,
        enable_conflict: bool = True,
        enable_asymmetric_negative: bool = True,
        novelty_floor: float = 0.30,
        tau_reacquisition: float = 3.0,
        conflict_power: float = 1.0,
        center_scale: float = 5.0,
        velocity_scale: float = 5.0,
        geometry_center_weight: float = 0.6,
        geometry_yaw_weight: float = 0.2,
        geometry_velocity_weight: float = 0.2,
        reliable_observation_threshold: float = 0.05,
        negative_observability_threshold: float = 0.20,
        negative_source_quality_threshold: float = 0.20,
        conflict_geometry_threshold: float = 0.75,
        enable_strength_saturation: bool = False,
        strength_temperature: float = 10.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        mode = str(mode).lower()
        if mode not in self.MODES:
            raise ValueError("innovation mode must be off, track, or active")
        self.mode = mode
        self.weights = {
            "source_innovation": float(source_weight),
            "feature_innovation": float(feature_weight),
            "geometry_innovation": float(geometry_weight),
            "semantic_innovation": float(semantic_weight),
        }
        if any(value < 0.0 for value in self.weights.values()):
            raise ValueError("innovation weights must be non-negative")
        self.enabled = {
            "source_innovation": bool(enable_source),
            "feature_innovation": bool(enable_feature),
            "geometry_innovation": bool(enable_geometry),
            "semantic_innovation": bool(enable_semantic),
        }
        self.enable_reliability = bool(enable_reliability)
        self.enable_conflict = bool(enable_conflict)
        self.enable_asymmetric_negative = bool(enable_asymmetric_negative)
        self.novelty_floor = float(novelty_floor)
        self.tau_reacquisition = float(tau_reacquisition)
        self.conflict_power = float(conflict_power)
        self.center_scale = float(center_scale)
        self.velocity_scale = float(velocity_scale)
        self.geometry_weights = (
            float(geometry_center_weight),
            float(geometry_yaw_weight),
            float(geometry_velocity_weight),
        )
        self.reliable_observation_threshold = float(
            reliable_observation_threshold
        )
        self.negative_observability_threshold = float(
            negative_observability_threshold
        )
        self.negative_source_quality_threshold = float(
            negative_source_quality_threshold
        )
        self.conflict_geometry_threshold = float(conflict_geometry_threshold)
        self.enable_strength_saturation = bool(enable_strength_saturation)
        self.strength_temperature = float(strength_temperature)
        self.eps = float(eps)
        if not 0.0 <= self.novelty_floor <= 1.0:
            raise ValueError("novelty_floor must be in [0, 1]")
        if self.tau_reacquisition <= 0.0 or self.strength_temperature <= 0.0:
            raise ValueError("innovation temperatures must be positive")

    def _cosine_innovation(
        self, current: Tensor, previous: Tensor, half_scale: bool
    ) -> Tensor:
        output_dtype = current.dtype
        current, previous = current.detach(), previous.detach()
        if output_dtype == torch.float16:
            current, previous = current.float(), previous.float()
        current_norm = torch.linalg.vector_norm(current, dim=-1)
        previous_norm = torch.linalg.vector_norm(previous, dim=-1)
        cosine = (current * previous).sum(-1) / (
            current_norm * previous_norm
        ).clamp_min(self.eps)
        cosine = cosine.clamp(-1.0, 1.0)
        value = (0.5 * (1.0 - cosine) if half_scale else 1.0 - cosine)
        return value.clamp(0.0, 1.0).to(output_dtype)

    def forward(
        self,
        current_source: Tensor,
        previous_source: Tensor,
        current_feature: Tensor,
        previous_feature: Tensor,
        current_geometry: Tensor,
        previous_geometry: Tensor,
        current_class_probability: Tensor,
        previous_class_probability: Tensor,
        current_ternary_probability: Tensor,
        previous_ternary_probability: Tensor,
        previous_age: Tensor,
        previous_strength: Tensor,
        previous_presence: Tensor,
        has_prior: Tensor,
        valid_feature_pair: Tensor,
        valid_geometry: Tensor,
        observability: Tensor,
        source_quality: Tensor,
        reliable_observation: Tensor,
        camera_coverage: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if current_feature.shape != previous_feature.shape:
            raise ValueError(
                "current and previous feature tensors must be query-aligned"
            )
        if current_geometry.shape != previous_geometry.shape:
            raise ValueError(
                "current and previous geometry tensors must be query-aligned"
            )
        if current_source.shape != previous_source.shape:
            raise ValueError(
                "current and previous source tensors must be query-aligned"
            )
        if current_feature.shape[:2] != observability.shape:
            raise ValueError("innovation inputs must share [B,Q] layout")
        output_dtype = observability.dtype
        compute_half = output_dtype == torch.float16
        if compute_half:
            current_source = current_source.float()
            previous_source = previous_source.float()
            current_feature = current_feature.float()
            previous_feature = previous_feature.float()
            current_geometry = current_geometry.float()
            previous_geometry = previous_geometry.float()
            current_class_probability = current_class_probability.float()
            previous_class_probability = previous_class_probability.float()
            current_ternary_probability = current_ternary_probability.float()
            previous_ternary_probability = previous_ternary_probability.float()
            previous_age = previous_age.float()
            previous_strength = previous_strength.float()
            previous_presence = previous_presence.float()
            observability = observability.float()
            source_quality = source_quality.float()
            if camera_coverage is not None:
                camera_coverage = camera_coverage.float()

        current_source_norm = torch.linalg.vector_norm(
            current_source.float() if compute_half else current_source, dim=-1
        )
        previous_source_norm = torch.linalg.vector_norm(
            previous_source.float() if compute_half else previous_source, dim=-1
        )
        source_innovation = self._cosine_innovation(
            current_source, previous_source, half_scale=False
        )
        has_source = current_source_norm > self.eps
        has_previous_source = previous_source_norm > self.eps
        source_innovation = torch.where(
            ~has_source,
            torch.zeros_like(source_innovation),
            torch.where(
                ~has_previous_source,
                torch.ones_like(source_innovation),
                source_innovation,
            ),
        )
        valid_source = has_source

        feature_innovation = self._cosine_innovation(
            current_feature, previous_feature, half_scale=True
        )
        is_new_query = ~has_prior
        feature_innovation = torch.where(
            is_new_query, torch.ones_like(feature_innovation), feature_innovation
        )
        feature_innovation = torch.where(
            valid_feature_pair | is_new_query,
            feature_innovation,
            torch.zeros_like(feature_innovation),
        )

        geometry_dtype = current_geometry.dtype
        current_geometry = current_geometry.detach()
        previous_geometry = previous_geometry.detach()
        if geometry_dtype == torch.float16:
            current_geometry = current_geometry.float()
            previous_geometry = previous_geometry.float()
        center_residual = torch.linalg.vector_norm(
            current_geometry[..., :3] - previous_geometry[..., :3], dim=-1
        ) / max(self.center_scale, self.eps)
        yaw_residual = wrapped_angle_difference(
            current_geometry[..., 3], previous_geometry[..., 3]
        ) / math.pi
        velocity_residual = torch.linalg.vector_norm(
            current_geometry[..., 4:6] - previous_geometry[..., 4:6], dim=-1
        ) / max(self.velocity_scale, self.eps)
        wc, wy, wv = self.geometry_weights
        geometry_residual = (
            wc * center_residual + wy * yaw_residual + wv * velocity_residual
        )
        geometry_innovation = (1.0 - torch.exp(-geometry_residual)).clamp(0.0, 1.0)
        geometry_innovation = torch.where(
            is_new_query,
            torch.ones_like(geometry_innovation),
            torch.where(
                valid_geometry,
                geometry_innovation,
                torch.zeros_like(geometry_innovation),
            ),
        )

        class_semantic = normalized_js_divergence(
            current_class_probability.detach(),
            previous_class_probability.detach(),
            self.eps,
        )
        ternary_semantic = normalized_js_divergence(
            current_ternary_probability.detach(),
            previous_ternary_probability.detach(),
            self.eps,
        )
        valid_semantic = has_prior
        semantic_innovation = 0.5 * (class_semantic + ternary_semantic)
        semantic_innovation = torch.where(
            is_new_query,
            torch.ones_like(semantic_innovation),
            semantic_innovation,
        )

        age = previous_age.float() if compute_half else previous_age
        temporal_reacquisition = (
            1.0 - torch.exp(-age.clamp_min(0.0) / self.tau_reacquisition)
        ).clamp(0.0, 1.0)
        is_reacquired = has_prior & (previous_age > 0.0) & reliable_observation
        is_continuous = has_prior & (previous_age <= 0.0) & reliable_observation

        class_reliability = _entropy_reliability(
            current_class_probability.float()
            if compute_half
            else current_class_probability,
            self.eps,
        )
        ternary_reliability = _entropy_reliability(
            current_ternary_probability.float()
            if compute_half
            else current_ternary_probability,
            self.eps,
        )
        semantic_reliability = (0.5 * (
            class_reliability + ternary_reliability
        )).clamp(0.0, 1.0)
        geometry_proxy = torch.exp(
            -torch.relu(
                geometry_residual
                - self.conflict_geometry_threshold
            )
        )
        geometric_reliability = torch.where(
            valid_geometry, geometry_proxy, torch.ones_like(geometry_proxy)
        ).clamp(0.0, 1.0)
        observation_reliability = (
            reliable_observation.to(observability.dtype)
            * observability.clamp(0.0, 1.0)
        )
        source_reliability = source_quality.clamp(0.0, 1.0)
        combined_reliability = (
            observation_reliability
            * source_reliability
            * semantic_reliability
            * geometric_reliability
        ).clamp(0.0, 1.0)
        if not self.enable_reliability:
            combined_reliability = reliable_observation.to(observability.dtype)
        evidence_reliability = (
            combined_reliability
            if self.enable_reliability
            else observation_reliability
        )

        semantic_conflict = torch.where(
            has_prior, semantic_innovation, torch.zeros_like(semantic_innovation)
        )
        existence_conflict = torch.where(
            has_prior,
            (
                previous_presence * current_ternary_probability[..., 1]
                + (1.0 - previous_presence)
                * current_ternary_probability[..., 0]
            ).clamp(0.0, 1.0),
            torch.zeros_like(previous_presence),
        )
        geometry_conflict = torch.where(
            valid_geometry,
            (
                (geometry_residual - self.conflict_geometry_threshold)
                / max(1.0 - self.conflict_geometry_threshold, self.eps)
            ).clamp(0.0, 1.0),
            torch.zeros_like(previous_presence),
        )
        conflict = (
            (semantic_conflict + geometry_conflict + existence_conflict) / 3.0
        ) * combined_reliability
        conflict = conflict.clamp(0.0, 1.0)
        if not self.enable_conflict:
            conflict = torch.zeros_like(conflict)

        components = {
            "source_innovation": (source_innovation, valid_source),
            "feature_innovation": (
                feature_innovation,
                valid_feature_pair | is_new_query,
            ),
            "geometry_innovation": (
                geometry_innovation,
                valid_geometry | is_new_query,
            ),
            "semantic_innovation": (
                semantic_innovation,
                valid_semantic | is_new_query,
            ),
        }
        numerator = torch.zeros_like(observability)
        denominator = torch.zeros_like(observability)
        for name, (value, valid) in components.items():
            if self.enabled[name] and self.weights[name] > 0.0:
                weight = observability.new_tensor(self.weights[name])
                numerator = numerator + value * valid.to(observability.dtype) * weight
                denominator = denominator + valid.to(observability.dtype) * weight
        base = torch.where(
            denominator > self.eps,
            numerator / denominator.clamp_min(self.eps),
            torch.zeros_like(numerator),
        ).clamp(0.0, 1.0)
        reacquired = 1.0 - (1.0 - base) * (1.0 - temporal_reacquisition)
        compatible = reacquired * (1.0 - conflict).pow(self.conflict_power)
        reliable = compatible * combined_reliability
        novelty_gain = self.novelty_floor + (1.0 - self.novelty_floor) * reliable
        novelty_gain = torch.where(
            reliable_observation,
            novelty_gain,
            torch.zeros_like(novelty_gain),
        )
        novelty_gain = torch.where(
            is_new_query & reliable_observation,
            torch.ones_like(novelty_gain),
            novelty_gain,
        ).clamp(0.0, 1.0)

        saturation = 1.0 / (
            1.0
            + previous_strength.clamp_min(0.0)
            / self.strength_temperature
        )
        if not self.enable_strength_saturation:
            saturation = torch.ones_like(saturation)
        positive_gain = (novelty_gain * saturation).clamp(0.0, 1.0)

        p_unobserved = current_ternary_probability[..., 2].clamp(0.0, 1.0)
        positive_reliability = (
            evidence_reliability * (1.0 - p_unobserved)
        ).clamp(0.0, 1.0)
        if camera_coverage is None:
            camera_coverage = has_source.to(observability.dtype)
        negative_visibility_gate = (
            (observability >= self.negative_observability_threshold)
            & (source_quality >= self.negative_source_quality_threshold)
            & (camera_coverage > 0.0)
            & (p_unobserved < 0.5)
        ).to(observability.dtype)
        if not self.enable_asymmetric_negative:
            negative_visibility_gate = torch.ones_like(negative_visibility_gate)
        negative_reliability = (
            evidence_reliability
            * negative_visibility_gate
            * (1.0 - p_unobserved)
        ).clamp(0.0, 1.0)
        negative_gain = positive_gain

        output = {
            "source_innovation": source_innovation,
            "feature_innovation": feature_innovation,
            "geometry_innovation": geometry_innovation,
            "class_semantic_innovation": class_semantic,
            "ternary_semantic_innovation": ternary_semantic,
            "semantic_innovation": semantic_innovation,
            "temporal_reacquisition": temporal_reacquisition,
            "is_new_query": is_new_query,
            "is_continuous_observation": is_continuous,
            "is_reacquired_query": is_reacquired,
            "valid_feature_pair": valid_feature_pair,
            "valid_geometry": valid_geometry,
            "center_residual": center_residual,
            "yaw_residual": yaw_residual,
            "velocity_residual": velocity_residual,
            "observation_reliability": observation_reliability,
            "source_reliability": source_reliability,
            "semantic_reliability": semantic_reliability,
            "geometric_reliability": geometric_reliability,
            "combined_reliability": combined_reliability,
            "semantic_conflict": semantic_conflict,
            "geometry_conflict": geometry_conflict,
            "existence_conflict": existence_conflict,
            "conflict": conflict,
            "base_innovation": base,
            "reacquired_innovation": reacquired,
            "compatible_innovation": compatible,
            "reliable_innovation": reliable,
            "novelty_gain": novelty_gain,
            "positive_novelty_gain": positive_gain,
            "negative_novelty_gain": negative_gain,
            "positive_reliability": positive_reliability,
            "negative_reliability": negative_reliability,
            "negative_visibility_gate": negative_visibility_gate,
            "strength_saturation": saturation,
        }
        if compute_half:
            output = {
                key: (
                    value.to(output_dtype)
                    if value.is_floating_point()
                    else value
                )
                for key, value in output.items()
            }
        return output
