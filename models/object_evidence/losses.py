"""Object Evidence Preservation and privileged geometry losses."""

import torch
import torch.nn.functional as F
from torch import Tensor


def _weighted_cosine_loss(source: Tensor, target: Tensor, weight: Tensor, eps: float) -> Tensor:
    if source.ndim != 2 or target.shape != source.shape or weight.shape != (len(source),):
        raise ValueError("expected source/target [N,C] and weight [N]")
    denominator = weight.sum()
    zero = source.sum() * 0
    if not bool((denominator > 0).detach()):
        return zero
    distance = 1 - F.cosine_similarity(source, target.detach(), dim=-1, eps=eps)
    return (weight.to(distance) * distance).sum() / (denominator.to(distance) + eps)


def object_evidence_loss(
    fault_tokens: Tensor,
    clean_tokens: Tensor,
    observability_gap: Tensor,
    valid_mask: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Gap-weighted cosine preservation; gradients flow only into fault tokens."""
    if fault_tokens.shape != clean_tokens.shape or fault_tokens.ndim != 2 \
            or fault_tokens.shape[1] != 256:
        raise ValueError("clean and fault tokens must both have shape [N,256]")
    if observability_gap.shape != (len(fault_tokens),) or valid_mask.shape != observability_gap.shape:
        raise ValueError("observability_gap and valid_mask must have shape [N]")
    clean = F.normalize(F.layer_norm(clean_tokens.detach(), (256,)), dim=-1, eps=eps)
    fault = F.normalize(F.layer_norm(fault_tokens, (256,)), dim=-1, eps=eps)
    weight = observability_gap.to(fault) * valid_mask.to(fault.dtype)
    return _weighted_cosine_loss(fault, clean, weight, eps)


def privileged_geometry_loss(
    projected_fault_tokens: Tensor,
    lidar_teacher_tokens: Tensor,
    observability_gap: Tensor,
    valid_mask: Tensor,
    num_lidar_pts: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Gap and LiDAR-reliability weighted cosine guidance."""
    if projected_fault_tokens.shape != lidar_teacher_tokens.shape:
        raise ValueError("projected student and LiDAR teacher tokens must have the same shape")
    if num_lidar_pts.shape != observability_gap.shape:
        raise ValueError("num_lidar_pts must have shape [N]")
    reliability = (num_lidar_pts >= 5).to(projected_fault_tokens.dtype)
    reliability = reliability * (num_lidar_pts.to(projected_fault_tokens) / 20).clamp(max=1)
    weight = observability_gap.to(projected_fault_tokens) * reliability * valid_mask.to(
        projected_fault_tokens.dtype
    )
    student = F.normalize(projected_fault_tokens, dim=-1, eps=eps)
    teacher = F.normalize(lidar_teacher_tokens.detach().to(student), dim=-1, eps=eps)
    return _weighted_cosine_loss(student, teacher, weight, eps)
