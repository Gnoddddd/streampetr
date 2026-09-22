"""Safe soft-distribution Clean-to-Dirty correspondence distillation."""

from typing import Dict, Tuple

import torch
from torch import Tensor


def masked_softmax(
    logits: Tensor, valid_mask: Tensor, temperature: float
) -> Tuple[Tensor, Tensor]:
    """Softmax over ``J,T,Vh`` with safe all-invalid query handling."""
    if logits.shape != valid_mask.shape:
        raise ValueError("logits and valid_mask shapes must match")
    if logits.ndim != 6:
        raise ValueError("correspondence logits must have shape [B,N,Vq,J,T,Vh]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    prefix = logits.shape[:3]
    flat_logits = (logits / float(temperature)).reshape(*prefix, -1)
    flat_mask = valid_mask.reshape(*prefix, -1).bool()
    query_valid = flat_mask.any(dim=-1)
    masked = flat_logits.masked_fill(~flat_mask, torch.finfo(logits.dtype).min)
    safe = torch.where(query_valid[..., None], masked, torch.zeros_like(masked))
    probabilities = torch.softmax(safe, dim=-1) * flat_mask.to(logits.dtype)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(logits.dtype).eps
    )
    return probabilities.reshape_as(logits), query_valid


def _valid_mean(values: Tensor, valid: Tensor) -> Tensor:
    if bool(valid.any().item()):
        return values.masked_select(valid).mean()
    return values.sum() * 0.0


def correspondence_distillation(
    teacher_logits: Tensor,
    student_logits: Tensor,
    valid_mask: Tensor,
    temperature: float,
    center_candidate_index: int,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Compute ``KL(stopgrad(P_teacher) || P_student)`` over valid queries."""
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher and student logits shapes must match")
    teacher_probs, teacher_query_valid = masked_softmax(
        teacher_logits.detach(), valid_mask, temperature
    )
    student_probs, student_query_valid = masked_softmax(
        student_logits, valid_mask, temperature
    )
    query_valid = teacher_query_valid & student_query_valid
    eps = torch.finfo(student_probs.dtype).eps
    teacher_log = teacher_probs.clamp_min(eps).log()
    student_log = student_probs.clamp_min(eps).log()
    pointwise_kl = torch.where(
        teacher_probs > 0,
        teacher_probs * (teacher_log - student_log),
        torch.zeros_like(teacher_probs),
    )
    query_kl = pointwise_kl.flatten(start_dim=3).sum(dim=-1)
    loss = (
        query_kl.masked_select(query_valid).mean()
        if bool(query_valid.any().item())
        else student_logits.sum() * 0.0
    )
    teacher_entropy_query = -torch.where(
        teacher_probs > 0,
        teacher_probs * teacher_log,
        torch.zeros_like(teacher_probs),
    ).flatten(start_dim=3).sum(dim=-1)
    student_entropy_query = -torch.where(
        student_probs > 0,
        student_probs * student_log,
        torch.zeros_like(student_probs),
    ).flatten(start_dim=3).sum(dim=-1)
    key_count = valid_mask.flatten(start_dim=3).sum(dim=-1).to(student_probs.dtype)
    normalizer = key_count.clamp_min(1).log()
    teacher_normalized = torch.where(
        key_count > 1,
        teacher_entropy_query / normalizer.clamp_min(eps),
        torch.zeros_like(normalizer),
    )
    student_normalized = torch.where(
        key_count > 1,
        student_entropy_query / normalizer.clamp_min(eps),
        torch.zeros_like(normalizer),
    )
    teacher_flat = teacher_probs.flatten(start_dim=3)
    student_flat = student_probs.flatten(start_dim=3)
    teacher_top_prob, teacher_top_index = teacher_flat.max(dim=-1)
    student_top_prob, student_top_index = student_flat.max(dim=-1)
    agreement = (teacher_top_index == student_top_index).to(student_probs.dtype)
    teacher_center = teacher_probs[:, :, :, center_candidate_index].sum(dim=(-1, -2))
    student_center = student_probs[:, :, :, center_candidate_index].sum(dim=(-1, -2))
    diagnostics = {
        "num_valid_queries": query_valid.sum().detach(),
        "teacher_entropy": _valid_mean(teacher_entropy_query, query_valid).detach(),
        "student_entropy": _valid_mean(student_entropy_query, query_valid).detach(),
        "teacher_normalized_entropy": _valid_mean(teacher_normalized, query_valid).detach(),
        "student_normalized_entropy": _valid_mean(student_normalized, query_valid).detach(),
        "teacher_student_kl": _valid_mean(query_kl, query_valid).detach(),
        "teacher_top1_probability": _valid_mean(teacher_top_prob, query_valid).detach(),
        "student_top1_probability": _valid_mean(student_top_prob, query_valid).detach(),
        "top1_agreement": _valid_mean(agreement, query_valid).detach(),
        "teacher_center_candidate_mass": _valid_mean(teacher_center, query_valid).detach(),
        "student_center_candidate_mass": _valid_mean(student_center, query_valid).detach(),
        "teacher_probs": teacher_probs.detach(),
        "student_probs": student_probs,
        "query_valid": query_valid,
    }
    return loss, diagnostics


__all__ = ["correspondence_distillation", "masked_softmax"]
