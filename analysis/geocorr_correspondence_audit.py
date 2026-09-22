"""Read-only descriptive metrics for real-clean GeoCorr correspondence."""

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor

from models.geocorr_recovery import find_center_candidate_index


def _entropy(probabilities: Tensor, dimensions: Tuple[int, ...]) -> Tensor:
    eps = torch.finfo(probabilities.dtype).eps
    terms = torch.where(
        probabilities > 0,
        probabilities * probabilities.clamp_min(eps).log(),
        torch.zeros_like(probabilities),
    )
    return -terms.sum(dim=dimensions)


def _normalized_entropy(entropy: Tensor, valid_count: Tensor) -> Tensor:
    denominator = valid_count.to(entropy.dtype).clamp_min(1).log()
    return torch.where(
        valid_count > 1,
        entropy / denominator.clamp_min(torch.finfo(entropy.dtype).eps),
        torch.zeros_like(entropy),
    )


def _top1_margin(probabilities: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    top2 = probabilities.topk(k=2, dim=-1).values
    top_probability, top_index = probabilities.max(dim=-1)
    return top_probability, top_index, top2[..., 0] - top2[..., 1]


def correspondence_audit_metrics(
    probabilities: Tensor,
    valid_mask: Tensor,
    candidate_offsets: Sequence[Sequence[float]],
) -> Dict[str, Tensor]:
    """Calculate full, position, view and same-view metrics.

    Args:
        probabilities: ``[B,N,Vq,J,T,Vh]`` masked probabilities.
        valid_mask: Same shape; invalid keys must not enter any statistic.
        candidate_offsets: ``[J,2]`` offsets, including one unique ``(0,0)``.
    """
    if probabilities.shape != valid_mask.shape or probabilities.ndim != 6:
        raise ValueError("probabilities and mask must share [B,N,Vq,J,T,Vh]")
    batch, objects, current_views, candidates, _, history_views = probabilities.shape
    if current_views != history_views:
        raise ValueError("same-view diagnostics require matching camera orders")
    offsets = torch.as_tensor(
        candidate_offsets, device=probabilities.device, dtype=probabilities.dtype
    )
    if offsets.shape != (candidates, 2):
        raise ValueError("candidate_offsets must have shape [J,2]")
    center = find_center_candidate_index(candidate_offsets)
    query_valid = valid_mask.flatten(start_dim=3).any(-1)

    full_entropy = _entropy(probabilities, (3, 4, 5))
    full_count = valid_mask.flatten(start_dim=3).sum(-1)
    full_flat = probabilities.flatten(start_dim=3)
    full_top1 = full_flat.max(-1).values

    position = probabilities.sum(dim=(4, 5))
    position_valid = valid_mask.flatten(start_dim=4).any(-1)
    position_entropy = _entropy(position, (3,))
    position_top1, position_index, position_margin = _top1_margin(position)
    position_offset = offsets[position_index]

    view = probabilities.sum(dim=(3, 4))
    view_valid = valid_mask.any(dim=4).any(dim=3)
    view_entropy = _entropy(view, (3,))
    view_top1, view_index = view.max(-1)

    same = probabilities.diagonal(offset=0, dim1=2, dim2=5).permute(0, 1, 4, 2, 3)
    same_mask = valid_mask.diagonal(offset=0, dim1=2, dim2=5).permute(0, 1, 4, 2, 3)
    same_total = same.sum(dim=(3, 4), keepdim=True)
    same = torch.where(
        same_total > 0,
        same / same_total.clamp_min(torch.finfo(probabilities.dtype).eps),
        torch.zeros_like(same),
    )
    same_position = same.sum(dim=4)
    same_position_valid = same_mask.any(dim=4)
    same_query_valid = same_mask.flatten(start_dim=3).any(-1)
    same_entropy = _entropy(same_position, (3,))
    same_top1, same_index, same_margin = _top1_margin(same_position)

    return {
        "query_valid": query_valid,
        "valid_historical_key_count": full_count,
        "full_entropy": full_entropy,
        "normalized_full_entropy": _normalized_entropy(full_entropy, full_count),
        "full_top1_probability": full_top1,
        "position_marginal": position,
        "position_entropy": position_entropy,
        "normalized_position_entropy": _normalized_entropy(
            position_entropy, position_valid.sum(-1)
        ),
        "position_top1_probability": position_top1,
        "position_top1_index": position_index,
        "position_top1_offset": position_offset,
        "position_top1_minus_top2_margin": position_margin,
        "center_candidate_mass": position[..., center],
        "view_marginal": view,
        "view_entropy": view_entropy,
        "normalized_view_entropy": _normalized_entropy(view_entropy, view_valid.sum(-1)),
        "view_top1_probability": view_top1,
        "view_top1_index": view_index,
        "valid_historical_view_count": view_valid.sum(dim=-1),
        "same_view_query_valid": same_query_valid,
        "same_view_position_entropy": same_entropy,
        "same_view_normalized_position_entropy": _normalized_entropy(
            same_entropy, same_position_valid.sum(-1)
        ),
        "same_view_position_top1_probability": same_top1,
        "same_view_position_top1_index": same_index,
        "same_view_center_mass": same_position[..., center],
        "same_view_top1_margin": same_margin,
    }


def shuffle_history_objects(
    history_tokens: Tensor,
    history_valid_mask: Tensor,
    object_valid_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Roll valid historical objects by one for a deterministic negative control.

    Returns shuffled tokens, shuffled validity, source indices ``[B,N]`` and a
    comparable-object mask. Batches with fewer than two valid objects are not
    marked comparable.
    """
    if history_tokens.ndim != 6 or history_valid_mask.shape != history_tokens.shape[:-1]:
        raise ValueError("history tokens/mask must have [B,N,J,T,V,(C)]")
    batch, objects = history_tokens.shape[:2]
    if object_valid_mask is None:
        object_valid_mask = torch.ones(
            batch, objects, dtype=torch.bool, device=history_tokens.device
        )
    if object_valid_mask.shape != (batch, objects):
        raise ValueError("object_valid_mask must have shape [B,N]")
    permutation = torch.arange(objects, device=history_tokens.device).repeat(batch, 1)
    comparable = torch.zeros_like(object_valid_mask)
    for batch_index in range(batch):
        indices = object_valid_mask[batch_index].nonzero(as_tuple=False).flatten()
        if indices.numel() >= 2:
            permutation[batch_index, indices] = indices.roll(1)
            comparable[batch_index, indices] = True
    batch_indices = torch.arange(batch, device=history_tokens.device)[:, None]
    return (
        history_tokens[batch_indices, permutation],
        history_valid_mask[batch_indices, permutation],
        permutation,
        comparable,
    )


def confidence_group_indices(scores: Tensor) -> Dict[str, object]:
    """Return stable score-ranked Top-K and exhaustive confidence quartiles."""
    if scores.ndim != 1:
        raise ValueError("scores must have shape [N]")
    count = scores.numel()
    ranked_list = sorted(
        range(count), key=lambda index: (-float(scores[index].detach().cpu()), index)
    )
    ranked = torch.tensor(ranked_list, dtype=torch.long, device=scores.device)
    groups: Dict[str, object] = {
        "all": torch.arange(count, dtype=torch.long, device=scores.device),
        "top25": ranked[: min(25, count)],
        "top50": ranked[: min(50, count)],
        "top100": ranked[: min(100, count)],
    }
    sizes = [count // 4 + (1 if index < count % 4 else 0) for index in range(4)]
    names = ("q1_high", "q2", "q3", "q4_low")
    start = 0
    quartiles = {}
    for name, size in zip(names, sizes):
        quartiles[name] = ranked[start : start + size]
        start += size
    groups["quartiles"] = quartiles
    return groups


def select_object_group(tensor: Tensor, indices: Tensor) -> Tensor:
    """Select an aligned ``[B,N,...]`` object group without changing its order."""
    if tensor.ndim < 2 or indices.ndim != 1:
        raise ValueError("tensor and indices must have [B,N,...] and [K]")
    if indices.numel() and (indices.min() < 0 or indices.max() >= tensor.shape[1]):
        raise IndexError("object group index is out of range")
    return tensor.index_select(1, indices.to(tensor.device))


def _describe(values: Tensor, valid: Tensor) -> Dict[str, float]:
    selected = values.masked_select(valid)
    if selected.numel() == 0:
        return {"mean": 0.0, "median": 0.0}
    return {
        "mean": float(selected.float().mean().item()),
        "median": float(selected.float().median().item()),
    }


def _rate(values: Tensor, valid: Tensor) -> float:
    selected = values.masked_select(valid)
    return float(selected.float().mean().item()) if selected.numel() else 0.0


def _score_summary(scores: Optional[Tensor]) -> Dict[str, float]:
    if scores is None or scores.numel() == 0:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    values = scores.detach().float()
    return {
        "min": float(values.min().item()),
        "median": float(values.median().item()),
        "max": float(values.max().item()),
    }


def _comparison_entry(
    correct: Tensor,
    shuffled: Tensor,
    valid: Tensor,
    entropy_direction: bool,
) -> Dict[str, object]:
    correct_summary = _describe(correct, valid)
    shuffled_summary = _describe(shuffled, valid)
    sign = 1.0 if entropy_direction else -1.0
    return {
        "correct": correct_summary,
        "shuffled": shuffled_summary,
        "delta": {
            name: sign * (shuffled_summary[name] - correct_summary[name])
            for name in ("mean", "median")
        },
    }


def summarize_correspondence_audit(
    metrics: Dict[str, Tensor],
    candidate_offsets: Sequence[Sequence[float]],
    shuffled_metrics: Optional[Dict[str, Tensor]] = None,
    shuffled_comparable: Optional[Tensor] = None,
    num_frame_pairs: int = 1,
    scores: Optional[Tensor] = None,
) -> Dict[str, object]:
    """Convert per-query tensors into a compact JSON-safe summary."""
    valid = metrics["query_valid"]
    same_valid = metrics["same_view_query_valid"]
    summary = {
        "num_frame_pairs": int(num_frame_pairs),
        "num_objects": int(valid.any(dim=2).sum().item()),
        "num_valid_queries": int(valid.sum().item()),
        "num_same_view_queries": int(same_valid.sum().item()),
        "same_view_valid_queries": int(same_valid.sum().item()),
        "score": _score_summary(scores),
    }
    for name in (
        "normalized_full_entropy",
        "normalized_position_entropy",
        "normalized_view_entropy",
        "position_top1_probability",
        "center_candidate_mass",
        "position_top1_minus_top2_margin",
    ):
        summary[name] = _describe(metrics[name], valid)
    for name in (
        "same_view_normalized_position_entropy",
        "same_view_center_mass",
        "same_view_position_top1_probability",
        "same_view_top1_margin",
    ):
        summary[name] = _describe(metrics[name], same_valid)
    center = find_center_candidate_index(candidate_offsets)
    summary["center_as_top1_rate"] = _rate(
        metrics["position_top1_index"] == center, valid
    )
    summary["same_view_center_as_top1_rate"] = _rate(
        metrics["same_view_position_top1_index"] == center, same_valid
    )

    histogram: Dict[str, int] = {}
    offsets = torch.as_tensor(candidate_offsets)
    for index in metrics["position_top1_index"].masked_select(valid).cpu().tolist():
        offset = offsets[index].tolist()
        key = "(%g,%g)" % (offset[0], offset[1])
        histogram[key] = histogram.get(key, 0) + 1
    summary["position_top1_offset_histogram"] = histogram

    if shuffled_metrics is not None and shuffled_comparable is not None:
        comparison = valid & shuffled_metrics["query_valid"] & shuffled_comparable[:, :, None]
        negative = {}
        for name in (
            "normalized_position_entropy",
            "position_top1_probability",
            "center_candidate_mass",
            "position_top1_minus_top2_margin",
        ):
            negative[name] = _comparison_entry(
                metrics[name], shuffled_metrics[name], comparison,
                entropy_direction=name == "normalized_position_entropy",
            )
        correct_center = metrics["position_top1_index"] == center
        shuffled_center = shuffled_metrics["position_top1_index"] == center
        correct_rate = _rate(correct_center, comparison)
        shuffled_rate = _rate(shuffled_center, comparison)
        negative["center_as_top1_rate"] = {
            "correct": correct_rate,
            "shuffled": shuffled_rate,
            "delta": correct_rate - shuffled_rate,
        }
        negative["delta_position_entropy"] = negative[
            "normalized_position_entropy"
        ]["delta"]["mean"]
        negative["delta_position_top1_probability"] = negative[
            "position_top1_probability"
        ]["delta"]["mean"]
        negative["delta_center_mass"] = negative[
            "center_candidate_mass"
        ]["delta"]["mean"]
        negative["delta_top1_margin"] = negative[
            "position_top1_minus_top2_margin"
        ]["delta"]["mean"]
        negative["delta_center_as_top1_rate"] = negative[
            "center_as_top1_rate"
        ]["delta"]
        negative["num_comparable_queries"] = int(comparison.sum().item())
        summary["shuffled_history_control"] = negative
    return summary


def summarize_valid_view_groups(
    metrics: Dict[str, Tensor], candidate_offsets: Sequence[Sequence[float]]
) -> Dict[str, Dict[str, object]]:
    """Summarize queries with one, two, or at least three valid history views."""
    valid = metrics["query_valid"]
    view_count = metrics["valid_historical_view_count"]
    center = find_center_candidate_index(candidate_offsets)
    groups = {
        "1_view": view_count == 1,
        "2_views": view_count == 2,
        "3_or_more_views": view_count >= 3,
    }
    result = {}
    for name, membership in groups.items():
        selected = valid & membership
        result[name] = {
            "query_count": int(selected.sum().item()),
            "num_valid_queries": int(selected.sum().item()),
            "normalized_position_entropy": _describe(
                metrics["normalized_position_entropy"], selected
            ),
            "position_top1_probability": _describe(
                metrics["position_top1_probability"], selected
            ),
            "center_candidate_mass": _describe(
                metrics["center_candidate_mass"], selected
            ),
            "position_top1_minus_top2_margin": _describe(
                metrics["position_top1_minus_top2_margin"], selected
            ),
            "center_as_top1_rate": _rate(
                metrics["position_top1_index"] == center, selected
            ),
        }
    return result


__all__ = [
    "correspondence_audit_metrics",
    "confidence_group_indices",
    "select_object_group",
    "shuffle_history_objects",
    "summarize_correspondence_audit",
    "summarize_valid_view_groups",
]
