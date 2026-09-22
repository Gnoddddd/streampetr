"""Pure selection and numerical summaries for the Stage 3-B real smoke."""

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from datasets.temporal_occ_nuscenes import NUSCENES_CAMERA_ORDER


def select_manifest_pairs(
    records: Sequence[Mapping[str, object]],
    conditions: Optional[Sequence[str]],
    max_pairs: int,
) -> List[Mapping[str, object]]:
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    allowed = set(conditions) if conditions else None
    selected = []
    for record in records:
        if allowed is not None and str(record.get("raw_condition")) not in allowed:
            continue
        if list(record.get("camera_names", ())) != list(NUSCENES_CAMERA_ORDER):
            raise ValueError("manifest camera order does not match canonical order")
        selected.append(record)
        if len(selected) == max_pairs:
            break
    if not selected:
        raise ValueError("no manifest pairs match the requested conditions")
    return selected


def tensor_nonfinite_counts(tensors: Mapping[str, Tensor]) -> Tuple[int, int]:
    nan_count = sum(int(torch.isnan(value).sum().item()) for value in tensors.values())
    inf_count = sum(int(torch.isinf(value).sum().item()) for value in tensors.values())
    return nan_count, inf_count


def masked_statistics(values: Tensor, valid_mask: Tensor) -> Dict[str, float]:
    if values.shape == valid_mask.shape + (1,):
        values = values.squeeze(-1)
    if values.shape != valid_mask.shape:
        raise ValueError("values and valid mask shapes do not match")
    selected = values.masked_select(valid_mask.bool())
    if selected.numel() == 0:
        return {name: 0.0 for name in ("min", "mean", "median", "max")}
    return {
        "min": float(selected.min().item()),
        "mean": float(selected.mean().item()),
        "median": float(selected.median().item()),
        "max": float(selected.max().item()),
    }


def probability_sum_error(probabilities: Tensor, query_valid: Tensor) -> float:
    if probabilities.ndim != 6 or probabilities.shape[:3] != query_valid.shape:
        raise ValueError("probability/query validity shapes do not match")
    sums = probabilities.reshape(*probabilities.shape[:3], -1).sum(dim=-1)
    target = query_valid.to(sums.dtype)
    return float((sums - target).abs().max().item())


def probability_entropy_mean(probabilities: Tensor, query_valid: Tensor) -> float:
    eps = torch.finfo(probabilities.dtype).eps
    entropy = -torch.where(
        probabilities > 0,
        probabilities * probabilities.clamp_min(eps).log(),
        torch.zeros_like(probabilities),
    ).reshape(*probabilities.shape[:3], -1).sum(dim=-1)
    selected = entropy.masked_select(query_valid.bool())
    return float(selected.mean().item()) if selected.numel() else 0.0


def descriptor_l2_mean(descriptor: Tensor, valid_mask: Tensor) -> float:
    if descriptor.shape[:-1] != valid_mask.shape:
        raise ValueError("descriptor/query validity shapes do not match")
    norms = torch.linalg.norm(descriptor, dim=-1)
    selected = norms.masked_select(valid_mask.bool())
    return float(selected.mean().item()) if selected.numel() else 0.0


__all__ = [
    "descriptor_l2_mean",
    "masked_statistics",
    "probability_entropy_mean",
    "probability_sum_error",
    "select_manifest_pairs",
    "tensor_nonfinite_counts",
]
