"""Pure helpers for CARE-3D counterfactual P0.

The functions in this module deliberately avoid detector-specific imports so the
counterfactual supervision rules can be unit-tested without building StreamPETR.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


PROTOCOLS = ("blur_back", "crash_back", "dark_back")
PREDICTOR_INPUT_KEYS = {
    "object_features",
    "temporal_features",
    "decision_features",
    "camera_support",
    "camera_quality",
}
FORBIDDEN_PREDICTOR_TOKENS = ("fault", "future", "target_frame", "t_plus_1", "outcome")


def clone_counterfactual_states(state: Mapping[str, Tensor | None], branches: int) -> list[dict]:
    """Clone one clean post-state into independent counterfactual branches."""
    if branches <= 0:
        raise ValueError("branches must be positive")
    output = []
    for _ in range(branches):
        cloned = {}
        for key, value in state.items():
            cloned[key] = None if value is None else value.detach().clone()
        output.append(cloned)
    return output


def states_exact(left: Mapping[str, Tensor | None], right: Mapping[str, Tensor | None]) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        a, b = left[key], right[key]
        if a is None or b is None:
            if a is not b:
                return False
        elif not torch.equal(a, b):
            return False
    return True


def flattened_rank(logits: np.ndarray | Tensor, query: int, label: int) -> int:
    """Return the 1-indexed rank using PyTorch's deployment ordering."""
    scores = torch.as_tensor(logits).float().sigmoid().reshape(-1)
    class_count = int(torch.as_tensor(logits).shape[-1])
    flat_index = int(query) * class_count + int(label)
    if flat_index < 0 or flat_index >= scores.numel():
        raise IndexError("query/label pair is outside logits")
    _, order = torch.topk(scores, scores.numel())
    locations = torch.nonzero(order == flat_index, as_tuple=False)
    if locations.numel() != 1:
        raise RuntimeError("target flattened index is not unique")
    return int(locations[0, 0].item()) + 1


def same_query_counterfactual(
    clean_logits: np.ndarray | Tensor,
    fault_logits: np.ndarray | Tensor,
    query: int,
    label: int,
    k: int = 100,
) -> Dict[str, float | int | bool]:
    """Build evidence-drop and Top-K crossing labels for one fixed query/class.

    The query and class are selected only from the clean counterfactual.  The
    fault branch is never allowed to substitute another query.
    """
    clean = torch.as_tensor(clean_logits).float()
    fault = torch.as_tensor(fault_logits).float()
    if clean.shape != fault.shape:
        raise ValueError("clean and fault logits must have identical shape")
    if clean.ndim != 2:
        raise ValueError("logits must have shape [Q,C]")
    clean_score = float(clean[int(query), int(label)].sigmoid().item())
    fault_score = float(fault[int(query), int(label)].sigmoid().item())
    clean_rank = flattened_rank(clean, query, label)
    fault_rank = flattened_rank(fault, query, label)
    clean_topk = clean_rank <= int(k)
    fault_topk = fault_rank <= int(k)
    return {
        "clean_score": clean_score,
        "fault_score": fault_score,
        "evidence_drop": max(clean_score - fault_score, 0.0),
        "clean_flat_rank": clean_rank,
        "fault_flat_rank": fault_rank,
        "clean_topk": bool(clean_topk),
        "fault_topk": bool(fault_topk),
        "cross_topk": bool(clean_topk and not fault_topk),
    }


def assert_prospective_payload(payload: Mapping[str, object]) -> None:
    """Reject predictor inputs that contain outcome/fault/future information."""
    keys = set(payload)
    missing = PREDICTOR_INPUT_KEYS - keys
    if missing:
        raise RuntimeError(f"missing predictor input keys: {sorted(missing)}")
    for key in keys:
        lowered = key.lower()
        if any(token in lowered for token in FORBIDDEN_PREDICTOR_TOKENS):
            raise RuntimeError(f"forbidden predictor input field: {key}")


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise RuntimeError("detector freezing failed")
    return module


def assert_disjoint_splits(rows: Iterable[Mapping[str, object]]) -> None:
    ownership: Dict[str, str] = {}
    for row in rows:
        scene = str(row["scene_token"])
        split = str(row["split"])
        previous = ownership.setdefault(scene, split)
        if previous != split:
            raise RuntimeError(f"scene occurs in multiple splits: {scene}")


def assert_unique_sample_ids(sample_ids: Sequence[str]) -> None:
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("duplicate CARE-3D sample_id detected")


def masked_protocol_count(valid_mask: np.ndarray) -> int:
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[1] != len(PROTOCOLS):
        raise ValueError("valid_mask must have shape [N,3]")
    return int(mask.sum())
