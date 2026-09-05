"""Pure helpers for the Stage-4 supervision identity audit."""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np


IDENTITIES = ("same-GT positive", "other-GT matched", "unmatched-background")


def assignment_identity(assigned_gt_index: int, focal_gt_index: Optional[int]) -> str:
    """Classify a zero-based assigned GT index; -1 denotes background."""
    assigned_gt_index = int(assigned_gt_index)
    if assigned_gt_index < 0:
        return "unmatched-background"
    if focal_gt_index is not None and assigned_gt_index == int(focal_gt_index):
        return "same-GT positive"
    return "other-GT matched"


def trajectory_statistics(identities: Iterable[str]) -> dict:
    values = tuple(identities)
    if not values:
        raise ValueError("identity trajectory must contain at least one layer")
    unknown = set(values) - set(IDENTITIES)
    if unknown:
        raise ValueError(f"unknown identities: {sorted(unknown)}")
    same = np.asarray([value == "same-GT positive" for value in values], dtype=bool)
    same_indexes = np.flatnonzero(same)
    return {
        "same_gt_layer_count": int(np.count_nonzero(same)),
        "other_gt_layer_count": int(sum(value == "other-GT matched" for value in values)),
        "background_layer_count": int(sum(value == "unmatched-background" for value in values)),
        "same_gt_layer_fraction": float(np.mean(same)),
        "ever_same_gt": bool(np.any(same)),
        "always_same_gt": bool(np.all(same)),
        "never_same_gt": bool(not np.any(same)),
        "first_same_gt_layer": int(same_indexes[0]) if same_indexes.size else -1,
        "last_same_gt_layer": int(same_indexes[-1]) if same_indexes.size else -1,
        "identity_switch_count": int(sum(a != b for a, b in zip(values, values[1:]))),
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple:
    successes, total = int(successes), int(total)
    if total <= 0:
        return float("nan"), float("nan")
    rate = successes / total
    scale = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / scale
    half = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / scale
    return float(center - half), float(center + half)


def bootstrap_rate_difference(
    left: Iterable[bool],
    right: Iterable[bool],
    seed: int,
    iterations: int = 5000,
) -> dict:
    left = np.asarray(tuple(left), dtype=np.float64)
    right = np.asarray(tuple(right), dtype=np.float64)
    if not left.size or not right.size:
        return {
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "iterations": int(iterations),
        }
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(iterations), dtype=np.float64)
    for index in range(int(iterations)):
        a = left[rng.integers(0, left.size, left.size)]
        b = right[rng.integers(0, right.size, right.size)]
        estimates[index] = np.mean(a) - np.mean(b)
    return {
        "estimate": float(np.mean(left) - np.mean(right)),
        "ci_low": float(np.percentile(estimates, 2.5)),
        "ci_high": float(np.percentile(estimates, 97.5)),
        "iterations": int(iterations),
    }

