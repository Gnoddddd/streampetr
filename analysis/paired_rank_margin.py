"""Pure paired rank-margin and bootstrap helpers."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np


def stable_rank(flat_scores: np.ndarray, index: int) -> int:
    values = np.asarray(flat_scores, dtype=np.float64).reshape(-1)
    target = values[int(index)]
    return int(np.count_nonzero(values > target) +
               np.count_nonzero(values[:int(index)] == target) + 1)


def query_margin_statistics(
    logits: np.ndarray,
    boxes: np.ndarray,
    gt_center: Iterable[float],
    gt_class: int,
    topk: int = 100,
    geometry_threshold: float = 2.0,
) -> dict:
    """Best GT-class score among geometry-qualified queries and its margin."""
    logits = np.asarray(logits, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    center = np.asarray(tuple(gt_center), dtype=np.float64)
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    flat = scores.reshape(-1)
    k = min(max(int(topk), 1), flat.size)
    boundary = float(np.partition(flat, flat.size - k)[flat.size - k])
    distances = np.linalg.norm(boxes[:, :3] - center[:3], axis=1)
    near = np.flatnonzero(np.isfinite(distances) & (distances <= geometry_threshold))
    if not near.size:
        return {
            "candidate_available": False, "best_query": -1,
            "geometry_best_query": -1, "score": float("nan"),
            "rank": -1, "s_k": boundary, "margin": float("nan"),
            "center_distance": float(np.nanmin(distances)), "near_count": 0,
        }
    gt_scores = scores[near, int(gt_class)]
    # Stable query-index tie break matches flattened deployment ordering.
    best = int(near[np.lexsort((near, -gt_scores))[0]])
    geometry_best = int(near[np.argmin(distances[near])])
    flat_index = best * logits.shape[1] + int(gt_class)
    rank = stable_rank(flat, flat_index)
    return {
        "candidate_available": True, "best_query": best,
        "geometry_best_query": geometry_best, "score": float(scores[best, int(gt_class)]),
        "rank": rank, "s_k": boundary,
        "margin": float(scores[best, int(gt_class)] - boundary),
        "center_distance": float(distances[best]), "near_count": int(near.size),
    }


def fixed_query_statistics(
    logits: np.ndarray,
    boxes: np.ndarray,
    query: int,
    gt_center: Iterable[float],
    gt_class: int,
) -> dict:
    logits = np.asarray(logits, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    flat = scores.reshape(-1)
    index = int(query) * logits.shape[1] + int(gt_class)
    distance = float(np.linalg.norm(
        boxes[int(query), :3] - np.asarray(tuple(gt_center), dtype=float)[:3]
    ))
    return {
        "score": float(scores[int(query), int(gt_class)]),
        "rank": stable_rank(flat, index),
        "center_distance": distance,
        "geometry_qualified": bool(np.isfinite(distance) and distance <= 2.0),
    }


def bootstrap_difference(
    lost: Iterable[float],
    retained: Iterable[float],
    statistic: Callable[[np.ndarray], float],
    seed: int,
    iterations: int = 5000,
) -> dict:
    left = np.asarray([v for v in lost if np.isfinite(v)], dtype=np.float64)
    right = np.asarray([v for v in retained if np.isfinite(v)], dtype=np.float64)
    if not left.size or not right.size:
        return {"estimate": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "iterations": iterations}
    rng = np.random.default_rng(int(seed))
    values = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        a = left[rng.integers(0, left.size, left.size)]
        b = right[rng.integers(0, right.size, right.size)]
        values[index] = float(statistic(a) - statistic(b))
    return {
        "estimate": float(statistic(left) - statistic(right)),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "iterations": iterations,
    }

