"""Pure helpers for the fault-boundary root-cause audit."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def stable_rank(values: np.ndarray, index: int) -> int:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    index = int(index)
    target = flat[index]
    return int(np.count_nonzero(flat > target)
               + np.count_nonzero(flat[:index] == target) + 1)


def candidate_pool_statistics(
    logits: np.ndarray,
    boxes: np.ndarray,
    gt_center: Iterable[float],
    gt_class: int,
    topk: int = 100,
    radius: float = 2.0,
) -> dict:
    """Return the exact deployment boundary and all GT-near candidates."""
    logits = np.asarray(logits, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    if logits.ndim != 2 or boxes.ndim != 2 or boxes.shape[0] != logits.shape[0]:
        raise ValueError("logits and boxes must be query-major 2-D arrays")
    if not 0 <= int(gt_class) < logits.shape[1]:
        raise ValueError("gt_class is out of range")
    scores = sigmoid(logits)
    flat = scores.reshape(-1)
    k = min(max(int(topk), 1), flat.size)
    boundary = float(np.partition(flat, flat.size - k)[flat.size - k])
    center = np.asarray(tuple(gt_center), dtype=np.float64)[:3]
    distance = np.linalg.norm(boxes[:, :3] - center, axis=1)
    near = np.flatnonzero(
        np.isfinite(boxes[:, :3]).all(axis=1)
        & np.isfinite(distance) & (distance <= float(radius))
    )
    candidate_scores = scores[near, int(gt_class)]
    if near.size:
        order = np.lexsort((near, -candidate_scores))
        near = near[order]
        candidate_scores = candidate_scores[order]
        candidate_distances = distance[near]
        best_query = int(near[0])
        best_score = float(candidate_scores[0])
        best_distance = float(candidate_distances[0])
        rank = stable_rank(flat, best_query * logits.shape[1] + int(gt_class))
        margin = best_score - boundary
    else:
        candidate_distances = np.empty(0, dtype=np.float64)
        best_query = -1
        best_score = float("nan")
        best_distance = float(np.nanmin(distance)) if distance.size else float("nan")
        rank = -1
        margin = float("nan")
    return {
        "candidate_available": bool(near.size),
        "queries": near.astype(np.int64),
        "scores": candidate_scores.astype(np.float64),
        "distances": candidate_distances.astype(np.float64),
        "count": int(near.size),
        "best_query": best_query,
        "s_pos": best_score,
        "s_k": boundary,
        "margin": float(margin),
        "rank": int(rank),
        "best_distance": best_distance,
    }


def fixed_query_statistics(
    logits: np.ndarray,
    boxes: np.ndarray,
    query: int,
    gt_center: Iterable[float],
    gt_class: int,
) -> dict:
    scores = sigmoid(logits)
    boxes = np.asarray(boxes, dtype=np.float64)
    query = int(query)
    distance = float(np.linalg.norm(
        boxes[query, :3] - np.asarray(tuple(gt_center), dtype=np.float64)[:3]
    ))
    flat_index = query * scores.shape[1] + int(gt_class)
    return {
        "score": float(scores[query, int(gt_class)]),
        "center_distance": distance,
        "rank": stable_rank(scores, flat_index),
        "geometry_qualified": bool(np.isfinite(distance) and distance <= 2.0),
    }


def regression_cost(
    box: Iterable[float],
    gt_center: Iterable[float],
    gt_size: Iterable[float],
    gt_yaw: float,
    radius: float = 2.0,
) -> float:
    """Detached Stage-4 composite center/size/yaw regression cost (lower is better)."""
    box = np.asarray(tuple(box), dtype=np.float64)
    center = np.asarray(tuple(gt_center), dtype=np.float64)
    size = np.asarray(tuple(gt_size), dtype=np.float64)
    distance = float(np.linalg.norm(box[:3] - center[:3]))
    safe_size = np.maximum(np.abs(box[3:6]), 1e-4)
    target_size = np.maximum(np.abs(size[:3]), 1e-4)
    size_error = float(np.mean(np.abs(np.log(safe_size / target_size))))
    yaw_error = abs(float((box[6] - float(gt_yaw) + np.pi) % (2 * np.pi) - np.pi))
    return distance / float(radius) + size_error + yaw_error / np.pi


def rescue_category(m_cf_target: float, m_cf_boundary: float) -> str:
    target = bool(np.isfinite(m_cf_target) and m_cf_target > 0.0)
    boundary = bool(np.isfinite(m_cf_boundary) and m_cf_boundary > 0.0)
    if target and boundary:
        return "mixed"
    if target:
        return "target-driven"
    if boundary:
        return "competitor-driven"
    return "neither"


def count_matched_clean_max(
    clean_scores: Iterable[float],
    target_count: int,
    seed: int,
    repeats: int = 10000,
) -> dict:
    """Monte Carlo best-of-N counterfactual with fixed without-replacement draws."""
    values = np.asarray(tuple(clean_scores), dtype=np.float64)
    target_count = int(target_count)
    if target_count <= 0 or target_count > values.size:
        raise ValueError("target_count must be in [1, len(clean_scores)]")
    if target_count == values.size:
        maxima = np.asarray([np.max(values)], dtype=np.float64)
        effective_repeats = 1
    else:
        rng = np.random.default_rng(int(seed))
        # Candidate pools are small; explicit draws preserve exact no-replacement semantics.
        maxima = np.empty(int(repeats), dtype=np.float64)
        for index in range(int(repeats)):
            maxima[index] = np.max(rng.choice(values, target_count, replace=False))
        effective_repeats = int(repeats)
    return {
        "expected_max": float(np.mean(maxima)),
        "p025": float(np.percentile(maxima, 2.5)),
        "p50": float(np.percentile(maxima, 50.0)),
        "p975": float(np.percentile(maxima, 97.5)),
        "effective_repeats": effective_repeats,
    }


def average_ranks(values: Iterable[float]) -> np.ndarray:
    """Average one-based ranks with deterministic tie handling."""
    values = np.asarray(tuple(values), dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def auroc(risk: Iterable[float], outcome: Iterable[int]) -> float:
    risk = np.asarray(tuple(risk), dtype=np.float64)
    outcome = np.asarray(tuple(outcome), dtype=np.int64)
    finite = np.isfinite(risk)
    risk, outcome = risk[finite], outcome[finite]
    positives = outcome == 1
    n_pos, n_neg = int(np.count_nonzero(positives)), int(np.count_nonzero(~positives))
    if not n_pos or not n_neg:
        return float("nan")
    ranks = average_ranks(risk)
    return float((np.sum(ranks[positives]) - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def spearman(left: Iterable[float], right: Iterable[float]) -> float:
    left = np.asarray(tuple(left), dtype=np.float64)
    right = np.asarray(tuple(right), dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    x, y = average_ranks(left[finite]), average_ranks(right[finite])
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def projected_box_visibility(
    corners: np.ndarray,
    lidar2img: np.ndarray,
    image_hw: Iterable[int],
) -> dict:
    """Visible camera mask and clipped projected 3-D-box area fractions."""
    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape[0] == 3:
        corners = corners.T
    if corners.ndim != 2 or corners.shape[1] != 3:
        raise ValueError("corners must have shape [3,N] or [N,3]")
    points = np.concatenate([corners, np.ones((corners.shape[0], 1))], axis=1)
    matrices = np.asarray(lidar2img, dtype=np.float64)
    height, width = (float(value) for value in image_hw)
    visible = np.zeros(matrices.shape[0], dtype=bool)
    area_fraction = np.zeros(matrices.shape[0], dtype=np.float64)
    for camera, matrix in enumerate(matrices):
        projected = (matrix @ points.T).T
        front = projected[:, 2] > 1e-5
        if np.count_nonzero(front) < 2:
            continue
        xy = projected[front, :2] / projected[front, 2:3]
        x0, y0 = np.min(xy, axis=0)
        x1, y1 = np.max(xy, axis=0)
        x0, x1 = np.clip([x0, x1], 0.0, width)
        y0, y1 = np.clip([y0, y1], 0.0, height)
        area = max(float(x1 - x0), 0.0) * max(float(y1 - y0), 0.0)
        if area > 0.0:
            visible[camera] = True
            area_fraction[camera] = area / (height * width)
    return {"visible": visible, "area_fraction": area_fraction}
