"""Pure helpers for the frozen camera-reliability audit."""

from __future__ import annotations

from typing import Iterable

import numpy as np


CAMERA_NAMES = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def aggregate_camera_attention(
    attention: np.ndarray,
    token_camera: np.ndarray,
    camera_count: int = 6,
) -> np.ndarray:
    """Sum query-to-token attention into query-to-camera attention mass."""
    weights = np.asarray(attention, dtype=np.float64)
    source = np.asarray(token_camera, dtype=np.int64)
    if weights.ndim != 2:
        raise ValueError("attention must have shape [query, token]")
    if source.shape != (weights.shape[1],):
        raise ValueError("token_camera length must equal token count")
    if np.any(source < 0) or np.any(source >= camera_count):
        raise ValueError("token camera index out of range")
    output = np.zeros((weights.shape[0], camera_count), dtype=np.float64)
    for camera in range(camera_count):
        output[:, camera] = weights[:, source == camera].sum(axis=1)
    return output.astype(np.float32)


def deployed_query_indices(
    logits: np.ndarray,
    boxes: np.ndarray,
    max_num: int,
    post_center_range: Iterable[float],
) -> np.ndarray:
    """Reproduce NMSFreeCoder's deployed-query selection and range filter."""
    logits = np.asarray(logits, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    if logits.ndim != 2 or boxes.ndim != 2 or boxes.shape[0] != logits.shape[0]:
        raise ValueError("invalid logits/boxes shapes")
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    flat = scores.reshape(-1)
    count = min(int(max_num), flat.size)
    # Stable descending order mirrors top-k for non-tied floating scores while
    # keeping the helper deterministic in dependency-light unit tests.
    selected = np.argsort(-flat, kind="stable")[:count]
    query = selected // logits.shape[1]
    bounds = np.asarray(tuple(post_center_range), dtype=np.float64)
    if bounds.shape != (6,):
        raise ValueError("post_center_range must have six values")
    centers = boxes[query, :3]
    valid = np.all(centers >= bounds[:3], axis=1)
    valid &= np.all(centers <= bounds[3:], axis=1)
    return query[valid].astype(np.int64)


def safe_correlation(x: Iterable[float], y: Iterable[float]) -> float:
    """Pearson correlation, returning NaN for insufficient/constant data."""
    left = np.asarray(tuple(x), dtype=np.float64)
    right = np.asarray(tuple(y), dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if left.size < 3 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def rank_correlation(x: Iterable[float], y: Iterable[float]) -> float:
    """Dependency-free Spearman correlation using average ranks for ties."""
    left = np.asarray(tuple(x), dtype=np.float64)
    right = np.asarray(tuple(y), dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if left.size < 3:
        return float("nan")

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        output = np.empty(values.size, dtype=np.float64)
        start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and values[order[end]] == values[order[start]]:
                end += 1
            output[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        return output

    return safe_correlation(ranks(left), ranks(right))

