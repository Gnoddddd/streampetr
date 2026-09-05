"""Pure geometry and representation helpers for temporal tap localization."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def geometry_candidates(boxes: np.ndarray, gt_center, radius: float = 2.0) -> np.ndarray:
    """Return query indices selected only by finite 3-D center geometry."""

    boxes = np.asarray(boxes, dtype=np.float64)
    center = np.asarray(gt_center, dtype=np.float64)[:3]
    distances = np.linalg.norm(boxes[:, :3] - center, axis=1)
    indices = np.flatnonzero(
        np.isfinite(boxes[:, :3]).all(axis=1)
        & np.isfinite(distances)
        & (distances <= float(radius))
    )
    return indices[np.lexsort((indices, distances[indices]))].astype(np.int64)


def geometry_match(
    source_boxes: np.ndarray,
    target_boxes: np.ndarray,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
) -> list[tuple[int, int, float]]:
    """Deterministic score-free rectangular matching of candidate centers."""

    source_indices = np.asarray(source_indices, dtype=np.int64)
    target_indices = np.asarray(target_indices, dtype=np.int64)
    if not source_indices.size or not target_indices.size:
        return []
    source = np.asarray(source_boxes, dtype=np.float64)[source_indices, :3]
    target = np.asarray(target_boxes, dtype=np.float64)[target_indices, :3]
    geometric = np.linalg.norm(source[:, None, :] - target[None, :, :], axis=-1)
    # Only resolves exact equal-cost assignments; geometry remains dominant by
    # nine orders of magnitude relative to the bounded tie-break.
    tie = 1e-9 * (
        np.arange(len(source_indices), dtype=np.float64)[:, None] * (len(target_indices) + 1)
        + np.arange(len(target_indices), dtype=np.float64)[None, :]
    )
    rows, columns = linear_sum_assignment(geometric + tie)
    pairs = [
        (int(source_indices[row]), int(target_indices[column]), float(geometric[row, column]))
        for row, column in zip(rows, columns)
    ]
    return sorted(pairs, key=lambda value: (value[1], value[0]))


def local_non_gt_candidates(
    boxes: np.ndarray,
    all_gt_centers: np.ndarray,
    audit_gt_center,
    count: int,
    exclusion_radius: float = 2.0,
) -> np.ndarray:
    """Choose a same-count local control set outside every GT neighborhood."""

    boxes = np.asarray(boxes, dtype=np.float64)
    centers = np.asarray(all_gt_centers, dtype=np.float64)[:, :3]
    count = int(count)
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    finite = np.isfinite(boxes[:, :3]).all(axis=1)
    if len(centers):
        distances_to_all = np.linalg.norm(
            boxes[:, None, :3] - centers[None, :, :], axis=-1
        )
        outside = (distances_to_all > float(exclusion_radius)).all(axis=1)
    else:
        outside = np.ones(len(boxes), dtype=bool)
    candidates = np.flatnonzero(finite & outside)
    audit_distance = np.linalg.norm(
        boxes[candidates, :3] - np.asarray(audit_gt_center, dtype=np.float64)[:3], axis=1
    )
    ordered = candidates[np.lexsort((candidates, audit_distance))]
    return ordered[:count].astype(np.int64)


def matched_representation_metrics(
    source_representation: np.ndarray,
    target_representation: np.ndarray,
    pairs: list[tuple[int, int, float]],
) -> dict:
    """Median pair drift for a GT, retaining pair-level values for audit."""

    cosine_distances = []
    normalized_l2 = []
    for source_index, target_index, _ in pairs:
        source = np.asarray(source_representation[source_index], dtype=np.float64)
        target = np.asarray(target_representation[target_index], dtype=np.float64)
        source_norm = float(np.linalg.norm(source))
        target_norm = float(np.linalg.norm(target))
        if source_norm == 0 or target_norm == 0:
            cosine = float("nan")
            l2 = float("nan")
        else:
            source_unit = source / source_norm
            target_unit = target / target_norm
            cosine = float(1.0 - np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))
            l2 = float(np.linalg.norm(source_unit - target_unit))
        cosine_distances.append(cosine)
        normalized_l2.append(l2)
    cosine_values = np.asarray(cosine_distances, dtype=np.float64)
    l2_values = np.asarray(normalized_l2, dtype=np.float64)
    return {
        "matched_pair_count": len(pairs),
        "cosine_distance": float(np.nanmedian(cosine_values)) if len(pairs) else float("nan"),
        "normalized_l2": float(np.nanmedian(l2_values)) if len(pairs) else float("nan"),
        "pair_cosine_distances": cosine_distances,
        "pair_normalized_l2": normalized_l2,
    }
