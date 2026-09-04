"""Pure geometry/ranking helpers for GT-query survival analysis."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2 * math.pi) - math.pi)


def flat_class_rank(logits: np.ndarray, query_index: int, class_index: int) -> int:
    """One-based deployment rank of a query/class pair among all pairs."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("logits must have shape [query, class]")
    score = values[int(query_index), int(class_index)]
    flat = values.reshape(-1)
    # Stable tie breaking follows flattened query-major order.
    target = int(query_index) * values.shape[1] + int(class_index)
    better = int(np.count_nonzero(flat > score))
    tied_before = int(np.count_nonzero(flat[:target] == score))
    return better + tied_before + 1


def geometry_statistics(
    boxes: np.ndarray,
    gt_center: Iterable[float],
    gt_size: Iterable[float],
    gt_yaw: float,
    threshold: float = 2.0,
) -> dict:
    """Count GT-near queries and return the best detached geometry query."""
    boxes = np.asarray(boxes, dtype=np.float64)
    center = np.asarray(tuple(gt_center), dtype=np.float64)
    size = np.asarray(tuple(gt_size), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError("boxes must have shape [query, >=7]")
    distance = np.linalg.norm(boxes[:, :3] - center[:3], axis=1)
    finite = np.isfinite(boxes[:, :7]).all(axis=1)
    near = np.flatnonzero(finite & (distance <= float(threshold)))
    if not near.size:
        return {
            "near_count": 0,
            "best_query": -1,
            "center_distance": float(np.nanmin(distance)) if distance.size else float("nan"),
            "geometry_cost": float("inf"),
        }
    safe_size = np.maximum(np.abs(boxes[near, 3:6]), 1e-4)
    target_size = np.maximum(np.abs(size[:3]), 1e-4)
    size_error = np.mean(np.abs(np.log(safe_size / target_size)), axis=1)
    yaw_error = np.asarray([abs(wrap_angle(v - gt_yaw)) for v in boxes[near, 6]])
    cost = distance[near] / threshold + size_error + yaw_error / math.pi
    best_position = int(np.argmin(cost))
    best = int(near[best_position])
    return {
        "near_count": int(near.size),
        "best_query": best,
        "center_distance": float(distance[best]),
        "geometry_cost": float(cost[best_position]),
    }


def projected_feature_support(
    center: Iterable[float],
    lidar2img: np.ndarray,
    token_index: np.ndarray,
    token_norm: np.ndarray,
    feature_hw: Iterable[int],
    image_hw: Iterable[int],
    radius_cells: float = 2.0,
) -> dict:
    """Measure selected ROI-token support around a projected GT center."""
    matrices = np.asarray(lidar2img, dtype=np.float64)
    indices = np.asarray(token_index, dtype=np.int64).reshape(-1)
    norms = np.asarray(token_norm, dtype=np.float64).reshape(-1)
    height, width = (int(v) for v in feature_hw)
    image_height, image_width = (float(v) for v in image_hw)
    point = np.asarray((*tuple(center)[:3], 1.0), dtype=np.float64)
    projected = matrices @ point
    camera_count = matrices.shape[0]
    token_camera = indices // (height * width)
    spatial = indices % (height * width)
    token_y, token_x = spatial // width, spatial % width
    supported = 0
    best_norm = 0.0
    best_distance = float("inf")
    visible = 0
    for camera in range(camera_count):
        depth = float(projected[camera, 2])
        if depth <= 1e-5:
            continue
        x = float(projected[camera, 0] / depth)
        y = float(projected[camera, 1] / depth)
        if not (0 <= x < image_width and 0 <= y < image_height):
            continue
        visible += 1
        feature_x = x / image_width * width
        feature_y = y / image_height * height
        mask = token_camera == camera
        if not np.any(mask):
            continue
        distance = np.sqrt(
            (token_x[mask] - feature_x) ** 2 + (token_y[mask] - feature_y) ** 2
        )
        close = distance <= radius_cells
        if np.any(close):
            supported += 1
            close_norms = norms[mask][close]
            best_norm = max(best_norm, float(np.nanmax(close_norms)))
            best_distance = min(best_distance, float(np.nanmin(distance[close])))
    return {
        "visible_cameras": visible,
        "supported_cameras": supported,
        "best_feature_norm": best_norm,
        "best_token_distance": best_distance,
    }

