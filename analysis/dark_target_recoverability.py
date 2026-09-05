"""Pure helpers for the Dark target-evidence recoverability audit."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def darken_normalized_image(image, factor, mean, std):
    mean = image.new_tensor(mean).view(-1, 1, 1)
    std = image.new_tensor(std).view(-1, 1, 1)
    return float(factor) * image + (float(factor) - 1.0) * mean / std


def projected_roi(corners, lidar2img, image_hw, eps=1e-5):
    points = np.asarray(corners, dtype=np.float64)
    if points.shape[0] == 3:
        points = points.T
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("corners must have shape [N,3] or [3,N]")
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    projection = (np.asarray(lidar2img, dtype=np.float64) @ points.T).T
    front = projection[:, 2] > float(eps)
    if np.count_nonzero(front) < 2:
        return None
    xy = projection[front, :2] / projection[front, 2:3]
    height, width = map(float, image_hw)
    x0, y0 = np.min(xy, axis=0)
    x1, y1 = np.max(xy, axis=0)
    x0, x1 = np.clip([x0, x1], 0.0, width)
    y0, y1 = np.clip([y0, y1], 0.0, height)
    if not (x1 > x0 and y1 > y0):
        return None
    return float(x0), float(y0), float(x1), float(y1)


def roi_cell_mask(roi, image_hw, feature_hw, fallback=True):
    x0, y0, x1, y1 = map(float, roi)
    image_h, image_w = map(float, image_hw)
    feature_h, feature_w = map(int, feature_hw)
    xs = (np.arange(feature_w) + 0.5) * image_w / feature_w
    ys = (np.arange(feature_h) + 0.5) * image_h / feature_h
    mask = ((ys[:, None] >= y0) & (ys[:, None] < y1)
            & (xs[None, :] >= x0) & (xs[None, :] < x1))
    if fallback and not np.any(mask):
        column = int(np.clip(math.floor((x0 + x1) / 2 / image_w * feature_w),
                             0, feature_w - 1))
        row = int(np.clip(math.floor((y0 + y1) / 2 / image_h * feature_h),
                          0, feature_h - 1))
        mask[row, column] = True
    return mask


def background_ring_mask(roi, other_rois, image_hw, feature_hw):
    x0, y0, x1, y1 = map(float, roi)
    height, width = map(float, image_hw)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rw, rh = x1 - x0, y1 - y0
    expanded = (max(0, cx - rw), max(0, cy - rh),
                min(width, cx + rw), min(height, cy + rh))
    ring = roi_cell_mask(expanded, image_hw, feature_hw, fallback=False)
    ring &= ~roi_cell_mask(roi, image_hw, feature_hw, fallback=False)
    for other in other_rois:
        ring &= ~roi_cell_mask(other, image_hw, feature_hw, fallback=False)
    return ring


def replace_local_feature(destination, source, mask, camera_index):
    output = destination.clone()
    mask = torch.as_tensor(mask, device=output.device, dtype=torch.bool)
    output[int(camera_index), :, mask] = source[int(camera_index), :, mask]
    return output


def centroid_separability(feature, target_mask, background_mask):
    if feature.ndim != 3:
        raise ValueError("feature must be [C,H,W]")
    target = torch.as_tensor(target_mask, device=feature.device, dtype=torch.bool)
    background = torch.as_tensor(background_mask, device=feature.device, dtype=torch.bool)
    if not torch.any(target) or not torch.any(background):
        return {"target_cells": int(target.sum()), "background_cells": int(background.sum()),
                "cosine_distance": float("nan"), "normalized_l2": float("nan")}
    left, right = feature[:, target].T.float(), feature[:, background].T.float()
    lm, rm = left.mean(0), right.mean(0)
    cosine = torch.nn.functional.cosine_similarity(lm[None], rm[None], eps=1e-8)[0]
    numerator = torch.linalg.vector_norm(lm - rm)
    lv = torch.mean(torch.sum((left - lm) ** 2, -1))
    rv = torch.mean(torch.sum((right - rm) ** 2, -1))
    return {"target_cells": len(left), "background_cells": len(right),
            "cosine_distance": float(1 - cosine),
            "normalized_l2": float(numerator / torch.sqrt(0.5 * (lv + rv) + 1e-8))}


def recovery_fraction(clean, dark, variant):
    denominator = float(clean) - float(dark)
    return (float(variant) - float(dark)) / denominator if denominator > 0 else float("nan")


def destructive_fraction(clean, dark, variant):
    denominator = float(clean) - float(dark)
    return (float(clean) - float(variant)) / denominator if denominator > 0 else float("nan")


def match_retained_controls(lost_rows, retained_rows):
    pairs = []
    for frame in sorted({str(row["sample_token"]) for row in lost_rows}):
        lost = sorted([r for r in lost_rows if str(r["sample_token"]) == frame],
                      key=lambda r: str(r["gt_token"]))
        retained = sorted([r for r in retained_rows if str(r["sample_token"]) == frame],
                          key=lambda r: str(r["gt_token"]))
        if len(retained) < len(lost):
            raise ValueError(f"frame {frame} lacks retained controls")
        costs = np.empty((len(lost), len(retained)))
        for i, left in enumerate(lost):
            for j, right in enumerate(retained):
                costs[i, j] = (
                    10 * (str(left["gt_class"]) != str(right["gt_class"]))
                    + abs(float(left["gt_center_distance"]) - float(right["gt_center_distance"])) / 20
                    + abs(float(left["alternative_view_count"]) - float(right["alternative_view_count"]))
                    + abs(math.log(float(left["max_projected_box_area_fraction"]) + 1e-6)
                          - math.log(float(right["max_projected_box_area_fraction"]) + 1e-6))
                    + abs(float(left["clean_s_pos"]) - float(right["clean_s_pos"])) + j * 1e-12)
        rows, columns = linear_sum_assignment(costs)
        pairs.extend({"lost": lost[int(i)], "retained": retained[int(j)],
                      "match_cost": float(costs[i, j])} for i, j in zip(rows, columns))
    return pairs


def bootstrap_median(values, seed, iterations=5000):
    values = np.asarray(tuple(values), dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"estimate": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "iterations": iterations}
    rng = np.random.default_rng(seed)
    estimates = np.asarray([np.median(values[rng.integers(0, len(values), len(values))])
                            for _ in range(iterations)])
    return {"estimate": float(np.median(values)), "ci_low": float(np.percentile(estimates, 2.5)),
            "ci_high": float(np.percentile(estimates, 97.5)), "iterations": iterations}


def bootstrap_median_difference(left, right, seed, iterations=5000):
    left, right = np.asarray(tuple(left), float), np.asarray(tuple(right), float)
    left, right = left[np.isfinite(left)], right[np.isfinite(right)]
    if not len(left) or not len(right):
        return {"estimate": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "iterations": iterations}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        a = left[rng.integers(0, len(left), len(left))]
        b = right[rng.integers(0, len(right), len(right))]
        values.append(np.median(a) - np.median(b))
    return {"estimate": float(np.median(left) - np.median(right)),
            "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5)), "iterations": iterations}
