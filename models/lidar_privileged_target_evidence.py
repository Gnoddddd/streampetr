"""Train-only LiDAR/GT target-evidence supervision helpers."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def _box_corners(boxes: Tensor) -> Tensor:
    """Create [G,8,3] corners from center/size/yaw LiDAR boxes."""
    device, dtype = boxes.device, boxes.dtype
    signs = torch.tensor([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], device=device, dtype=dtype)
    # nuScenes stores size as width, length, height while box-local x points
    # forward (length) and y points left (width).
    local_size = boxes[:, [4, 3, 5]].abs()
    local = signs.unsqueeze(0) * local_size[:, None, :] / 2.0
    cosine, sine = torch.cos(boxes[:, 6]), torch.sin(boxes[:, 6])
    x = local[..., 0] * cosine[:, None] - local[..., 1] * sine[:, None]
    y = local[..., 0] * sine[:, None] + local[..., 1] * cosine[:, None]
    rotated = torch.stack((x, y, local[..., 2]), dim=-1)
    return rotated + boxes[:, None, :3]


def projected_box_visibility(
    gt_boxes: Tensor,
    lidar2img: Tensor,
    image_hw: tuple[int, int],
) -> Tensor:
    """Return [GT,camera] physical visibility using clipped corner boxes."""
    if gt_boxes.ndim != 2 or gt_boxes.shape[1] < 7:
        raise ValueError("gt_boxes must be [GT,>=7]")
    if lidar2img.ndim != 3 or lidar2img.shape[-2:] != (4, 4):
        raise ValueError("lidar2img must be [camera,4,4]")
    work_dtype = torch.float64 if gt_boxes.dtype == torch.float64 else torch.float32
    corners = _box_corners(gt_boxes.detach().to(dtype=work_dtype))
    homogeneous = torch.cat((corners, torch.ones_like(corners[..., :1])), dim=-1)
    projected = torch.einsum(
        "cij,gnj->gcni", lidar2img.detach().to(dtype=work_dtype), homogeneous
    )
    depth = projected[..., 2]
    x = projected[..., 0] / depth.clamp_min(1e-6)
    y = projected[..., 1] / depth.clamp_min(1e-6)
    height, width = (float(value) for value in image_hw)
    output = torch.zeros(gt_boxes.shape[0], lidar2img.shape[0], dtype=torch.bool,
                         device=gt_boxes.device)
    for gt in range(gt_boxes.shape[0]):
        for camera in range(lidar2img.shape[0]):
            front = depth[gt, camera] > 1e-5
            if int(front.sum()) < 2:
                continue
            xx, yy = x[gt, camera, front], y[gt, camera, front]
            x0 = xx.min().clamp(0.0, width)
            x1 = xx.max().clamp(0.0, width)
            y0 = yy.min().clamp(0.0, height)
            y1 = yy.max().clamp(0.0, height)
            output[gt, camera] = bool((x1 > x0) & (y1 > y0))
    return output


def select_target_evidence(
    logits: Tensor,
    boxes: Tensor,
    gt_boxes: Tensor,
    gt_labels: Tensor,
    lidar_supported: Tensor,
    lidar2img: Tensor,
    image_hw: tuple[int, int],
    fault_camera: int = 3,
    geometry_threshold: float = 2.0,
    fault_active: bool = True,
) -> tuple[List[Dict], List[Dict]]:
    """Detached selection of no-alternative-view GT-near S_pos candidates."""
    if logits.ndim != 2 or boxes.ndim != 2:
        raise ValueError("logits and boxes must be [query,...]")
    if not (len(gt_boxes) == len(gt_labels) == len(lidar_supported)):
        raise ValueError("GT inputs must have equal length")
    if not 0 <= int(fault_camera) < lidar2img.shape[0]:
        raise ValueError("fault_camera is out of range")
    selected: List[Dict] = []
    diagnostics: List[Dict] = []
    with torch.no_grad():
        visibility = projected_box_visibility(gt_boxes, lidar2img, image_hw)
        distances = torch.cdist(
            boxes[:, :3].detach().float(), gt_boxes[:, :3].detach().float()
        ) if len(gt_boxes) else boxes.new_empty((len(boxes), 0))
        scores = logits.detach().float().sigmoid()
        for gt in range(len(gt_boxes)):
            label = int(gt_labels[gt])
            near = torch.nonzero(
                torch.isfinite(distances[:, gt])
                & (distances[:, gt] <= float(geometry_threshold)),
                as_tuple=False,
            ).flatten()
            alternative = int(visibility[gt].sum()) - int(visibility[gt, fault_camera])
            lidar_ok = bool(lidar_supported[gt])
            back_visible = bool(visibility[gt, fault_camera])
            eligible = bool(
                fault_active and lidar_ok and back_visible
                and alternative == 0 and near.numel() > 0
            )
            query = -1
            center_distance = float("nan")
            score = float("nan")
            if near.numel():
                position = int(torch.argmax(scores[near, label]))
                query = int(near[position])
                center_distance = float(distances[query, gt])
                score = float(scores[query, label])
            item = {
                "gt": gt, "gt_class": label, "lidar_supported": lidar_ok,
                "fault_camera_visible": back_visible,
                "alternative_view_count": alternative,
                "near_query_count": int(near.numel()), "positive_query": query,
                "center_distance": center_distance, "s_pos": score,
                "eligible": eligible, "selected": eligible,
                "duplicate_suppressed": False,
            }
            diagnostics.append(item)
            if eligible:
                selected.append(item)
        # Deduplicate only identical query/class supervision. Retain the closer GT.
        keep = {}
        for item in selected:
            key = (int(item["positive_query"]), int(item["gt_class"]))
            if key not in keep or item["center_distance"] < keep[key]["center_distance"]:
                if key in keep:
                    diagnostics[int(keep[key]["gt"])]["selected"] = False
                    diagnostics[int(keep[key]["gt"])]["duplicate_suppressed"] = True
                keep[key] = item
            else:
                diagnostics[int(item["gt"])]["selected"] = False
                diagnostics[int(item["gt"])]["duplicate_suppressed"] = True
        selected = list(keep.values())
    return selected, diagnostics


def target_evidence_loss(
    logits: Tensor,
    selected: List[Dict],
) -> tuple[Tensor, List[Dict]]:
    """Positive BCE through selected GT-class logits only; no ranking term."""
    terms, diagnostics = [], []
    for item in selected:
        value = logits[int(item["positive_query"]), int(item["gt_class"])].float()
        loss = F.softplus(-value)
        terms.append(loss)
        diagnostics.append({
            **item, "raw_loss": float(loss.detach()),
            "analytic_gradient_magnitude": float((1.0 - value.sigmoid()).detach()),
            "finite": bool(torch.isfinite(loss.detach())),
        })
    zero = logits.sum() * 0.0
    return (torch.stack(terms).mean() if terms else zero), diagnostics
