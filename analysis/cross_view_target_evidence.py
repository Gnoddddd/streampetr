"""Pure helpers for cross-view target-evidence decomposition."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch


VARIANTS = (
    "cam_back_target_clean",
    "other_visible_target_clean",
    "all_visible_target_clean",
    "all_visible_background_clean",
)


def same_area_background_mask(target_mask, excluded_mask):
    """Choose nearest native background cells with exactly the target area."""
    target = np.asarray(target_mask, dtype=bool)
    excluded = np.asarray(excluded_mask, dtype=bool)
    if target.shape != excluded.shape or target.ndim != 2:
        raise ValueError("target and excluded masks must be same-shape 2-D arrays")
    count = int(np.count_nonzero(target))
    if count == 0:
        return None
    candidates = np.argwhere(~excluded)
    if len(candidates) < count:
        return None
    center = np.argwhere(target).mean(axis=0)
    distance = np.sum((candidates - center[None]) ** 2, axis=1)
    order = np.lexsort((candidates[:, 1], candidates[:, 0], distance))
    chosen = candidates[order[:count]]
    output = np.zeros_like(target)
    output[chosen[:, 0], chosen[:, 1]] = True
    return output


def patch_camera_masks(destination, source, masks: Mapping[int, np.ndarray]):
    """Replace native cells for a mapping of camera index to spatial mask."""
    if destination.shape != source.shape or destination.ndim != 4:
        raise ValueError("features must be same-shape [camera,C,H,W]")
    output = destination.clone()
    for camera, mask in masks.items():
        selected = torch.as_tensor(mask, device=output.device, dtype=torch.bool)
        if tuple(selected.shape) != tuple(output.shape[-2:]):
            raise ValueError("mask shape does not match feature map")
        output[int(camera), :, selected] = source[int(camera), :, selected]
    return output


def stable_rescue(summary: dict) -> bool:
    return bool(
        float(summary["tp_recovery_rate"]) >= 0.5
        and float(summary["topk_recovery_rate"]) >= 0.5
        and float(summary["median_rescue_fraction"]) >= 0.5
    )


def weak_background(summary: dict) -> bool:
    return bool(
        float(summary["tp_recovery_rate"]) <= 0.2
        and float(summary["topk_recovery_rate"]) <= 0.2
        and float(summary["median_rescue_fraction"]) < 0.25
    )


def classify_mechanism(summaries: Mapping[str, dict], evidence_available: bool,
                       alternative_coverage: float) -> dict:
    """Apply the preregistered mutually exclusive mechanism decision tree."""
    back = summaries["cam_back_target_clean"]
    other = summaries["other_visible_target_clean"]
    all_views = summaries["all_visible_target_clean"]
    background = summaries["all_visible_background_clean"]
    back_stable = stable_rescue(back)
    other_stable = stable_rescue(other)
    all_stable = stable_rescue(all_views)
    background_weak = weak_background(background)
    synergy = bool(
        float(all_views["tp_recovery_rate"])
        >= max(float(back["tp_recovery_rate"]), float(other["tp_recovery_rate"])) + 0.2
        and float(all_views["median_rescue_fraction"])
        >= max(float(back["median_rescue_fraction"]),
               float(other["median_rescue_fraction"])) + 0.25
    )
    if evidence_available and all_stable and (other_stable or synergy) and background_weak:
        mechanism = "distributed_multi_view_evidence"
    elif (evidence_available and back_stable and not other_stable and all_stable
          and float(all_views["tp_recovery_rate"])
          < float(back["tp_recovery_rate"]) + 0.2 and background_weak):
        mechanism = "alternative_evidence_underuse"
    elif (back_stable and not other_stable and all_stable and background_weak
          and (not evidence_available or float(alternative_coverage) < 0.2)):
        mechanism = "primary_view_dependence"
    else:
        mechanism = "unsupported"
    return {
        "mechanism": mechanism,
        "back_stable": back_stable,
        "other_stable": other_stable,
        "all_stable": all_stable,
        "background_weak": background_weak,
        "synergy": synergy,
        "evidence_available": bool(evidence_available),
        "alternative_coverage": float(alternative_coverage),
    }


def alternative_stratum(count: int) -> str:
    count = int(count)
    return "0" if count <= 0 else "1" if count == 1 else "2+"
