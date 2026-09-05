"""Pure matching, geometry and gate helpers for LiDAR signal audit."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def circular_error(left: float, right: float) -> float:
    return float(abs((float(left) - float(right) + math.pi) % (2 * math.pi)
                     - math.pi))


def greedy_class_center_match(gt: list[dict], labels, centers, scores,
                              threshold: float = 2.0) -> dict[str, int]:
    """One-to-one class-aware BEV-center matching with deterministic ties."""
    labels = np.asarray(labels, int)
    centers = np.asarray(centers, float)
    scores = np.asarray(scores, float)
    edges = []
    for gt_index, target in enumerate(gt):
        for prediction_index, (label, center, score) in enumerate(
                zip(labels, centers, scores)):
            if int(target["label"]) != int(label):
                continue
            distance = float(np.linalg.norm(
                np.asarray(target["center"], float)[:2] - center[:2]))
            if distance <= float(threshold):
                edges.append((distance, -float(score), str(target["token"]),
                              prediction_index, gt_index))
    used_gt, used_predictions, output = set(), set(), {}
    for _, _, _, prediction_index, gt_index in sorted(edges):
        if gt_index in used_gt or prediction_index in used_predictions:
            continue
        used_gt.add(gt_index)
        used_predictions.add(prediction_index)
        output[str(gt[gt_index]["token"])] = int(prediction_index)
    return output


def sample_bev_features(feature: torch.Tensor, centers: np.ndarray,
                        point_cloud_range=(-51.2, -51.2, 51.2, 51.2),
                        stride: float = 0.8) -> np.ndarray:
    """Bilinearly sample a BEV tensor at LiDAR-frame object centers."""
    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError(f"expected [1,C,H,W], got {tuple(feature.shape)}")
    centers = np.asarray(centers, float)
    if not len(centers):
        return np.empty((0, int(feature.shape[1])), np.float32)
    height, width = int(feature.shape[2]), int(feature.shape[3])
    x_index = (centers[:, 0] - point_cloud_range[0]) / stride - 0.5
    y_index = (centers[:, 1] - point_cloud_range[1]) / stride - 0.5
    grid = np.stack([2 * x_index / (width - 1) - 1,
                     2 * y_index / (height - 1) - 1], axis=-1)
    grid = torch.as_tensor(grid, device=feature.device,
                           dtype=feature.dtype).view(1, -1, 1, 2)
    sampled = F.grid_sample(feature, grid, mode="bilinear",
                            padding_mode="zeros", align_corners=True)
    return sampled[0, :, :, 0].transpose(0, 1).detach().float().cpu().numpy()


def cosine_distance(left, right) -> float:
    left, right = np.asarray(left, float), np.asarray(right, float)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= 1e-12:
        return float("nan")
    return float(1.0 - np.dot(left, right) / denominator)


def teacher_coverage_decision(protocols: dict[str, dict], pooled: dict,
                              temporal: dict) -> dict:
    protocol_coverage = all(float(value["lost_match_rate"]) >= 0.70
                            for value in protocols.values())
    protocol_score = all(float(value["lost_median_score"]) >= 0.20
                         for value in protocols.values())
    protocol_center = all(float(value["lost_median_xy_error"]) <= 1.50
                          for value in protocols.values())
    coverage = (float(pooled["lost_match_rate"]) >= 0.80
                and float(pooled["retained_match_rate"]) >= 0.80
                and protocol_coverage)
    quality = (float(pooled["lost_median_score"]) >= 0.25
               and float(pooled["lost_median_xy_error"]) <= 1.0
               and float(pooled["lost_median_relative_size_l1"]) <= 0.35
               and float(pooled["lost_median_yaw_error_deg"]) <= 30.0
               and protocol_score and protocol_center)
    stability = (int(temporal["lost_pair_count"]) >= 10
                 and float(temporal["median_abs_score_delta"]) <= 0.20
                 and float(temporal["median_abs_center_error_delta"]) <= 0.50
                 and float(temporal["median_representation_cosine_distance"]) <= 0.30)
    passed = bool(coverage and quality and stability)
    return {"teacher_coverage_pass": passed,
            "coverage_rate_pass": bool(coverage),
            "teacher_quality_pass": bool(quality),
            "teacher_temporal_stability_pass": bool(stability),
            "decision": ("CONTINUE_TO_CROSS_MODAL_GAP_AUDIT" if passed
                         else "NO_GO_LIDAR_TEACHER_COVERAGE")}


def cross_modal_signal_decision(coverage_pass: bool, gap: dict,
                                superiority: dict) -> dict:
    score_gap = (float(gap["score_median"]) >= 0.05
                 and float(gap["score_ci_low"]) > 0.0
                 and gap["score_cross_protocol"]
                 and float(gap["score_enrichment_median"]) >= 0.03
                 and float(gap["score_enrichment_ci_low"]) > 0.0)
    representation_gap = (
        float(gap["representation_median"]) >= 0.02
        and float(gap["representation_ci_low"]) > 0.0
        and gap["representation_cross_protocol"]
        and float(gap["representation_enrichment_median"]) >= 0.01
        and float(gap["representation_enrichment_ci_low"]) > 0.0)
    geometry = (float(gap["center_median"]) >= 0.10
                and float(gap["center_ci_low"]) > 0.0
                and gap["center_cross_protocol"])
    lost_specific = bool(score_gap and representation_gap)
    target_strength = bool(superiority["score_strength"]
                           or superiority["geometry_strength"])
    temporal = bool(superiority["temporal_representation"]
                    and superiority["temporal_score_or_geometry"])
    lidar_better = bool(target_strength and temporal)
    if not coverage_pass:
        decision = "NO_GO_LIDAR_TEACHER_COVERAGE"
    elif not lost_specific:
        decision = "NO_GO_NO_LOST_SPECIFIC_LIDAR_GAP"
    elif not lidar_better:
        decision = "NO_GO_LIDAR_NOT_BETTER_THAN_CLEAN_TEACHER"
    else:
        decision = "GO_LIDAR_PRIVILEGED_SIGNAL"
    return {"decision": decision, "score_gap_pass": bool(score_gap),
            "representation_gap_pass": bool(representation_gap),
            "geometry_corroboration": bool(geometry),
            "lost_specific_gap_pass": lost_specific,
            "target_strength_pass": target_strength,
            "temporal_stability_advantage_pass": temporal,
            "lidar_better_than_clean_pass": lidar_better,
            "signal_pass": decision == "GO_LIDAR_PRIVILEGED_SIGNAL"}
