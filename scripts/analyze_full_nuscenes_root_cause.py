#!/usr/bin/env python3
"""Full-val paired root-cause, counterfactual, and visibility analysis."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import mmcv
import numpy as np
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes

from analysis.fault_boundary_root_cause import (
    candidate_pool_statistics,
    projected_box_visibility,
    rescue_category,
)


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "reports/full_nuscenes/mechanism_confirmation/paired_inference"
REPORT = ROOT / "reports/full_nuscenes/mechanism_confirmation/root_cause"
DATA = ROOT / "data/nuscenes"
INFO = DATA / "nuscenes2d_temporal_infos_val.pkl"
PROTOCOLS = {
    "dark_back": "CAM_BACK Dark",
    "blur_back": "CAM_BACK Motion Blur",
    "crash_back": "CAM_BACK Crash",
}
CLASSES = (
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)
CLASS_INDEX = {name: index for index, name in enumerate(CLASSES)}
BOOTSTRAPS = 5000
SEED = 314159


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def finite_median(rows: list[dict], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def rate(rows: list[dict], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows])) if rows else float("nan")


def scene_trajectory_estimates(
    rows: list[dict], key: str, trajectory_stat: str = "median",
) -> np.ndarray:
    """Collapse frames to trajectories, then trajectories to scene estimates."""
    by_scene: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = float(row[key])
        if math.isfinite(value):
            by_scene[str(row["scene_token"])][str(row["instance_token"])].append(value)
    estimates = []
    for trajectories in by_scene.values():
        trajectory_values = []
        for values in trajectories.values():
            if trajectory_stat == "rate":
                trajectory_values.append(float(np.mean(values)))
            else:
                trajectory_values.append(float(np.median(values)))
        if trajectory_values:
            # Each scene receives equal bootstrap weight; trajectories prevent
            # repeated frames from being treated as independent observations.
            estimates.append(float(np.mean(trajectory_values)))
    return np.asarray(estimates, dtype=float)


def cluster_ci(
    rows: list[dict], key: str, seed: int, group: Optional[str] = None,
) -> dict:
    selected = rows if group is None else [row for row in rows if row["outcome"] == group]
    scene_values = scene_trajectory_estimates(selected, key)
    estimate = float(np.mean(scene_values)) if scene_values.size else float("nan")
    if not scene_values.size or not math.isfinite(estimate):
        return {"estimate": estimate, "ci_low": float("nan"),
                "ci_high": float("nan"), "iterations": BOOTSTRAPS,
                "scene_clusters": int(scene_values.size)}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, scene_values.size, size=(BOOTSTRAPS, scene_values.size))
    values = np.mean(scene_values[indices], axis=1)
    low, high = np.percentile(values, [2.5, 97.5])
    return {"estimate": estimate, "ci_low": float(low), "ci_high": float(high),
            "iterations": BOOTSTRAPS, "scene_clusters": int(scene_values.size)}


def cluster_rate_ci(
    rows: list[dict], key: str, seed: int, group: Optional[str] = None,
) -> dict:
    selected = rows if group is None else [row for row in rows if row["outcome"] == group]
    scene_values = scene_trajectory_estimates(selected, key, trajectory_stat="rate")
    estimate = float(np.mean(scene_values)) if scene_values.size else float("nan")
    if not scene_values.size:
        return {"estimate": estimate, "ci_low": float("nan"), "ci_high": float("nan"),
                "iterations": BOOTSTRAPS, "scene_clusters": 0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, scene_values.size, size=(BOOTSTRAPS, scene_values.size))
    values = np.mean(scene_values[indices], axis=1)
    low, high = np.percentile(values, [2.5, 97.5])
    return {"estimate": estimate, "ci_low": float(low), "ci_high": float(high),
            "iterations": BOOTSTRAPS, "scene_clusters": int(scene_values.size)}


def cluster_difference_ci(rows: list[dict], key: str, seed: int) -> dict:
    differences = []
    for scene in sorted({str(row["scene_token"]) for row in rows}):
        scene_rows = [row for row in rows if str(row["scene_token"]) == scene]
        lost = scene_trajectory_estimates(
            [row for row in scene_rows if row["outcome"] == "fault_induced_lost"], key
        )
        retained = scene_trajectory_estimates(
            [row for row in scene_rows if row["outcome"] == "retained"], key
        )
        if lost.size and retained.size:
            differences.append(float(lost[0] - retained[0]))
    scene_values = np.asarray(differences, dtype=float)
    estimate = float(np.mean(scene_values)) if scene_values.size else float("nan")
    if not scene_values.size:
        return {"estimate": estimate, "ci_low": float("nan"), "ci_high": float("nan"),
                "iterations": BOOTSTRAPS, "scene_clusters": 0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, scene_values.size, size=(BOOTSTRAPS, scene_values.size))
    values = np.mean(scene_values[indices], axis=1)
    low, high = np.percentile(values, [2.5, 97.5])
    return {"estimate": estimate, "ci_low": float(low), "ci_high": float(high),
            "iterations": BOOTSTRAPS, "scene_clusters": int(scene_values.size)}


def load_trace(group: str, token: str) -> dict[str, np.ndarray]:
    path = RUN / group / "trace" / f"{token}.npz"
    with np.load(path) as value:
        return {key: value[key].copy() for key in value.files}


def assert_active_protocol_trace(group: str, trace: dict[str, np.ndarray]) -> None:
    expected_online = np.ones(6, dtype=float)
    expected_quality = np.ones(6, dtype=float)
    expected_severity = np.zeros(6, dtype=float)
    if group in ("dark_back", "blur_back"):
        expected_quality[3] = 0.1
        expected_severity[3] = 0.9
    elif group == "crash_back":
        expected_online[3] = 0.0
        expected_quality[3] = 0.0
        expected_severity[3] = 1.0
    for key, expected in (
        ("camera_online", expected_online),
        ("camera_quality", expected_quality),
        ("corruption_severity", expected_severity),
    ):
        if not np.allclose(np.asarray(trace[key], dtype=float), expected, atol=1e-6):
            raise RuntimeError(f"frozen protocol mismatch: {group} {key}={trace[key]}")


def official_matches(nusc: NuScenes, token: str, predictions: dict) -> set[str]:
    sample = nusc.get("sample", token)
    gt = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        name = category_to_detection_name(ann["category_name"])
        if name in CLASS_INDEX:
            gt.append((ann_token, name, np.asarray(ann["translation"][:2], float)))
    candidates = [value for value in predictions["results"].get(token, [])
                  if float(value["detection_score"]) >= 0.1]
    pairs = []
    for gt_index, (_, name, center) in enumerate(gt):
        for pred_index, prediction in enumerate(candidates):
            if prediction["detection_name"] != name:
                continue
            distance = float(np.linalg.norm(
                np.asarray(prediction["translation"][:2], float) - center
            ))
            if distance <= 2.0:
                pairs.append((distance, gt_index, pred_index))
    used_gt, used_prediction, matched = set(), set(), set()
    for _, gt_index, pred_index in sorted(pairs):
        if gt_index in used_gt or pred_index in used_prediction:
            continue
        used_gt.add(gt_index)
        used_prediction.add(pred_index)
        matched.add(gt[gt_index][0])
    return matched


def local_gt(nusc: NuScenes, token: str) -> list[dict]:
    sample = nusc.get("sample", token)
    _, boxes, _ = nusc.get_sample_data(sample["data"]["LIDAR_TOP"])
    output = []
    for box in boxes:
        name = category_to_detection_name(box.name)
        if name not in CLASS_INDEX:
            continue
        ann = nusc.get("sample_annotation", box.token)
        output.append({
            "token": box.token,
            "instance_token": ann["instance_token"],
            "name": name,
            "label": CLASS_INDEX[name],
            "center": np.asarray(box.center, float),
            "corners": np.asarray(box.corners(), float),
            "visibility_token": int(ann["visibility_token"]),
        })
    return output


def distance_bin(value: float) -> str:
    if value < 20.0:
        return "[0,20)"
    if value < 40.0:
        return "[20,40)"
    return "[40,+inf)"


def metrics_summary(group: str) -> dict:
    path = RUN / group / "formatted" / "pts_bbox" / "metrics_summary.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    return {"protocol": group, "condition": "Clean" if group == "clean" else PROTOCOLS[group],
            "mAP": value["mean_ap"], "NDS": value["nd_score"],
            "mATE": value["tp_errors"]["trans_err"],
            "mASE": value["tp_errors"]["scale_err"],
            "mAOE": value["tp_errors"]["orient_err"],
            "mAVE": value["tp_errors"]["vel_err"],
            "mAAE": value["tp_errors"]["attr_err"]}


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    info = sorted(mmcv.load(str(INFO))["infos"], key=lambda row: row["timestamp"])
    active = [row for row in info if 3 <= int(row["frame_idx"]) <= 12]
    if len(active) != 1500:
        raise RuntimeError(f"expected 1500 active frames, found {len(active)}")
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(DATA), verbose=False)
    payloads = {
        group: json.loads((RUN / group / "formatted" / "pts_bbox" /
                           "results_nusc.json").read_text(encoding="utf-8"))
        for group in ("clean", *PROTOCOLS)
    }
    full_metrics = [metrics_summary(group) for group in ("clean", *PROTOCOLS)]
    write_csv(REPORT / "full_val_metrics.csv", full_metrics)

    rows: list[dict] = []
    for frame_number, info_row in enumerate(active, 1):
        token = str(info_row["token"])
        scene = str(info_row["scene_token"])
        clean_frame = load_trace("clean", token)
        assert_active_protocol_trace("clean", clean_frame)
        clean_matches = official_matches(nusc, token, payloads["clean"])
        gt_values = local_gt(nusc, token)
        for protocol, condition in PROTOCOLS.items():
            fault_frame = load_trace(protocol, token)
            assert_active_protocol_trace(protocol, fault_frame)
            fault_matches = official_matches(nusc, token, payloads[protocol])
            for gt in gt_values:
                if gt["token"] not in clean_matches:
                    continue
                outcome = ("retained" if gt["token"] in fault_matches
                           else "fault_induced_lost")
                clean = candidate_pool_statistics(
                    clean_frame["layer_logits"][-1], clean_frame["layer_boxes"][-1],
                    gt["center"], gt["label"], 100, 2.0,
                )
                fault = candidate_pool_statistics(
                    fault_frame["layer_logits"][-1], fault_frame["layer_boxes"][-1],
                    gt["center"], gt["label"], 100, 2.0,
                )
                visibility = projected_box_visibility(
                    gt["corners"], clean_frame["lidar2img"], clean_frame["image_hw"]
                )
                visible = visibility["visible"]
                alternative = int(np.count_nonzero(np.delete(visible, 3)))
                distance = float(np.linalg.norm(gt["center"][:2]))
                if fault["candidate_available"]:
                    delta_s_pos = float(fault["s_pos"] - clean["s_pos"])
                    delta_s_k = float(fault["s_k"] - clean["s_k"])
                    delta_margin = float(fault["margin"] - clean["margin"])
                    target_cf = float(clean["s_pos"] - fault["s_k"])
                    boundary_cf = float(fault["s_pos"] - clean["s_k"])
                    rescue = rescue_category(target_cf, boundary_cf)
                    decomposition = delta_margin - (delta_s_pos - delta_s_k)
                else:
                    delta_s_pos = delta_margin = target_cf = boundary_cf = float("nan")
                    delta_s_k = float(fault["s_k"] - clean["s_k"])
                    rescue = "candidate_missing"
                    decomposition = float("nan")
                rows.append({
                    "protocol": protocol, "condition": condition,
                    "sample_token": token, "scene_token": scene,
                    "frame_idx": int(info_row["frame_idx"]),
                    "gt_token": gt["token"], "instance_token": gt["instance_token"],
                    "gt_class": gt["name"], "outcome": outcome,
                    "distance_m": distance, "distance_bin": distance_bin(distance),
                    "visibility_token": gt["visibility_token"],
                    "cam_back_visible": bool(visible[3]),
                    "physical_visible_view_count": int(np.count_nonzero(visible)),
                    "alternative_view_count": alternative,
                    "alternative_view_zero": alternative == 0,
                    "has_alternative_view": alternative > 0,
                    "max_projected_area_fraction": float(np.max(visibility["area_fraction"])),
                    "clean_candidate_count": int(clean["count"]),
                    "fault_candidate_count": int(fault["count"]),
                    "fault_candidate_available": bool(fault["candidate_available"]),
                    "clean_s_pos": clean["s_pos"], "fault_s_pos": fault["s_pos"],
                    "clean_s_k": clean["s_k"], "fault_s_k": fault["s_k"],
                    "delta_s_pos": delta_s_pos, "delta_s_k": delta_s_k,
                    "target_burden": (max(-delta_s_pos, 0.0)
                                      if math.isfinite(delta_s_pos) else float("nan")),
                    "competitor_burden": max(delta_s_k, 0.0),
                    "clean_margin": clean["margin"], "fault_margin": fault["margin"],
                    "delta_margin": delta_margin, "decomposition_error": decomposition,
                    "clean_rank": clean["rank"], "fault_rank": fault["rank"],
                    "topk_crossing": bool(fault["candidate_available"]
                                          and clean["rank"] <= 100 < fault["rank"]),
                    "m_cf_target": target_cf, "m_cf_boundary": boundary_cf,
                    "target_rescue": bool(math.isfinite(target_cf) and target_cf > 0),
                    "boundary_rescue": bool(math.isfinite(boundary_cf) and boundary_cf > 0),
                    "rescue_category": rescue,
                })
        if frame_number % 100 == 0:
            print(f"active frames {frame_number}/{len(active)} rows={len(rows)}", flush=True)
    write_csv(REPORT / "per_gt.csv", rows)

    summary_rows = []
    for protocol in (*PROTOCOLS, "pooled"):
        protocol_rows = rows if protocol == "pooled" else [r for r in rows if r["protocol"] == protocol]
        for outcome in ("fault_induced_lost", "retained"):
            selected = [r for r in protocol_rows if r["outcome"] == outcome]
            counter = Counter(r["rescue_category"] for r in selected)
            summary_rows.append({
                "protocol": protocol,
                "condition": "Pooled" if protocol == "pooled" else PROTOCOLS[protocol],
                "outcome": outcome, "n": len(selected),
                "fault_candidate_coverage": rate(selected, "fault_candidate_available"),
                "median_delta_s_pos": finite_median(selected, "delta_s_pos"),
                "median_delta_s_k": finite_median(selected, "delta_s_k"),
                "median_target_burden": finite_median(selected, "target_burden"),
                "median_competitor_burden": finite_median(selected, "competitor_burden"),
                "median_delta_margin": finite_median(selected, "delta_margin"),
                "topk_crossing_rate": rate(selected, "topk_crossing"),
                "target_rescue_rate": rate(selected, "target_rescue"),
                "boundary_rescue_rate": rate(selected, "boundary_rescue"),
                "target_only_n": counter["target-driven"],
                "boundary_only_n": counter["competitor-driven"],
                "mixed_n": counter["mixed"], "neither_n": counter["neither"],
                "alternative_view_zero_rate": float(np.mean([
                    int(r["alternative_view_count"]) == 0 for r in selected
                ])) if selected else float("nan"),
                "median_alternative_view_count": finite_median(selected, "alternative_view_count"),
            })
    write_csv(REPORT / "mechanism_summary.csv", summary_rows)

    scene_groups = defaultdict(list)
    for row in rows:
        scene_groups[(row["protocol"], row["scene_token"], row["outcome"])].append(row)
    scene_rows = []
    for (protocol, scene, outcome), selected in sorted(scene_groups.items()):
        scene_rows.append({"protocol": protocol, "scene_token": scene,
            "outcome": outcome, "n": len(selected),
            "median_delta_s_pos": finite_median(selected, "delta_s_pos"),
            "median_delta_s_k": finite_median(selected, "delta_s_k"),
            "topk_crossing_rate": rate(selected, "topk_crossing"),
            "target_rescue_rate": rate(selected, "target_rescue"),
            "boundary_rescue_rate": rate(selected, "boundary_rescue"),
            "alternative_view_zero_rate": float(np.mean([
                int(r["alternative_view_count"]) == 0 for r in selected]))})
    write_csv(REPORT / "per_scene.csv", scene_rows)

    ci_rows = []
    ci_index = 0
    for protocol in (*PROTOCOLS, "pooled"):
        selected = rows if protocol == "pooled" else [r for r in rows if r["protocol"] == protocol]
        for outcome in ("fault_induced_lost", "retained"):
            for metric in ("delta_s_pos", "delta_s_k", "delta_margin",
                           "target_burden", "competitor_burden",
                           "alternative_view_count"):
                result = cluster_ci(selected, metric, SEED + ci_index, outcome)
                ci_index += 1
                ci_rows.append({"category": "scene_mean_of_trajectory_medians", "protocol": protocol,
                    "outcome": outcome, "metric": metric,
                    "cluster": "scene_bootstrap_on_trajectory_aggregates", **result})
            result = cluster_rate_ci(
                selected, "alternative_view_zero", SEED + ci_index, outcome
            )
            ci_index += 1
            ci_rows.append({"category": "rate", "protocol": protocol,
                "outcome": outcome, "metric": "alternative_view_zero_rate",
                "cluster": "scene_bootstrap_on_trajectory_aggregates", **result})
        for metric in ("delta_s_pos", "delta_s_k", "delta_margin",
                       "target_burden", "competitor_burden", "alternative_view_count"):
            result = cluster_difference_ci(selected, metric, SEED + ci_index)
            ci_index += 1
            ci_rows.append({"category": "paired_scene_lost_minus_retained", "protocol": protocol,
                "outcome": "contrast", "metric": metric,
                "cluster": "scene_bootstrap_on_trajectory_aggregates", **result})
    write_csv(REPORT / "cluster_bootstrap_ci.csv", ci_rows)

    strata_rows = []
    dimensions = {
        "class": "gt_class", "distance": "distance_bin",
        "visibility": "visibility_token", "alternative_view": "has_alternative_view",
    }
    for protocol in (*PROTOCOLS, "pooled"):
        protocol_rows = rows if protocol == "pooled" else [r for r in rows if r["protocol"] == protocol]
        for dimension, field in dimensions.items():
            for value in sorted({str(r[field]) for r in protocol_rows}):
                subset = [r for r in protocol_rows if str(r[field]) == value]
                for outcome in ("fault_induced_lost", "retained"):
                    chosen = [r for r in subset if r["outcome"] == outcome]
                    result = cluster_ci(chosen, "delta_s_pos", SEED + ci_index)
                    ci_index += 1
                    strata_rows.append({"protocol": protocol, "dimension": dimension,
                        "stratum": value, "outcome": outcome, "n": len(chosen),
                        "median_delta_s_pos": finite_median(chosen, "delta_s_pos"),
                        "topk_crossing_rate": rate(chosen, "topk_crossing"),
                        "alternative_view_zero_rate": float(np.mean([
                            int(r["alternative_view_count"]) == 0 for r in chosen
                        ])) if chosen else float("nan"),
                        "cluster": "scene_bootstrap_on_trajectory_aggregates",
                        "ci_low": result["ci_low"], "ci_high": result["ci_high"]})
    write_csv(REPORT / "stratified.csv", strata_rows)

    max_decomposition = max(abs(float(r["decomposition_error"])) for r in rows
                            if math.isfinite(float(r["decomposition_error"])))
    summary_index = {(r["protocol"], r["outcome"]): r for r in summary_rows}
    ci_indexed = {(r["category"], r["protocol"], r["outcome"], r["metric"]): r for r in ci_rows}
    pooled_ci = ci_indexed[("scene_mean_of_trajectory_medians", "pooled",
                            "fault_induced_lost", "delta_s_pos")]
    contrast_ci = ci_indexed[("paired_scene_lost_minus_retained", "pooled",
                              "contrast", "delta_s_pos")]
    alternative_ci = ci_indexed[("rate", "pooled", "fault_induced_lost",
                                 "alternative_view_zero_rate")]
    direction = all(summary_index[(p, "fault_induced_lost")]["median_delta_s_pos"] < 0
                    for p in PROTOCOLS)
    protocol_ci_direction = all(
        ci_indexed[("scene_mean_of_trajectory_medians", p,
                    "fault_induced_lost", "delta_s_pos")]["ci_high"] < 0
        for p in PROTOCOLS
    )
    no_boundary_inflation = all(
        summary_index[(p, "fault_induced_lost")]["median_delta_s_k"] <= 0
        for p in PROTOCOLS
    )
    target_dominates = (
        summary_index[("pooled", "fault_induced_lost")]["median_target_burden"]
        > summary_index[("pooled", "fault_induced_lost")]["median_competitor_burden"]
    )
    crossing = all(summary_index[(p, "fault_induced_lost")]["topk_crossing_rate"] > 0
                   for p in PROTOCOLS)
    stronger_than_retained = contrast_ci["ci_high"] < 0
    confirmed = bool(direction and protocol_ci_direction and pooled_ci["ci_high"] < 0
                     and no_boundary_inflation
                     and target_dominates and crossing and stronger_than_retained)
    mini_zero_rate = 51.0 / 53.0
    mini_sampling_limited = bool(alternative_ci["ci_high"] < mini_zero_rate)
    decision = [{"root_cause_confirmed": confirmed,
        "delta_s_pos_negative_all_protocols": direction,
        "delta_s_pos_ci_below_zero_all_protocols": protocol_ci_direction,
        "pooled_delta_s_pos_ci_below_zero": pooled_ci["ci_high"] < 0,
        "delta_s_k_nonpositive_all_protocols": no_boundary_inflation,
        "pooled_target_burden_exceeds_competitor_burden": target_dominates,
        "crossing_present_all_protocols": crossing,
        "lost_stronger_than_retained": stronger_than_retained,
        "mini_alternative_view_zero_rate": mini_zero_rate,
        "full_lost_alternative_view_zero_rate": alternative_ci["estimate"],
        "full_lost_alternative_view_zero_ci_low": alternative_ci["ci_low"],
        "full_lost_alternative_view_zero_ci_high": alternative_ci["ci_high"],
        "mini_51_of_53_consistent_with_sampling_limit": mini_sampling_limited,
        "max_decomposition_error": max_decomposition}]
    write_csv(REPORT / "mechanism_decision.csv", decision)
    (REPORT / "root_cause_decision.json").write_text(
        json.dumps(decision[0], indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision[0], indent=2), flush=True)


if __name__ == "__main__":
    main()
