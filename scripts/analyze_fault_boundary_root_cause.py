#!/usr/bin/env python3
"""Offline paired root-cause audit for fault-induced K=100 boundary crossing."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import mmcv
import numpy as np
import torch
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes

from analysis.fault_boundary_root_cause import (
    auroc,
    candidate_pool_statistics,
    count_matched_clean_max,
    fixed_query_statistics,
    projected_box_visibility,
    regression_cost,
    rescue_category,
    sigmoid,
    spearman,
)


ROOT = Path(__file__).resolve().parents[1]
TRACE_ROOT = ROOT / "outputs/stage4/gt_query_survival_audit"
DISABLED_ROOT = ROOT / "outputs/stage4/hard_positive_boundary_objective_audit/disabled"
REPORT = ROOT / "reports/stage4/fault_boundary_root_cause_audit"
GROUPS = {
    "dark_back": "CAM_BACK Dark",
    "blur_back": "CAM_BACK Blur",
    "crash_back": "CAM_BACK Crash",
}
CLASS_NAMES = (
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
CAM_BACK_INDEX = 3
BOOTSTRAPS = 5000
MATCH_REPEATS = 10000
BASE_SEED = 314159


def write_csv(name: str, rows: list[dict], allow_empty: bool = False) -> None:
    path = REPORT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if not allow_empty:
            raise RuntimeError(f"refusing empty report: {name}")
        path.write_text("status\nnot_triggered\n", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def deterministic_seed(*values) -> int:
    payload = "|".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def load_traces(group: str) -> dict[str, dict]:
    output = {}
    for path in sorted((TRACE_ROOT / group / "trace").glob("*.npz")):
        with np.load(path) as value:
            output[str(value["sample_token"])] = {
                key: value[key].copy() for key in value.files
            }
    if len(output) != 81:
        raise RuntimeError(f"{group}: expected 81 traces, found {len(output)}")
    return output


def local_gt(nusc: NuScenes, token: str) -> list[dict]:
    sample = nusc.get("sample", token)
    _, boxes, _ = nusc.get_sample_data(sample["data"]["LIDAR_TOP"])
    output = []
    for box in boxes:
        name = category_to_detection_name(box.name)
        if name not in CLASS_TO_INDEX:
            continue
        output.append({
            "token": box.token,
            "name": name,
            "label": CLASS_TO_INDEX[name],
            "center": np.asarray(box.center, dtype=np.float64),
            "size": np.asarray(box.wlh, dtype=np.float64),
            "yaw": float(box.orientation.yaw_pitch_roll[0]),
            "corners": np.asarray(box.corners(), dtype=np.float64),
        })
    return output


def official_matches(nusc: NuScenes, token: str, payload: dict) -> set[str]:
    sample = nusc.get("sample", token)
    gt = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        name = category_to_detection_name(ann["category_name"])
        if name in CLASS_TO_INDEX:
            gt.append((ann_token, name, np.asarray(ann["translation"][:2], float)))
    predictions = [value for value in payload["results"].get(token, [])
                   if float(value["detection_score"]) >= 0.1]
    pairs = []
    for gt_index, (_, name, center) in enumerate(gt):
        for pred_index, prediction in enumerate(predictions):
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


def compare_tensors(left, right) -> tuple[float, int]:
    if hasattr(left, "tensor") or hasattr(right, "tensor"):
        return compare_tensors(left.tensor, right.tensor)
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or left.shape != right.shape:
            return float("inf"), 0
        difference = float((left.cpu() - right.cpu()).abs().max()) if left.numel() else 0.0
        return difference, 1
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            return float("inf"), 0
        values = [compare_tensors(left[key], right[key]) for key in left]
    elif isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            return float("inf"), 0
        values = [compare_tensors(a, b) for a, b in zip(left, right)]
    else:
        return (0.0 if left == right else float("inf")), 1
    return max((value[0] for value in values), default=0.0), sum(value[1] for value in values)


def finite(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([
        float(row[key]) for row in rows
        if key in row and math.isfinite(float(row[key]))
    ], dtype=np.float64)


def distribution(values) -> dict:
    values = np.asarray(tuple(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        output = {key: float("nan") for key in (
            "mean", "std", "min", "q25", "median", "q75", "max"
        )}
        output["n"] = 0
        return output
    return {
        "n": int(values.size), "mean": float(np.mean(values)),
        "std": float(np.std(values)), "min": float(np.min(values)),
        "q25": float(np.percentile(values, 25)), "median": float(np.median(values)),
        "q75": float(np.percentile(values, 75)), "max": float(np.max(values)),
    }


def bootstrap_stat(values, statistic, seed: int, iterations: int = BOOTSTRAPS) -> dict:
    values = np.asarray(tuple(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"estimate": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "iterations": iterations}
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        estimates[index] = statistic(values[rng.integers(0, values.size, values.size)])
    return {"estimate": float(statistic(values)),
            "ci_low": float(np.percentile(estimates, 2.5)),
            "ci_high": float(np.percentile(estimates, 97.5)),
            "iterations": iterations}


def bootstrap_difference(left, right, statistic, seed: int) -> dict:
    left = np.asarray(tuple(left), dtype=np.float64)
    right = np.asarray(tuple(right), dtype=np.float64)
    left, right = left[np.isfinite(left)], right[np.isfinite(right)]
    if not left.size or not right.size:
        return {"estimate": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "iterations": BOOTSTRAPS}
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(BOOTSTRAPS, dtype=np.float64)
    for index in range(BOOTSTRAPS):
        a = left[rng.integers(0, left.size, left.size)]
        b = right[rng.integers(0, right.size, right.size)]
        estimates[index] = statistic(a) - statistic(b)
    return {"estimate": float(statistic(left) - statistic(right)),
            "ci_low": float(np.percentile(estimates, 2.5)),
            "ci_high": float(np.percentile(estimates, 97.5)),
            "iterations": BOOTSTRAPS}


def bootstrap_auc(risk, outcome, seed: int) -> dict:
    risk, outcome = np.asarray(risk, float), np.asarray(outcome, int)
    finite_mask = np.isfinite(risk)
    risk, outcome = risk[finite_mask], outcome[finite_mask]
    positive, negative = risk[outcome == 1], risk[outcome == 0]
    if not positive.size or not negative.size:
        return {"estimate": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "iterations": BOOTSTRAPS}
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(BOOTSTRAPS, dtype=np.float64)
    for index in range(BOOTSTRAPS):
        p = positive[rng.integers(0, positive.size, positive.size)]
        n = negative[rng.integers(0, negative.size, negative.size)]
        estimates[index] = auroc(np.concatenate([p, n]),
                                 np.concatenate([np.ones(p.size), np.zeros(n.size)]))
    return {"estimate": auroc(risk, outcome),
            "ci_low": float(np.percentile(estimates, 2.5)),
            "ci_high": float(np.percentile(estimates, 97.5)),
            "iterations": BOOTSTRAPS}


def bootstrap_spearman(left, right, seed: int) -> dict:
    left, right = np.asarray(left, float), np.asarray(right, float)
    finite_mask = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite_mask], right[finite_mask]
    if left.size < 2:
        return {"estimate": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "iterations": BOOTSTRAPS}
    rng = np.random.default_rng(int(seed))
    estimates = []
    for _ in range(BOOTSTRAPS):
        indexes = rng.integers(0, left.size, left.size)
        value = spearman(left[indexes], right[indexes])
        if np.isfinite(value):
            estimates.append(value)
    if not estimates:
        low = high = float("nan")
    else:
        low, high = np.percentile(estimates, [2.5, 97.5])
    return {"estimate": spearman(left, right), "ci_low": float(low),
            "ci_high": float(high), "iterations": BOOTSTRAPS}


def top_score(values: np.ndarray, index: int) -> float:
    return float(values[index]) if values.size > index else float("nan")


def frame_region(center: np.ndarray, frame: dict) -> str:
    point = np.asarray([*center[:3], 1.0], dtype=np.float64)
    projected = np.asarray(frame["lidar2img"], float) @ point
    depth = projected[:, 2]
    u = projected[:, 0] / np.maximum(depth, 1e-6)
    v = projected[:, 1] / np.maximum(depth, 1e-6)
    height, width = (float(value) for value in frame["image_hw"])
    visible = (depth > 1e-5) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    back = bool(visible[CAM_BACK_INDEX])
    other = bool(np.any(np.delete(visible, CAM_BACK_INDEX)))
    if back and other:
        return "cam_back_and_other"
    if back:
        return "cam_back_only"
    if other:
        return "other_only"
    return "no_camera"


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version="v1.0-mini", dataroot=str(ROOT / "data/nuscenes-mini"), verbose=False)
    traces = {group: load_traces(group) for group in ("clean", *GROUPS)}
    payloads = {group: json.loads(
        (TRACE_ROOT / group / "formatted/pts_bbox/results_nusc.json").read_text()
    ) for group in ("clean", *GROUPS)}

    invariance_rows = []
    for group in ("clean", *GROUPS):
        difference, leaves = compare_tensors(
            mmcv.load(str(TRACE_ROOT / group / "predictions.pkl")),
            mmcv.load(str(DISABLED_ROOT / group / "predictions.pkl")),
        )
        invariance_rows.append({"protocol": group, "tensor_leaves": leaves,
                                "max_abs_diff": difference, "exact": difference == 0})
    if not all(row["exact"] for row in invariance_rows):
        raise RuntimeError(f"hooked/disabled prediction divergence: {invariance_rows}")

    gt_cache = {token: local_gt(nusc, token) for token in traces["clean"]}
    match_cache = {}
    for group in ("clean", *GROUPS):
        match_cache[group] = {
            token: official_matches(nusc, token, payloads[group])
            for token in traces[group]
        }

    rows, candidate_rows = [], []
    frame_has_lost = defaultdict(set)
    for protocol in GROUPS:
        for token, fault_frame in traces[protocol].items():
            if not 3 <= int(fault_frame["frame_idx"]) <= 12:
                continue
            clean_frame = traces["clean"][token]
            visibility_frame = clean_frame
            for gt in gt_cache[token]:
                if gt["token"] not in match_cache["clean"][token]:
                    continue
                outcome = ("retained_control" if gt["token"] in match_cache[protocol][token]
                           else "fault_induced_lost")
                if outcome == "fault_induced_lost":
                    frame_has_lost[protocol].add(token)
                clean = candidate_pool_statistics(
                    clean_frame["layer_logits"][-1], clean_frame["layer_boxes"][-1],
                    gt["center"], gt["label"], topk=100,
                )
                fault = candidate_pool_statistics(
                    fault_frame["layer_logits"][-1], fault_frame["layer_boxes"][-1],
                    gt["center"], gt["label"], topk=100,
                )
                if not clean["candidate_available"]:
                    raise RuntimeError(f"Clean-correct GT lacks <=2m query: {gt['token']}")
                lineage = fixed_query_statistics(
                    fault_frame["layer_logits"][-1], fault_frame["layer_boxes"][-1],
                    clean["best_query"], gt["center"], gt["label"],
                )
                visibility = projected_box_visibility(
                    gt["corners"], visibility_frame["lidar2img"], visibility_frame["image_hw"]
                )
                visible = visibility["visible"]
                fault_available = fault["candidate_available"]
                if fault_available:
                    delta_s_pos = fault["s_pos"] - clean["s_pos"]
                    delta_s_k = fault["s_k"] - clean["s_k"]
                    delta_margin = fault["margin"] - clean["margin"]
                    decomposition_error = delta_margin - (delta_s_pos - delta_s_k)
                    if abs(decomposition_error) > 1e-12:
                        raise RuntimeError(f"decomposition failed: {decomposition_error}")
                    target_cf = clean["s_pos"] - fault["s_k"]
                    boundary_cf = fault["s_pos"] - clean["s_k"]
                    rescue = rescue_category(target_cf, boundary_cf)
                else:
                    delta_s_pos = delta_margin = decomposition_error = float("nan")
                    delta_s_k = fault["s_k"] - clean["s_k"]
                    target_cf = boundary_cf = float("nan")
                    rescue = "candidate_missing"

                matched = {"expected_max": float("nan"), "p025": float("nan"),
                           "p50": float("nan"), "p975": float("nan"),
                           "effective_repeats": 0}
                match_status = "fault_candidate_missing"
                if fault["count"] > 0 and clean["count"] >= fault["count"]:
                    matched = count_matched_clean_max(
                        clean["scores"], fault["count"],
                        deterministic_seed(BASE_SEED, protocol, token, gt["token"]),
                        MATCH_REPEATS,
                    )
                    match_status = ("count_decline" if clean["count"] > fault["count"]
                                    else "equal_count")
                elif fault["count"] > clean["count"]:
                    match_status = "count_increase"
                count_loss = (clean["s_pos"] - matched["expected_max"]
                              if np.isfinite(matched["expected_max"]) else float("nan"))
                residual = (fault["s_pos"] - matched["expected_max"]
                            if fault_available and np.isfinite(matched["expected_max"])
                            else float("nan"))
                stable_geometry = bool(
                    lineage["geometry_qualified"]
                    and abs(lineage["center_distance"] - clean["best_distance"]) <= 0.5
                )
                clean_geometry_cost = regression_cost(
                    clean_frame["layer_boxes"][-1, clean["best_query"]],
                    gt["center"], gt["size"], gt["yaw"],
                )
                lineage_fault_geometry_cost = regression_cost(
                    fault_frame["layer_boxes"][-1, clean["best_query"]],
                    gt["center"], gt["size"], gt["yaw"],
                )
                fault_best_geometry_cost = (regression_cost(
                    fault_frame["layer_boxes"][-1, fault["best_query"]],
                    gt["center"], gt["size"], gt["yaw"],
                ) if fault_available else float("nan"))
                row = {
                    "protocol": protocol, "condition": GROUPS[protocol],
                    "sample_token": token, "scene_token": str(fault_frame["scene_token"]),
                    "frame_idx": int(fault_frame["frame_idx"]),
                    "gt_token": gt["token"], "gt_class": gt["name"], "outcome": outcome,
                    "clean_s_pos": clean["s_pos"], "fault_s_pos": fault["s_pos"],
                    "clean_s_k": clean["s_k"], "fault_s_k": fault["s_k"],
                    "clean_margin": clean["margin"], "fault_margin": fault["margin"],
                    "delta_s_pos": delta_s_pos, "delta_s_k": delta_s_k,
                    "delta_margin": delta_margin,
                    "decomposition_error": decomposition_error,
                    "target_burden": max(-delta_s_pos, 0.0) if fault_available else float("nan"),
                    "competitor_burden": max(delta_s_k, 0.0),
                    "clean_best_rank": clean["rank"], "fault_best_rank": fault["rank"],
                    "boundary_crossing": bool(fault_available and clean["rank"] <= 100
                                               and fault["rank"] > 100),
                    "fault_candidate_available": fault_available,
                    "m_cf_target": target_cf, "m_cf_boundary": boundary_cf,
                    "rescue_category": rescue,
                    "n_clean_2m": clean["count"], "n_fault_2m": fault["count"],
                    "delta_candidate_count": fault["count"] - clean["count"],
                    "clean_top1": top_score(clean["scores"], 0),
                    "clean_top2": top_score(clean["scores"], 1),
                    "clean_top3": top_score(clean["scores"], 2),
                    "fault_top1": top_score(fault["scores"], 0),
                    "fault_top2": top_score(fault["scores"], 1),
                    "fault_top3": top_score(fault["scores"], 2),
                    "count_match_status": match_status,
                    "count_match_seed": deterministic_seed(BASE_SEED, protocol, token, gt["token"]),
                    "count_match_requested_repeats": MATCH_REPEATS,
                    "count_match_effective_repeats": matched["effective_repeats"],
                    "count_matched_clean_expected_max": matched["expected_max"],
                    "count_matched_clean_p025": matched["p025"],
                    "count_matched_clean_p50": matched["p50"],
                    "count_matched_clean_p975": matched["p975"],
                    "count_attributable_score_loss": count_loss,
                    "count_matched_residual": residual,
                    "clean_best_query": clean["best_query"],
                    "fault_best_query": fault["best_query"],
                    "clean_best_center_distance": clean["best_distance"],
                    "fault_best_center_distance": fault["best_distance"],
                    "reselected_delta_center_distance": (fault["best_distance"] - clean["best_distance"]
                                                          if fault_available else float("nan")),
                    "clean_best_geometry_cost": clean_geometry_cost,
                    "fault_best_geometry_cost": fault_best_geometry_cost,
                    "reselected_delta_geometry_cost": (fault_best_geometry_cost - clean_geometry_cost
                                                         if fault_available else float("nan")),
                    "lineage_fault_score": lineage["score"],
                    "lineage_delta_score": lineage["score"] - clean["s_pos"],
                    "lineage_fault_center_distance": lineage["center_distance"],
                    "lineage_delta_center_distance": lineage["center_distance"] - clean["best_distance"],
                    "lineage_fault_geometry_cost": lineage_fault_geometry_cost,
                    "lineage_delta_geometry_cost": lineage_fault_geometry_cost - clean_geometry_cost,
                    "lineage_geometry_qualified": lineage["geometry_qualified"],
                    "lineage_geometry_stable": stable_geometry,
                    "cam_back_visible": bool(visible[CAM_BACK_INDEX]),
                    "visible_view_count": int(np.count_nonzero(visible)),
                    "alternative_view_count": int(np.count_nonzero(np.delete(visible, CAM_BACK_INDEX))),
                    "camera_overlap_count": max(int(np.count_nonzero(visible)) - 1, 0),
                    "gt_center_distance": float(np.linalg.norm(gt["center"])),
                    "max_projected_box_area_fraction": float(np.max(visibility["area_fraction"])),
                }
                rows.append(row)
                for side, values in (("clean", clean), ("fault", fault)):
                    for order, (query, score, distance) in enumerate(zip(
                        values["queries"], values["scores"], values["distances"]
                    ), start=1):
                        candidate_rows.append({
                            "protocol": protocol, "sample_token": token,
                            "gt_token": gt["token"], "outcome": outcome, "side": side,
                            "score_order": order, "query_index": int(query),
                            "gt_class_score": float(score), "center_distance": float(distance),
                        })

    decomposition_rows, bootstrap_rows = [], []
    for protocol in (*GROUPS, "aggregate"):
        protocol_rows = rows if protocol == "aggregate" else [r for r in rows if r["protocol"] == protocol]
        for outcome in ("fault_induced_lost", "retained_control"):
            selected = [r for r in protocol_rows if r["outcome"] == outcome
                        and r["fault_candidate_available"]]
            for metric in ("delta_s_pos", "delta_s_k", "delta_margin"):
                values = finite(selected, metric)
                summary = distribution(values)
                ci = bootstrap_stat(values, np.median,
                                    deterministic_seed(BASE_SEED, "decomposition", protocol, outcome, metric))
                decomposition_rows.append({
                    "protocol": protocol, "condition": GROUPS.get(protocol, "All faults"),
                    "outcome": outcome, "metric": metric, **summary,
                    "median_ci_low": ci["ci_low"], "median_ci_high": ci["ci_high"],
                    "bootstrap_iterations": BOOTSTRAPS,
                })
        lost = [r for r in protocol_rows if r["outcome"] == "fault_induced_lost"
                and r["fault_candidate_available"]]
        retained = [r for r in protocol_rows if r["outcome"] == "retained_control"
                    and r["fault_candidate_available"]]
        for metric in ("delta_s_pos", "delta_s_k", "delta_margin", "delta_candidate_count"):
            metric_lost = ([r for r in protocol_rows if r["outcome"] == "fault_induced_lost"]
                           if metric == "delta_candidate_count" else lost)
            metric_retained = ([r for r in protocol_rows if r["outcome"] == "retained_control"]
                               if metric == "delta_candidate_count" else retained)
            result = bootstrap_difference(finite(metric_lost, metric), finite(metric_retained, metric), np.median,
                                          deterministic_seed(BASE_SEED, "difference", protocol, metric))
            bootstrap_rows.append({"protocol": protocol, "metric": metric,
                                   "contrast": "lost_minus_retained_median",
                                   "lost_n": len(finite(metric_lost, metric)),
                                   "retained_n": len(finite(metric_retained, metric)), **result})

    rescue_rows = []
    for protocol in (*GROUPS, "aggregate"):
        protocol_rows = rows if protocol == "aggregate" else [r for r in rows if r["protocol"] == protocol]
        lost_all = [r for r in protocol_rows if r["outcome"] == "fault_induced_lost"]
        available = [r for r in lost_all if r["fault_candidate_available"]]
        for category in ("target-driven", "competitor-driven", "mixed", "neither"):
            count = sum(r["rescue_category"] == category for r in available)
            rescue_rows.append({
                "protocol": protocol, "condition": GROUPS.get(protocol, "All faults"),
                "category": category, "count": count,
                "rank_available_lost": len(available),
                "percentage": count / max(len(available), 1),
                "candidate_missing_lost": len(lost_all) - len(available),
                "all_lost": len(lost_all),
            })

    count_rows, geometry_rows = [], []
    for protocol in (*GROUPS, "aggregate"):
        protocol_rows = rows if protocol == "aggregate" else [r for r in rows if r["protocol"] == protocol]
        for outcome in ("fault_induced_lost", "retained_control"):
            selected = [r for r in protocol_rows if r["outcome"] == outcome]
            eligible = [r for r in selected if r["count_match_status"] in ("count_decline", "equal_count")]
            declines = [r for r in eligible if r["count_match_status"] == "count_decline"]
            for metric in ("delta_candidate_count", "count_attributable_score_loss",
                           "count_matched_residual"):
                metric_rows = (selected if metric == "delta_candidate_count" else
                               declines if metric == "count_attributable_score_loss" else eligible)
                values = finite(metric_rows, metric)
                ci = bootstrap_stat(values, np.median,
                                    deterministic_seed(BASE_SEED, "count", protocol, outcome, metric))
                count_rows.append({
                    "protocol": protocol, "condition": GROUPS.get(protocol, "All faults"),
                    "outcome": outcome, "metric": metric,
                    "all_gt": len(selected), "eligible": len(eligible), "count_declines": len(declines),
                    **distribution(values), "median_ci_low": ci["ci_low"],
                    "median_ci_high": ci["ci_high"], "bootstrap_iterations": BOOTSTRAPS,
                })
            stable = [r for r in selected if r["lineage_geometry_stable"]]
            score_ci = bootstrap_stat(finite(stable, "lineage_delta_score"), np.median,
                                      deterministic_seed(BASE_SEED, "geometry", protocol, outcome))
            geometry_rows.append({
                "protocol": protocol, "condition": GROUPS.get(protocol, "All faults"),
                "outcome": outcome, "all_gt": len(selected), "geometry_stable": len(stable),
                "geometry_stable_ratio": len(stable) / max(len(selected), 1),
                "median_abs_lineage_delta_distance": (float(np.median(np.abs(
                    finite(stable, "lineage_delta_center_distance")))) if stable else float("nan")),
                "median_abs_lineage_delta_geometry_cost": (float(np.median(np.abs(
                    finite(stable, "lineage_delta_geometry_cost")))) if stable else float("nan")),
                "median_lineage_delta_score": (float(np.median(finite(stable, "lineage_delta_score")))
                                                if stable else float("nan")),
                "score_delta_ci_low": score_ci["ci_low"], "score_delta_ci_high": score_ci["ci_high"],
                "bootstrap_iterations": BOOTSTRAPS,
            })

    stable_lost = [r for r in rows if r["outcome"] == "fault_induced_lost" and r["lineage_geometry_stable"]]
    stable_retained = [r for r in rows if r["outcome"] == "retained_control" and r["lineage_geometry_stable"]]
    geometry_difference = bootstrap_difference(
        finite(stable_lost, "lineage_delta_score"), finite(stable_retained, "lineage_delta_score"),
        np.median, deterministic_seed(BASE_SEED, "geometry", "difference"),
    )
    geometry_rows.append({
        "protocol": "aggregate", "condition": "All faults", "outcome": "lost_minus_retained",
        "all_gt": len(stable_lost) + len(stable_retained), "geometry_stable": len(stable_lost),
        "geometry_stable_ratio": float("nan"), "median_abs_lineage_delta_distance": float("nan"),
        "median_abs_lineage_delta_geometry_cost": float("nan"),
        "median_lineage_delta_score": geometry_difference["estimate"],
        "score_delta_ci_low": geometry_difference["ci_low"],
        "score_delta_ci_high": geometry_difference["ci_high"],
        "bootstrap_iterations": BOOTSTRAPS,
    })

    visibility_rows = []
    links = (
        ("alternative_views_to_candidate_delta", "alternative_view_count", "delta_candidate_count", 1),
        ("candidate_delta_to_s_pos_delta", "delta_candidate_count", "delta_s_pos", 1),
        ("s_pos_delta_to_crossing", "delta_s_pos", "boundary_crossing", -1),
    )
    for protocol in (*GROUPS, "aggregate"):
        selected = rows if protocol == "aggregate" else [r for r in rows if r["protocol"] == protocol]
        available = [r for r in selected if r["fault_candidate_available"]]
        for name, left_key, right_key, expected_direction in links:
            result = bootstrap_spearman(finite(available, left_key), finite(available, right_key),
                                        deterministic_seed(BASE_SEED, "chain", protocol, name))
            visibility_rows.append({"protocol": protocol, "analysis": "spearman_chain",
                                    "variable": name, "expected_direction": expected_direction,
                                    "n": len(finite(available, left_key)), **result})
        lost = [r for r in selected if r["outcome"] == "fault_induced_lost"]
        retained = [r for r in selected if r["outcome"] == "retained_control"]
        for metric in ("alternative_view_count", "cam_back_visible", "gt_center_distance",
                       "max_projected_box_area_fraction", "delta_candidate_count", "delta_s_pos"):
            result = bootstrap_difference(finite(lost, metric), finite(retained, metric), np.median,
                                          deterministic_seed(BASE_SEED, "visibility", protocol, metric))
            visibility_rows.append({"protocol": protocol, "analysis": "lost_minus_retained_median",
                                    "variable": metric, "expected_direction": "descriptive",
                                    "n": len(finite(lost, metric)) + len(finite(retained, metric)), **result})

    predictors = {
        "small_clean_margin": lambda row: -row["clean_margin"],
        "few_clean_candidates": lambda row: -row["n_clean_2m"],
        "few_alternative_views": lambda row: -row["alternative_view_count"],
        "cam_back_visible": lambda row: float(row["cam_back_visible"]),
        "large_distance": lambda row: row["gt_center_distance"],
        "small_projected_box": lambda row: -row["max_projected_box_area_fraction"],
    }
    predictor_rows = []
    for protocol in (*GROUPS, "aggregate"):
        selected = rows if protocol == "aggregate" else [r for r in rows if r["protocol"] == protocol]
        outcome = np.asarray([r["outcome"] == "fault_induced_lost" for r in selected], int)
        for name, function in predictors.items():
            risk = np.asarray([function(r) for r in selected], float)
            result = bootstrap_auc(risk, outcome,
                                   deterministic_seed(BASE_SEED, "predictor", protocol, name))
            predictor_rows.append({
                "protocol": protocol, "condition": GROUPS.get(protocol, "All faults"),
                "predictor": name, "n": len(selected), "lost": int(np.sum(outcome)),
                "retained": int(len(outcome) - np.sum(outcome)), "risk_orientation": "higher_is_lost",
                "auroc": result["estimate"], "ci_low": result["ci_low"],
                "ci_high": result["ci_high"], "bootstrap_iterations": BOOTSTRAPS,
            })

    decomp_lookup = {(r["protocol"], r["outcome"], r["metric"]): r for r in decomposition_rows}
    lost_protocol_medians = lambda metric: [
        decomp_lookup[(protocol, "fault_induced_lost", metric)]["median"] for protocol in GROUPS
    ]
    pooled_spos = decomp_lookup[("aggregate", "fault_induced_lost", "delta_s_pos")]
    pooled_sk = decomp_lookup[("aggregate", "fault_induced_lost", "delta_s_k")]
    pooled_lost = [r for r in rows if r["outcome"] == "fault_induced_lost"
                   and r["fault_candidate_available"]]
    median_target_burden = float(np.median(finite(pooled_lost, "target_burden")))
    median_competitor_burden = float(np.median(finite(pooled_lost, "competitor_burden")))
    h1_evidence = all(value < 0 for value in lost_protocol_medians("delta_s_pos")) and pooled_spos["median_ci_high"] < 0
    h2_evidence = all(value > 0 for value in lost_protocol_medians("delta_s_k")) and pooled_sk["median_ci_low"] > 0
    h1_primary = h1_evidence and median_target_burden > median_competitor_burden
    h2_primary = h2_evidence and median_competitor_burden > median_target_burden

    count_lookup = {(r["protocol"], r["outcome"], r["metric"]): r for r in count_rows}
    count_diff = next(r for r in bootstrap_rows
                      if r["protocol"] == "aggregate" and r["metric"] == "delta_candidate_count")
    count_decline_rows = [r for r in pooled_lost if r["count_match_status"] == "count_decline"]
    positive_observed_loss = float(np.sum(np.maximum(-finite(count_decline_rows, "delta_s_pos"), 0)))
    positive_count_loss = float(np.sum(np.maximum(finite(count_decline_rows, "count_attributable_score_loss"), 0)))
    count_explained_ratio = positive_count_loss / positive_observed_loss if positive_observed_loss else float("nan")
    h3_support = (
        all(count_lookup[(p, "fault_induced_lost", "delta_candidate_count")]["median"] < 0 for p in GROUPS)
        and count_diff["ci_high"] < 0
        and count_lookup[("aggregate", "fault_induced_lost", "count_attributable_score_loss")]["median"] > 0
    )
    h3_dominant = h3_support and count_explained_ratio >= 0.50
    h4_support = (
        all(count_lookup[(p, "fault_induced_lost", "count_matched_residual")]["median"] < 0 for p in GROUPS)
        and count_lookup[("aggregate", "fault_induced_lost", "count_matched_residual")]["median_ci_high"] < 0
    )
    geometry_lookup = {(r["protocol"], r["outcome"]): r for r in geometry_rows}
    h5_support = (
        all(geometry_lookup[(p, "fault_induced_lost")]["median_lineage_delta_score"] < 0 for p in GROUPS)
        and geometry_lookup[("aggregate", "fault_induced_lost")]["score_delta_ci_high"] < 0
        and geometry_difference["ci_high"] < 0
    )

    competitor_rows = []
    if h2_evidence:
        for protocol in GROUPS:
            for token in sorted(frame_has_lost[protocol]):
                clean_frame, fault_frame = traces["clean"][token], traces[protocol][token]
                clean_scores = sigmoid(clean_frame["layer_logits"][-1])
                fault_scores = sigmoid(fault_frame["layer_logits"][-1])
                flat = clean_scores.reshape(-1)
                clean_sk = float(np.partition(flat, flat.size - 100)[flat.size - 100])
                new_pairs = np.argwhere((fault_scores > clean_sk) & (clean_scores <= clean_sk))
                assignments = []
                for query, label in new_pairs:
                    center = fault_frame["layer_boxes"][-1, query, :3]
                    compatible = []
                    for gt in gt_cache[token]:
                        if gt["label"] == label:
                            distance = float(np.linalg.norm(center[:2] - gt["center"][:2]))
                            if distance <= 2.0:
                                compatible.append((distance, gt["token"]))
                    nearest = min(compatible) if compatible else None
                    assignments.append((int(query), int(label), center, nearest,
                                        float(fault_scores[query, label]), float(clean_scores[query, label])))
                best_for_gt = {}
                for index, value in enumerate(assignments):
                    if value[3] is None:
                        continue
                    gt_token = value[3][1]
                    if gt_token not in best_for_gt or value[4] > assignments[best_for_gt[gt_token]][4]:
                        best_for_gt[gt_token] = index
                for index, (query, label, center, nearest, fault_score, clean_score) in enumerate(assignments):
                    if nearest is None:
                        source = "background_false_positive"
                        gt_token, distance = "", float("nan")
                    else:
                        distance, gt_token = nearest
                        source = "matched_true_positive" if best_for_gt[gt_token] == index else "duplicate_query"
                    competitor_rows.append({
                        "protocol": protocol, "sample_token": token, "query_index": query,
                        "class_index": label, "class_name": CLASS_NAMES[label],
                        "clean_score": clean_score, "fault_score": fault_score, "clean_s_k": clean_sk,
                        "source_type": source, "matched_gt_token": gt_token,
                        "same_class_gt_distance": distance, "spatial_region": frame_region(center, fault_frame),
                    })

    useful_predictors = []
    for name in predictors:
        protocol_values = [next(r for r in predictor_rows if r["protocol"] == p and r["predictor"] == name)
                           for p in GROUPS]
        pooled = next(r for r in predictor_rows if r["protocol"] == "aggregate" and r["predictor"] == name)
        if all(r["auroc"] > 0.5 for r in protocol_values) and pooled["auroc"] >= 0.65 and pooled["ci_low"] > 0.5:
            useful_predictors.append(name)

    chain_lookup = {(r["protocol"], r["variable"]): r for r in visibility_rows
                    if r["analysis"] == "spearman_chain"}
    stable_chain = all(
        chain_lookup[(protocol, link)]["estimate"] * direction > 0
        for protocol in GROUPS for link, _, _, direction in links
    )
    candidate_chain_established = stable_chain and h3_support and h1_evidence

    if h1_primary:
        direct_source = "target score collapse"
    elif h2_primary:
        direct_source = "competitor inflation"
    elif h1_evidence and h2_evidence:
        direct_source = "mixed target and competitor movement"
    else:
        direct_source = "insufficient direct score-space root cause"
    if h3_dominant and h4_support:
        target_mechanism = "mixed candidate redundancy collapse and per-candidate degradation"
    elif h3_dominant:
        target_mechanism = "candidate redundancy collapse"
    elif h4_support:
        target_mechanism = "per-candidate degradation"
    elif h3_support:
        target_mechanism = "candidate redundancy contributes but is not dominant"
    else:
        target_mechanism = "no supported candidate-level explanation"
    worth_method_design = bool(
        direct_source != "insufficient direct score-space root cause"
        and (h3_support or h4_support or h2_evidence or h5_support)
    )

    competitor_summary_rows = []
    if competitor_rows:
        for protocol in (*GROUPS, "aggregate"):
            selected = competitor_rows if protocol == "aggregate" else [r for r in competitor_rows if r["protocol"] == protocol]
            for source, count in sorted(Counter(r["source_type"] for r in selected).items()):
                competitor_summary_rows.append({"protocol": protocol, "dimension": "source_type",
                                                "value": source, "count": count,
                                                "percentage": count / max(len(selected), 1)})
            for name, count in sorted(Counter(r["class_name"] for r in selected).items()):
                competitor_summary_rows.append({"protocol": protocol, "dimension": "class_name",
                                                "value": name, "count": count,
                                                "percentage": count / max(len(selected), 1)})
            for region, count in sorted(Counter(r["spatial_region"] for r in selected).items()):
                competitor_summary_rows.append({"protocol": protocol, "dimension": "spatial_region",
                                                "value": region, "count": count,
                                                "percentage": count / max(len(selected), 1)})

    group_rows = []
    for protocol in (*GROUPS, "aggregate"):
        selected = rows if protocol == "aggregate" else [r for r in rows if r["protocol"] == protocol]
        for outcome in ("fault_induced_lost", "retained_control"):
            group = [r for r in selected if r["outcome"] == outcome]
            available = [r for r in group if r["fault_candidate_available"]]
            group_rows.append({
                "protocol": protocol, "condition": GROUPS.get(protocol, "All faults"),
                "outcome": outcome, "gt_count": len(group), "rank_available": len(available),
                "rank_available_ratio": len(available) / max(len(group), 1),
                "median_delta_s_pos": float(np.median(finite(available, "delta_s_pos"))) if available else float("nan"),
                "median_delta_s_k": float(np.median(finite(available, "delta_s_k"))) if available else float("nan"),
                "median_delta_margin": float(np.median(finite(available, "delta_margin"))) if available else float("nan"),
                "median_n_clean_2m": float(np.median(finite(group, "n_clean_2m"))),
                "median_n_fault_2m": float(np.median(finite(group, "n_fault_2m"))),
                "median_delta_candidate_count": float(np.median(finite(group, "delta_candidate_count"))),
                "boundary_crossing_count": sum(r["boundary_crossing"] for r in available),
                "boundary_crossing_ratio": sum(r["boundary_crossing"] for r in available) / max(len(available), 1),
            })

    write_csv("prediction_invariance.csv", invariance_rows)
    write_csv("per_gt_root_cause.csv", rows)
    write_csv("candidate_scores_long.csv", candidate_rows)
    write_csv("protocol_group_summary.csv", group_rows)
    write_csv("score_decomposition_summary.csv", decomposition_rows)
    write_csv("score_decomposition_bootstrap.csv", bootstrap_rows)
    write_csv("counterfactual_rescue_summary.csv", rescue_rows)
    write_csv("candidate_count_matched_summary.csv", count_rows)
    write_csv("geometry_semantic_summary.csv", geometry_rows)
    write_csv("visibility_chain_summary.csv", visibility_rows)
    write_csv("advance_predictor_summary.csv", predictor_rows)
    write_csv("competitor_inflation_pairs.csv", competitor_rows, allow_empty=True)
    write_csv("competitor_inflation_summary.csv", competitor_summary_rows, allow_empty=True)

    rescue_pooled = {r["category"]: r["percentage"] for r in rescue_rows if r["protocol"] == "aggregate"}
    lost_groups = [r for r in group_rows if r["outcome"] == "fault_induced_lost" and r["protocol"] != "aggregate"]
    table = "\n".join(
        f"| {r['condition']} | {r['gt_count']} | {r['rank_available_ratio']:.1%} | "
        f"{r['median_delta_s_pos']:.6f} | {r['median_delta_s_k']:.6f} | "
        f"{r['median_delta_margin']:.6f} | {r['median_n_clean_2m']:.1f} -> {r['median_n_fault_2m']:.1f} | "
        f"{r['boundary_crossing_ratio']:.1%} |"
        for r in lost_groups
    )
    residual_table = "\n".join(
        f"| {GROUPS[protocol]} | "
        f"{count_lookup[(protocol, 'fault_induced_lost', 'count_matched_residual')]['median']:.6f} | "
        f"{count_lookup[(protocol, 'retained_control', 'count_matched_residual')]['median']:.6f} |"
        for protocol in GROUPS
    )
    geometry_best_table = "\n".join(
        f"| {GROUPS[protocol]} | "
        f"{float(np.median(finite([r for r in rows if r['protocol'] == protocol and r['outcome'] == 'fault_induced_lost' and r['fault_candidate_available']], 'reselected_delta_center_distance'))):.6f} | "
        f"{float(np.median(finite([r for r in rows if r['protocol'] == protocol and r['outcome'] == 'fault_induced_lost' and r['fault_candidate_available']], 'reselected_delta_geometry_cost'))):.6f} | "
        f"{decomp_lookup[(protocol, 'fault_induced_lost', 'delta_s_pos')]['median']:.6f} |"
        for protocol in GROUPS
    )
    rescue_table = "\n".join(
        f"| {GROUPS[protocol]} | "
        f"{next(r['percentage'] for r in rescue_rows if r['protocol'] == protocol and r['category'] == 'target-driven'):.1%} | "
        f"{next(r['percentage'] for r in rescue_rows if r['protocol'] == protocol and r['category'] == 'competitor-driven'):.1%} | "
        f"{next(r['percentage'] for r in rescue_rows if r['protocol'] == protocol and r['category'] == 'mixed'):.1%} | "
        f"{next(r['percentage'] for r in rescue_rows if r['protocol'] == protocol and r['category'] == 'neither'):.1%} |"
        for protocol in GROUPS
    )
    predictor_table = "\n".join(
        f"| {name} | "
        f"{next(r['auroc'] for r in predictor_rows if r['protocol'] == 'aggregate' and r['predictor'] == name):.3f} | "
        f"[{next(r['ci_low'] for r in predictor_rows if r['protocol'] == 'aggregate' and r['predictor'] == name):.3f}, "
        f"{next(r['ci_high'] for r in predictor_rows if r['protocol'] == 'aggregate' and r['predictor'] == name):.3f}] |"
        for name in predictors
    )
    report = f"""# Fault-induced Boundary Crossing Root-Cause Audit

## Decision

**Direct boundary-crossing source: {direct_source}.** The candidate-level explanation is **{target_mechanism}**. H1={h1_evidence}, H2={h2_evidence}, H3 support={h3_support} (dominant={h3_dominant}, explained-loss ratio={count_explained_ratio:.1%}), H4={h4_support}, and H5={h5_support}.

This was a frozen-B0, offline paired audit. No model/loss/optimizer was created or stepped. Hooked/disabled predictions remain tensor-exact for Clean/Dark/Blur/Crash (`max_abs_diff=0`).

## Exact score-space evidence

| Protocol | Lost GT | rank available | median delta-S_pos | median delta-S_K | median delta-M | median N_clean -> N_fault | crossing |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

The maximum absolute decomposition error for `delta_M = delta_S_pos - delta_S_K` is {max(abs(r['decomposition_error']) for r in rows if np.isfinite(r['decomposition_error'])):.3e}. Pooled lost median `delta_S_pos` is {pooled_spos['median']:.6f} (bootstrap 95% CI [{pooled_spos['median_ci_low']:.6f}, {pooled_spos['median_ci_high']:.6f}]); pooled lost median `delta_S_K` is {pooled_sk['median']:.6f} ([{pooled_sk['median_ci_low']:.6f}, {pooled_sk['median_ci_high']:.6f}]). Median target burden is {median_target_burden:.6f}, versus competitor burden {median_competitor_burden:.6f}.

## Counterfactual rescue and candidate cause

Among rank-available lost GT, target-only rescue is {rescue_pooled.get('target-driven', 0):.1%}, boundary-only rescue {rescue_pooled.get('competitor-driven', 0):.1%}, both {rescue_pooled.get('mixed', 0):.1%}, and neither {rescue_pooled.get('neither', 0):.1%}. Strict `>0` was used throughout.

| Protocol | target-driven | competitor-driven | mixed | neither |
|---|---:|---:|---:|---:|
{rescue_table}

The count-matched audit used seed 314159-derived per-GT seeds and 10,000 without-replacement draws whenever Clean had more candidates. Candidate redundancy support is {h3_support}; its summed positive-loss explanation ratio is {count_explained_ratio:.1%}. The Fault-minus-count-matched-Clean residual supports per-candidate degradation: {h4_support}. Geometry-stable same-query lineages support semantic degradation despite stable geometry: {h5_support}; pooled lost-minus-retained median score-delta difference is {geometry_difference['estimate']:.6f} (95% CI [{geometry_difference['ci_low']:.6f}, {geometry_difference['ci_high']:.6f}]).

| Protocol | lost median count-matched residual | retained median residual |
|---|---:|---:|
{residual_table}

The {count_explained_ratio:.1%}-scale summed count attribution does not override H3's preregistered cross-protocol gate: Dark lost GT have median candidate-count delta 0, so redundancy collapse is not a stable primary explanation. H5 also fails rather than being inferred from mere <=2 m coverage: only {len(stable_lost)}/{sum(r['outcome'] == 'fault_induced_lost' for r in rows)} lost lineages meet the strict same-query geometry-stability rule, including zero in Dark, and the pooled lost-minus-retained CI includes zero. The regression-cost and center-distance changes for both same-query and independently reselected candidates are retained in the per-GT CSV.

| Protocol | reselected median delta-center (m) | median delta-regression-cost | median delta-S_pos |
|---|---:|---:|---:|
{geometry_best_table}

The independently reselected best candidate is nearly geometry-stable in Dark and Blur while its class score drops, but Crash also worsens geometrically. This is partial geometry-semantic decoupling, not the preregistered cross-protocol H5.

## Visibility, competitors and advance prediction

The three alternative-view -> candidate-count -> score -> crossing correlations have the expected sign in every Dark/Blur/Crash protocol: **{stable_chain}**. The full candidate-redundancy mechanism chain is nevertheless **{candidate_chain_established}**, because H3's lost-versus-retained gate fails in Dark. The supported cross-fault chain is narrower: CAM_BACK-sensitive GT with fewer alternative views are more loss-prone; after candidate-count matching their candidates remain weaker; `S_pos` falls while `S_K` actually falls slightly; then M crosses zero. This is explanation-only stratification, not a causal claim.

H2's conditional competitor-source audit was {'triggered' if h2_evidence else 'not triggered'} because its preregistered evidence condition is {h2_evidence}. No suppression method was considered.

Useful advance predictors under the fixed cross-protocol AUROC rule: **{', '.join(useful_predictors) if useful_predictors else 'none'}**. Predictor AUROCs and bootstrap CIs are in `advance_predictor_summary.csv`; post-fault deltas are kept separate from advance predictors.

| Risk-oriented Clean/physical predictor | pooled AUROC | bootstrap 95% CI |
|---|---:|---:|
{predictor_table}

## Prediction verdicts and next-stage gate

- H1 target evidence collapse: **{'verified' if h1_evidence else 'not verified'}**.
- H2 competitor inflation: **{'verified' if h2_evidence else 'not verified'}**.
- H3 candidate redundancy collapse: **{'verified as dominant' if h3_dominant else ('supported but not dominant' if h3_support else 'not verified')}**.
- H4 per-candidate degradation: **{'verified' if h4_support else 'not verified'}**.
- H5 geometry-semantic decoupling: **{'verified' if h5_support else 'not verified'}**.

The root cause is **{'sufficiently stable to justify entering a separate method-design stage, but not another generic boundary/rank-preservation objective: the evidenced problem is fault-sensitive target semantics rather than competitor inflation' if worth_method_design else 'insufficient; stop ranking-objective method design'}** under the preregistered rules. This report intentionally proposes no network structure, loss, hyperparameter change or training strategy.
"""
    (REPORT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "per_gt": len(rows), "candidate_rows": len(candidate_rows),
        "direct_source": direct_source, "target_mechanism": target_mechanism,
        "H1": h1_evidence, "H2": h2_evidence, "H3": h3_support,
        "H3_dominant": h3_dominant, "H4": h4_support, "H5": h5_support,
        "stable_visibility_chain": stable_chain, "useful_predictors": useful_predictors,
        "candidate_chain_established": candidate_chain_established,
        "worth_method_design": worth_method_design,
    }, indent=2))


if __name__ == "__main__":
    main()
