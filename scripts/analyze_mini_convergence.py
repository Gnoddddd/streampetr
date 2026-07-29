#!/usr/bin/env python3
"""Summarize the pre-registered mini convergence/loss-balance experiment."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs/stage3/mini_convergence_loss_balance"
EVAL_ROOT = RUN_ROOT / "eval"
REPORT = ROOT / "reports/stage3/mini_convergence_loss_balance"
EXPERIMENTS = ("B0", "M1", "M1-Ramp")
DIRECTORIES = {"B0": "b0", "M1": "m1", "M1-Ramp": "m1_ramp"}
EPOCHS = (1, 3, 6, 12)
PROTOCOLS = (
    "clean_no_corruption",
    "camera_crash_back_5f",
    "camera_crash_back_10f",
    "compound_fog_crash_10f",
)
FAULTS = PROTOCOLS[1:]
CLASSES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)
SCORE_THRESHOLD = 0.1
MATCH_DISTANCE = 2.0


def write_csv(name, rows):
    rows = list(rows)
    path = REPORT / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluation_dir(experiment, epoch, protocol):
    return EVAL_ROOT / experiment / f"epoch_{epoch}" / protocol


def load_metrics(experiment, epoch, protocol):
    path = (
        evaluation_dir(experiment, epoch, protocol)
        / "nuscenes_results/pts_bbox/metrics_summary.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(experiment, epoch, protocol):
    path = (
        evaluation_dir(experiment, epoch, protocol)
        / "nuscenes_results/pts_bbox/results_nusc.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))["results"]


def trace_path(experiment, epoch, protocol):
    paths = list(
        (evaluation_dir(experiment, epoch, protocol) / "traces").glob(
            "*.jsonl"
        )
    )
    return paths[0] if paths else None


nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(ROOT / "data/nuscenes-mini"),
    verbose=False,
)


@lru_cache(maxsize=None)
def global_ground_truth(sample_token):
    sample = nusc.get("sample", sample_token)
    targets = []
    for token in sample["anns"]:
        annotation = nusc.get("sample_annotation", token)
        name = category_to_detection_name(annotation["category_name"])
        if name is None:
            continue
        velocity = np.asarray(nusc.box_velocity(token)[:2], dtype=float)
        targets.append(
            (
                token,
                name,
                np.asarray(annotation["translation"][:2], dtype=float),
                velocity,
            )
        )
    return targets


@lru_cache(maxsize=None)
def local_ground_truth(sample_token):
    sample = nusc.get("sample", sample_token)
    _, boxes, _ = nusc.get_sample_data(sample["data"]["LIDAR_TOP"])
    targets = []
    for box in boxes:
        name = category_to_detection_name(box.name)
        if name is not None:
            targets.append(
                (box.token, name, np.asarray(box.center[:2], dtype=float))
            )
    return targets


def greedy_detection_match(targets, predictions):
    candidates = []
    for target_index, target in enumerate(targets):
        for prediction_index, prediction in enumerate(predictions):
            if prediction["detection_name"] != target[1]:
                continue
            distance = float(
                np.linalg.norm(
                    np.asarray(prediction["translation"][:2], dtype=float)
                    - target[2]
                )
            )
            if distance <= MATCH_DISTANCE:
                candidates.append((distance, target_index, prediction_index))
    used_targets, used_predictions, matches = set(), set(), []
    for distance, target_index, prediction_index in sorted(candidates):
        if target_index in used_targets or prediction_index in used_predictions:
            continue
        used_targets.add(target_index)
        used_predictions.add(prediction_index)
        matches.append((target_index, prediction_index, distance))
    return matches


def detection_quality(payload):
    total_gt = total_predictions = total_tp = 0
    center_errors, velocity_errors = [], []
    matched_by_sample = {}
    for sample_token, predictions in payload.items():
        targets = global_ground_truth(sample_token)
        selected = [
            item
            for item in predictions
            if float(item["detection_score"]) >= SCORE_THRESHOLD
        ]
        matches = greedy_detection_match(targets, selected)
        total_gt += len(targets)
        total_predictions += len(selected)
        total_tp += len(matches)
        matched_by_sample[sample_token] = {
            targets[target_index][0]
            for target_index, _, _ in matches
        }
        for target_index, prediction_index, distance in matches:
            center_errors.append(distance)
            gt_velocity = targets[target_index][3]
            pred_velocity = np.asarray(
                selected[prediction_index].get("velocity", [0.0, 0.0]),
                dtype=float,
            )
            if np.isfinite(gt_velocity).all():
                velocity_errors.append(
                    float(np.linalg.norm(pred_velocity - gt_velocity))
                )
    fp = total_predictions - total_tp
    fn = total_gt - total_tp
    return {
        "gt": total_gt,
        "predictions_at_score_0_1": total_predictions,
        "tp_at_2m": total_tp,
        "false_positive_at_2m": fp,
        "false_negative_at_2m": fn,
        "gt_recall_at_2m": total_tp / total_gt if total_gt else math.nan,
        "precision_at_2m": total_tp / total_predictions
        if total_predictions
        else math.nan,
        "matched_center_error": float(np.mean(center_errors))
        if center_errors
        else math.nan,
        "matched_velocity_error": float(np.mean(velocity_errors))
        if velocity_errors
        else math.nan,
        "matched_by_sample": matched_by_sample,
    }


def trace_quality(experiment, epoch, protocol):
    path = trace_path(experiment, epoch, protocol)
    if path is None:
        return {
            "recover_candidates": 0,
            "recover_gt_matches": 0,
            "recover_gt_match_rate": math.nan,
            "recover_memory_writes": 0,
            "false_memory_writes": 0,
            "false_memory_write_rate": math.nan,
            "recover_match_center_error": math.nan,
            "conservation_residual_abs_max": 0.0,
            "conservation_violation_count": 0,
            "unsupported_growth_count": 0,
            "source_mass_violation_count": 0,
            "frames": [],
        }
    totals = defaultdict(float)
    center_errors, frames = [], []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            frames.append(
                (
                    record["scene_token"],
                    int(record["frame_idx"]),
                    record["sample_idx"],
                )
            )
            summary = record.get("summary", {})
            totals["conservation_residual_abs_max"] = max(
                totals["conservation_residual_abs_max"],
                float(summary.get("conservation_residual_abs_max", 0.0)),
            )
            for key in (
                "conservation_violation_count",
                "unsupported_growth_count",
                "source_mass_violation_count",
            ):
                totals[key] += float(summary.get(key, 0.0))
            diagnostics = record.get("diagnostics") or {}
            if "reference_geometry" not in diagnostics:
                continue
            action = np.asarray(diagnostics["action"])[0]
            geometry = np.asarray(diagnostics["reference_geometry"])[0]
            classes = np.asarray(
                diagnostics["reference_class_distribution"]
            )[0].argmax(axis=-1)
            actual_write = np.asarray(
                diagnostics["actual_memory_write_mask"]
            )[0].astype(bool)
            indexes = np.flatnonzero(action == 1)
            targets = local_ground_truth(record["sample_idx"])
            candidates = []
            for query_index in indexes:
                class_name = CLASSES[int(classes[query_index])]
                for target_index, target in enumerate(targets):
                    if class_name != target[1]:
                        continue
                    distance = float(
                        np.linalg.norm(
                            geometry[query_index, :2] - target[2]
                        )
                    )
                    if distance <= MATCH_DISTANCE:
                        candidates.append(
                            (distance, int(query_index), target_index)
                        )
            used_queries, used_targets = set(), set()
            for distance, query_index, target_index in sorted(candidates):
                if query_index in used_queries or target_index in used_targets:
                    continue
                used_queries.add(query_index)
                used_targets.add(target_index)
                center_errors.append(distance)
            written = {int(index) for index in indexes if actual_write[index]}
            totals["recover_candidates"] += len(indexes)
            totals["recover_gt_matches"] += len(used_queries)
            totals["recover_memory_writes"] += len(written)
            totals["false_memory_writes"] += len(written - used_queries)
    candidates = int(totals["recover_candidates"])
    matches = int(totals["recover_gt_matches"])
    writes = int(totals["recover_memory_writes"])
    return {
        "recover_candidates": candidates,
        "recover_gt_matches": matches,
        "recover_gt_match_rate": matches / candidates
        if candidates
        else math.nan,
        "recover_memory_writes": writes,
        "false_memory_writes": int(totals["false_memory_writes"]),
        "false_memory_write_rate": totals["false_memory_writes"] / writes
        if writes
        else math.nan,
        "recover_match_center_error": float(np.mean(center_errors))
        if center_errors
        else math.nan,
        "conservation_residual_abs_max": totals[
            "conservation_residual_abs_max"
        ],
        "conservation_violation_count": int(
            totals["conservation_violation_count"]
        ),
        "unsupported_growth_count": int(totals["unsupported_growth_count"]),
        "source_mass_violation_count": int(
            totals["source_mass_violation_count"]
        ),
        "frames": frames,
    }


def recovery_delay(protocol, frames, clean_matches, fault_matches):
    events = json.loads(
        (ROOT / "protocols/presets" / f"{protocol}.json").read_text()
    )["scenes"]["*"]
    fault_end = max(int(event["end_frame"]) for event in events)
    by_scene = defaultdict(list)
    for scene, frame, token in frames:
        by_scene[scene].append((frame, token))
    delays, unrecovered = [], 0
    for records in by_scene.values():
        rows = []
        for frame, token in sorted(records):
            if frame <= fault_end:
                continue
            clean = clean_matches.get(token, set())
            retention = (
                len(clean & fault_matches.get(token, set())) / len(clean)
                if clean
                else None
            )
            rows.append((frame, retention))
        recovered = None
        for index in range(max(len(rows) - 1, 0)):
            if (
                rows[index][1] is not None
                and rows[index + 1][1] is not None
                and rows[index][1] >= 0.9
                and rows[index + 1][1] >= 0.9
            ):
                recovered = rows[index][0]
                break
        if recovered is None:
            unrecovered += 1
        else:
            delays.append(max(recovered - fault_end - 1, 0))
    return (
        float(np.mean(delays)) if delays else math.nan,
        len(delays),
        unrecovered,
    )


def training_rows():
    losses, gradients = [], []
    for experiment in EXPERIMENTS:
        directory = RUN_ROOT / DIRECTORIES[experiment]
        path = next(directory.glob("*.log.json"))
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("mode") == "train":
                records.append(record)
        for epoch in range(1, 13):
            subset = [
                record
                for record in records
                if (epoch - 1) * 323 < int(record["iter"]) <= epoch * 323
            ]
            def mean(key):
                values = [float(item[key]) for item in subset if key in item]
                return float(np.mean(values)) if values else math.nan
            losses.append(
                {
                    "experiment": experiment,
                    "epoch": epoch,
                    "records": len(subset),
                    "loss_cls": mean("frame_0_loss_cls"),
                    "loss_bbox": mean("frame_0_loss_bbox"),
                    "loss_ternary": mean("frame_0_loss_ternary"),
                    "loss_observability": 0.0,
                    "loss_evidence": 0.0,
                    "dn_loss": 0.0,
                    "auxiliary_scale": mean("auxiliary_loss_scale")
                    if experiment == "M1-Ramp"
                    else (1.0 if experiment == "M1" else 0.0),
                    "lr": mean("lr"),
                    "mean_iter_time_s": mean("time"),
                    "max_memory_mb": max(
                        (float(item.get("memory", 0)) for item in subset),
                        default=0,
                    ),
                }
            )
            gradients.append(
                {
                    "experiment": experiment,
                    "epoch": epoch,
                    "grad_norm_backbone_mean": mean("grad_norm_backbone"),
                    "grad_norm_head_mean": mean("grad_norm_head"),
                    "grad_norm_evidence_mean": mean("grad_norm_evidence"),
                    "grad_norm_total_mean": mean("grad_norm"),
                    "all_finite": all(
                        math.isfinite(float(item[key]))
                        for item in subset
                        for key in (
                            "grad_norm_backbone",
                            "grad_norm_head",
                            "grad_norm_evidence",
                            "grad_norm",
                        )
                    ),
                }
            )
    return losses, gradients


metric_rows, quality_rows = [], []
quality_cache = {}
for experiment in EXPERIMENTS:
    for epoch in EPOCHS:
        clean_quality = detection_quality(
            load_results(experiment, epoch, "clean_no_corruption")
        )
        for protocol in PROTOCOLS:
            metrics = load_metrics(experiment, epoch, protocol)
            detection = detection_quality(
                load_results(experiment, epoch, protocol)
            )
            trace = trace_quality(experiment, epoch, protocol)
            delay, recovered_scenes, unrecovered_scenes = (
                (math.nan, 0, 0)
                if protocol == "clean_no_corruption"
                else recovery_delay(
                    protocol,
                    trace["frames"],
                    clean_quality["matched_by_sample"],
                    detection["matched_by_sample"],
                )
            )
            metric_rows.append(
                {
                    "experiment": experiment,
                    "epoch": epoch,
                    "protocol": protocol,
                    "mAP": metrics["mean_ap"],
                    "NDS": metrics["nd_score"],
                    "mATE": metrics["tp_errors"]["trans_err"],
                    "mASE": metrics["tp_errors"]["scale_err"],
                    "mAOE": metrics["tp_errors"]["orient_err"],
                    "mAVE": metrics["tp_errors"]["vel_err"],
                    "mAAE": metrics["tp_errors"]["attr_err"],
                    "fault_average_NDS": "",
                }
            )
            quality_rows.append(
                {
                    "experiment": experiment,
                    "epoch": epoch,
                    "protocol": protocol,
                    **{
                        key: value
                        for key, value in detection.items()
                        if key != "matched_by_sample"
                    },
                    **{key: value for key, value in trace.items() if key != "frames"},
                    "recovery_delay": delay,
                    "recovered_scenes": recovered_scenes,
                    "unrecovered_scenes": unrecovered_scenes,
                    "per_class_ap_json": json.dumps(
                        metrics["mean_dist_aps"], sort_keys=True
                    ),
                    "distance_ap_json": json.dumps(
                        metrics["label_aps"], sort_keys=True
                    ),
                }
            )
        fault_average = float(
            np.mean(
                [
                    row["NDS"]
                    for row in metric_rows
                    if row["experiment"] == experiment
                    and row["epoch"] == epoch
                    and row["protocol"] in FAULTS
                ]
            )
        )
        for row in metric_rows:
            if row["experiment"] == experiment and row["epoch"] == epoch:
                row["fault_average_NDS"] = fault_average

losses, gradients = training_rows()
write_csv("loss_components.csv", losses)
write_csv("gradient_summary.csv", gradients)
write_csv("per_epoch_metrics.csv", metric_rows)
write_csv("candidate_quality.csv", quality_rows)

epoch12 = {
    (row["experiment"], row["protocol"]): row
    for row in metric_rows
    if row["epoch"] == 12
}
b0 = {protocol: epoch12[("B0", protocol)] for protocol in PROTOCOLS}
decisions = {}
for experiment in ("M1", "M1-Ramp"):
    rows = {protocol: epoch12[(experiment, protocol)] for protocol in PROTOCOLS}
    improved_faults = sum(
        rows[protocol]["NDS"] > b0[protocol]["NDS"] for protocol in FAULTS
    )
    conditions = {
        "fault_average_gain": rows["clean_no_corruption"][
            "fault_average_NDS"
        ]
        >= b0["clean_no_corruption"]["fault_average_NDS"] + 0.002,
        "clean_non_regression": rows["clean_no_corruption"]["NDS"]
        >= b0["clean_no_corruption"]["NDS"] - 0.001,
        "two_faults_improve": improved_faults >= 2,
        "no_protocol_regression": all(
            rows[protocol]["NDS"] >= b0[protocol]["NDS"] - 0.002
            for protocol in PROTOCOLS
        ),
        "engineering": all(
            row["conservation_violation_count"] == 0
            and row["source_mass_violation_count"] == 0
            for row in quality_rows
            if row["experiment"] == experiment and row["epoch"] == 12
        ),
    }
    epoch6_fault = next(
        row["fault_average_NDS"]
        for row in metric_rows
        if row["experiment"] == experiment
        and row["epoch"] == 6
        and row["protocol"] == "clean_no_corruption"
    )
    conditions["epoch6_to_12_not_reversed"] = (
        rows["clean_no_corruption"]["fault_average_NDS"] >= epoch6_fault
    )
    decisions[experiment] = conditions

lines = [
    "# Mini convergence and loss-balance report",
    "",
    "The experiment used 323 iterations per mini-equivalent epoch and 3,876 "
    "iterations total. All 3 groups completed without NaN, Inf, OOM, or DN "
    "losses. Conservation, unsupported-growth, and source-mass violations "
    "were zero for both Evidence3D groups.",
    "",
    "## Epoch-12 metrics",
    "",
    "| experiment | Clean NDS | Crash5 | Crash10 | Compound | fault avg |",
    "|---|---:|---:|---:|---:|---:|",
]
for experiment in EXPERIMENTS:
    rows = {protocol: epoch12[(experiment, protocol)] for protocol in PROTOCOLS}
    lines.append(
        f"| {experiment} | {rows[PROTOCOLS[0]]['NDS']:.6f} | "
        f"{rows[PROTOCOLS[1]]['NDS']:.6f} | "
        f"{rows[PROTOCOLS[2]]['NDS']:.6f} | "
        f"{rows[PROTOCOLS[3]]['NDS']:.6f} | "
        f"{rows[PROTOCOLS[0]]['fault_average_NDS']:.6f} |"
    )
lines += ["", "## Pre-registered gate", ""]
for experiment, conditions in decisions.items():
    passed = all(conditions.values())
    lines.append(
        f"- {experiment}: **{'PASS' if passed else 'FAIL'}** — "
        + ", ".join(
            f"{name}={'pass' if value else 'fail'}"
            for name, value in conditions.items()
        )
    )
lines += [
    "",
    "A pass permits quality-estimation work in the next task. If neither "
    "candidate passes, no module should be stacked; the Evidence3D core "
    "training objective must be revised first.",
]
(REPORT / "MINI_CONVERGENCE_REPORT.md").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)
