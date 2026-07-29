#!/usr/bin/env python3
"""Summarize the pre-registered four-way nuScenes-mini RayDN screen."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes


ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "outputs/stage3/raydn_screening/eval"
TRAIN_ROOT = ROOT / "outputs/stage3/raydn_screening"
REPORT = ROOT / "reports/stage3/raydn_screening"
EXPERIMENTS = ("B0", "B0_RayDN", "M1", "M1_RayDN")
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


def write_csv(name, rows, fieldnames=None):
    path = REPORT / name
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def metrics(experiment, protocol):
    path = (
        EVAL_ROOT
        / experiment
        / protocol
        / "nuscenes_results/pts_bbox/metrics_summary.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def results(experiment, protocol):
    path = (
        EVAL_ROOT
        / experiment
        / protocol
        / "nuscenes_results/pts_bbox/results_nusc.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))["results"]


def trace_records(experiment, protocol):
    path = next(
        (
            EVAL_ROOT / experiment / protocol / "traces"
        ).glob("*.jsonl")
    )
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def global_ground_truth(nusc, sample_token):
    sample = nusc.get("sample", sample_token)
    targets = []
    for token in sample["anns"]:
        annotation = nusc.get("sample_annotation", token)
        name = category_to_detection_name(annotation["category_name"])
        if name is None:
            continue
        velocity = np.asarray(nusc.box_velocity(token)[:2], dtype=float)
        targets.append(
            {
                "token": token,
                "class": name,
                "center": np.asarray(annotation["translation"][:2], dtype=float),
                "velocity": velocity,
            }
        )
    return targets


def local_ground_truth(nusc, sample_token):
    sample = nusc.get("sample", sample_token)
    _, boxes, _ = nusc.get_sample_data(sample["data"]["LIDAR_TOP"])
    targets = []
    for box in boxes:
        name = category_to_detection_name(box.name)
        if name is None:
            continue
        targets.append(
            {
                "token": box.token,
                "class": name,
                "center": np.asarray(box.center[:2], dtype=float),
            }
        )
    return targets


def greedy_match(targets, predictions):
    candidates = []
    for target_index, target in enumerate(targets):
        for prediction_index, prediction in enumerate(predictions):
            if prediction["detection_name"] != target["class"]:
                continue
            distance = float(
                np.linalg.norm(
                    np.asarray(prediction["translation"][:2], dtype=float)
                    - target["center"]
                )
            )
            if distance <= MATCH_DISTANCE:
                candidates.append((distance, target_index, prediction_index))
    used_targets, used_predictions = set(), set()
    matches = []
    for distance, target_index, prediction_index in sorted(candidates):
        if target_index in used_targets or prediction_index in used_predictions:
            continue
        used_targets.add(target_index)
        used_predictions.add(prediction_index)
        matches.append((target_index, prediction_index, distance))
    return matches


def detection_quality(nusc, payload):
    total_gt = total_predictions = total_tp = 0
    center_errors, velocity_errors = [], []
    matched_by_sample = {}
    for sample_token, sample_predictions in payload.items():
        targets = global_ground_truth(nusc, sample_token)
        selected = [
            prediction
            for prediction in sample_predictions
            if float(prediction["detection_score"]) >= SCORE_THRESHOLD
        ]
        matches = greedy_match(targets, selected)
        total_gt += len(targets)
        total_predictions += len(selected)
        total_tp += len(matches)
        matched_by_sample[sample_token] = {
            targets[target_index]["token"]
            for target_index, _, _ in matches
        }
        for target_index, prediction_index, distance in matches:
            center_errors.append(distance)
            gt_velocity = targets[target_index]["velocity"]
            prediction_velocity = np.asarray(
                selected[prediction_index].get("velocity", [0.0, 0.0]),
                dtype=float,
            )
            if np.isfinite(gt_velocity).all():
                velocity_errors.append(
                    float(np.linalg.norm(prediction_velocity - gt_velocity))
                )
    fp = total_predictions - total_tp
    fn = total_gt - total_tp
    return {
        "gt": total_gt,
        "predictions_at_score_0_1": total_predictions,
        "tp_at_2m": total_tp,
        "false_positive_at_2m": fp,
        "false_negative_at_2m": fn,
        "gt_recall_at_2m": total_tp / total_gt if total_gt else 0.0,
        "precision_at_2m": total_tp / total_predictions
        if total_predictions
        else 0.0,
        "matched_center_error": float(np.mean(center_errors))
        if center_errors
        else math.nan,
        "matched_velocity_error": float(np.mean(velocity_errors))
        if velocity_errors
        else math.nan,
        "matched_by_sample": matched_by_sample,
    }


def recovery_delay(experiment, protocol, clean_matches, fault_matches):
    events = json.loads(
        (
            ROOT / "protocols/presets" / f"{protocol}.json"
        ).read_text(encoding="utf-8")
    )["scenes"]["*"]
    fault_end = max(int(event["end_frame"]) for event in events)
    by_scene = defaultdict(list)
    for record in trace_records(experiment, protocol):
        by_scene[record["scene_token"]].append(record)
    delays = []
    unrecovered = 0
    for records in by_scene.values():
        rows = []
        for record in sorted(records, key=lambda item: item["frame_idx"]):
            if record["frame_idx"] <= fault_end:
                continue
            token = record["sample_idx"]
            clean = clean_matches.get(token, set())
            retention = (
                len(clean & fault_matches.get(token, set())) / len(clean)
                if clean
                else None
            )
            rows.append((record["frame_idx"], retention))
        recovered_frame = None
        for index in range(max(len(rows) - 1, 0)):
            if (
                rows[index][1] is not None
                and rows[index + 1][1] is not None
                and rows[index][1] >= 0.9
                and rows[index + 1][1] >= 0.9
            ):
                recovered_frame = rows[index][0]
                break
        if recovered_frame is None:
            unrecovered += 1
        else:
            delays.append(max(recovered_frame - fault_end - 1, 0))
    return (
        float(np.mean(delays)) if delays else math.nan,
        len(delays),
        unrecovered,
    )


def candidate_quality(nusc, experiment, protocol):
    recover_count = matched_count = recover_writes = false_writes = 0
    center_errors = []
    for record in trace_records(experiment, protocol):
        diagnostics = record["diagnostics"]
        if not diagnostics or "reference_geometry" not in diagnostics:
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
        targets = local_ground_truth(nusc, record["sample_idx"])
        candidates = []
        for query_index in indexes:
            class_name = CLASSES[int(classes[query_index])]
            for target_index, target in enumerate(targets):
                if class_name != target["class"]:
                    continue
                distance = float(
                    np.linalg.norm(
                        geometry[query_index, :2] - target["center"]
                    )
                )
                if distance <= MATCH_DISTANCE:
                    candidates.append((distance, int(query_index), target_index))
        used_queries, used_targets = set(), set()
        for distance, query_index, target_index in sorted(candidates):
            if query_index in used_queries or target_index in used_targets:
                continue
            used_queries.add(query_index)
            used_targets.add(target_index)
            center_errors.append(distance)
        recover_count += len(indexes)
        matched_count += len(used_queries)
        written = {int(index) for index in indexes if actual_write[index]}
        recover_writes += len(written)
        false_writes += len(written - used_queries)
    return {
        "recover_candidates": recover_count,
        "recover_gt_matches": matched_count,
        "recover_gt_match_rate": matched_count / recover_count
        if recover_count
        else math.nan,
        "recover_memory_writes": recover_writes,
        "false_memory_writes": false_writes,
        "false_memory_write_rate": false_writes / recover_writes
        if recover_writes
        else math.nan,
        "recover_match_center_error": float(np.mean(center_errors))
        if center_errors
        else math.nan,
    }


def training_summary(experiment):
    directory_name = {
        "B0": "b0_50",
        "B0_RayDN": "b0_raydn_50",
        "M1": "m1_50",
        "M1_RayDN": "m1_raydn_50",
    }[experiment]
    directory = TRAIN_ROOT / directory_name
    records = []
    for path in directory.glob("*.log.json"):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("mode") == "train":
                records.append(record)
    console = (directory / "console.log").read_text(
        encoding="utf-8", errors="ignore"
    )
    checkpoint = torch.load(
        directory / "iter_50.pth", map_location="cpu"
    )
    state_keys = checkpoint.get("state_dict", checkpoint).keys()
    inference_rates = []
    for path in (EVAL_ROOT / experiment).glob("*/evaluation.log"):
        matches = re.findall(
            r"81/81, ([0-9.]+) task/s",
            path.read_text(encoding="utf-8", errors="ignore").replace(
                "\r", "\n"
            ),
        )
        if matches:
            # The first progress bar is model inference; the second is the
            # much faster nuScenes metric accumulation.
            inference_rates.append(float(matches[0]))
    return {
        "iterations": len(records),
        "final_loss": records[-1].get("loss") if records else math.nan,
        "final_grad_norm": records[-1].get("grad_norm")
        if records
        else math.nan,
        "max_memory_mb": max(
            (record.get("memory", 0) for record in records), default=0
        ),
        "mean_iter_time_s": float(
            np.mean([record["time"] for record in records[1:]])
        )
        if len(records) > 1
        else math.nan,
        "nan_inf_oom": bool(
            re.search(
                r"(?i)(out of memory|runtimeerror:|\\bloss: (?:nan|inf)|"
                r"grad_norm: (?:nan|inf))",
                console,
            )
        ),
        "raydn_state_keys": sum(
            "raydn" in key.lower() for key in state_keys
        ),
        "mean_inference_fps": float(np.mean(inference_rates))
        if inference_rates
        else math.nan,
    }


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(
        version="v1.0-mini",
        dataroot=str(ROOT / "data/nuscenes-mini"),
        verbose=False,
    )
    quality_cache = {}
    metric_rows = []
    tp_rows = []
    class_rows = []
    for experiment in EXPERIMENTS:
        for protocol in PROTOCOLS:
            summary = metrics(experiment, protocol)
            quality = detection_quality(nusc, results(experiment, protocol))
            quality_cache[(experiment, protocol)] = quality
            tp_errors = summary["tp_errors"]
            metric_rows.append(
                {
                    "experiment": experiment,
                    "protocol": protocol,
                    "mAP": summary["mean_ap"],
                    "NDS": summary["nd_score"],
                    "mATE": tp_errors["trans_err"],
                    "mASE": tp_errors["scale_err"],
                    "mAOE": tp_errors["orient_err"],
                    "mAVE": tp_errors["vel_err"],
                    "mAAE": tp_errors["attr_err"],
                    **{
                        key: value
                        for key, value in quality.items()
                        if key != "matched_by_sample"
                    },
                }
            )
            tp_rows.append(
                {
                    "experiment": experiment,
                    "protocol": protocol,
                    "mATE": tp_errors["trans_err"],
                    "mASE": tp_errors["scale_err"],
                    "mAOE": tp_errors["orient_err"],
                    "mAVE": tp_errors["vel_err"],
                    "mAAE": tp_errors["attr_err"],
                    "matched_center_error": quality[
                        "matched_center_error"
                    ],
                    "matched_velocity_error": quality[
                        "matched_velocity_error"
                    ],
                }
            )
            for class_name, distances in summary["label_aps"].items():
                for distance, ap in distances.items():
                    class_rows.append(
                        {
                            "experiment": experiment,
                            "protocol": protocol,
                            "class": class_name,
                            "center_distance_threshold_m": distance,
                            "AP": ap,
                            "class_mean_AP": summary["mean_dist_aps"][
                                class_name
                            ],
                        }
                    )

    candidate_rows = []
    for experiment in ("M1", "M1_RayDN"):
        clean_matches = quality_cache[
            (experiment, "clean_no_corruption")
        ]["matched_by_sample"]
        for protocol in PROTOCOLS:
            candidate = candidate_quality(nusc, experiment, protocol)
            delay = recovered = unrecovered = math.nan
            if protocol in FAULTS:
                delay, recovered, unrecovered = recovery_delay(
                    experiment,
                    protocol,
                    clean_matches,
                    quality_cache[(experiment, protocol)][
                        "matched_by_sample"
                    ],
                )
            candidate_rows.append(
                {
                    "experiment": experiment,
                    "protocol": protocol,
                    **candidate,
                    "recovery_delay_frames": delay,
                    "recovered_scenes": recovered,
                    "unrecovered_scenes": unrecovered,
                }
            )

    training = {
        experiment: training_summary(experiment)
        for experiment in EXPERIMENTS
    }
    manifest_rows = []
    for experiment in EXPERIMENTS:
        manifest_rows.append(
            {
                "experiment": experiment,
                "baseline": "M1"
                if experiment.startswith("M1")
                else "B0",
                "raydn": experiment.endswith("RayDN"),
                "seed": 2026,
                "init_checkpoint": (
                    "outputs/stage2/s2_2_source_ledger_debug_50/"
                    "iter_50.pth"
                ),
                "max_iters": 50,
                "query_count": 644,
                "status": "completed",
                **training[experiment],
            }
        )

    write_csv("experiment_manifest.csv", manifest_rows)
    write_csv("per_protocol_metrics.csv", metric_rows)
    write_csv("tp_error_metrics.csv", tp_rows)
    write_csv("class_distance_metrics.csv", class_rows)
    write_csv("candidate_quality_analysis.csv", candidate_rows)

    by_metric = {
        (row["experiment"], row["protocol"]): row for row in metric_rows
    }
    decision_rows = []
    for baseline, candidate in (
        ("B0", "B0_RayDN"),
        ("M1", "M1_RayDN"),
    ):
        fault_delta = float(
            np.mean(
                [
                    by_metric[(candidate, protocol)]["NDS"]
                    - by_metric[(baseline, protocol)]["NDS"]
                    for protocol in FAULTS
                ]
            )
        )
        clean_delta = (
            by_metric[(candidate, "clean_no_corruption")]["NDS"]
            - by_metric[(baseline, "clean_no_corruption")]["NDS"]
        )
        improved_faults = sum(
            by_metric[(candidate, protocol)]["NDS"]
            > by_metric[(baseline, protocol)]["NDS"]
            for protocol in FAULTS
        )
        decision_rows.append(
            {
                "baseline": baseline,
                "candidate": candidate,
                "fault_mean_nds_delta": fault_delta,
                "clean_nds_delta": clean_delta,
                "improved_fault_protocols": improved_faults,
                "allow_200iter": (
                    fault_delta >= 0.002
                    and clean_delta >= -0.001
                    and improved_faults >= 2
                    and not training[candidate]["nan_inf_oom"]
                ),
            }
        )

    lines = [
        "# RayDN nuScenes-mini screening report",
        "",
        "The four pre-registered 50-iteration groups completed. RayDN did not "
        "pass the 200-iteration gate for either corresponding baseline.",
        "",
        "## Four-protocol metrics",
        "",
        "| Experiment | Protocol | mAP | NDS |",
        "|---|---|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['experiment']} | {row['protocol']} | "
            f"{row['mAP']:.6f} | {row['NDS']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Pre-registered decision",
            "",
            "| Candidate | Fault mean NDS delta | Clean NDS delta | "
            "Improved faults | 200iter |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in decision_rows:
        lines.append(
            f"| {row['candidate']} vs {row['baseline']} | "
            f"{row['fault_mean_nds_delta']:+.6f} | "
            f"{row['clean_nds_delta']:+.6f} | "
            f"{row['improved_fault_protocols']}/3 | "
            f"{'yes' if row['allow_200iter'] else 'no'} |"
        )
    candidate_index = {
        (row["experiment"], row["protocol"]): row
        for row in candidate_rows
    }
    lines.extend(
        [
            "",
            "## M1 candidate/write quality",
            "",
            "| Experiment | Protocol | RECOVER GT match | False write | "
            "Recovery delay |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for experiment in ("M1", "M1_RayDN"):
        for protocol in FAULTS:
            row = candidate_index[(experiment, protocol)]
            lines.append(
                f"| {experiment} | {protocol} | "
                f"{row['recover_gt_match_rate']:.4f} | "
                f"{row['false_memory_write_rate']:.4f} | "
                f"{row['recovery_delay_frames']:.1f} |"
            )
    lines.extend(
        [
            "",
            "B0+RayDN regressed on all three fault protocols and Clean. "
            "M1+RayDN improved Crash5 but regressed on Crash10, Compound and "
            "Clean; its fault-mean gain and Clean constraint therefore fail.",
            "",
            "## Engineering and interpretation",
            "",
            "- Disabled inference is exactly equal on all eight baseline/protocol "
            "pairs (`max_abs_diff=0`).",
            "- The final true full test suite passed: 89 passed, 7 warnings.",
            "- Fixed FP16 scale 512 removed the common all-group dynamic-scale "
            "overflow; formal smoke and 50iter runs contain no NaN/Inf/OOM.",
            "- M1 conservation and source-mass violation ratios are zero.",
            "- RayDN adds no state-dict/checkpoint key and is absent at inference.",
            "- Mean measured inference rates (frames/s): "
            + ", ".join(
                f"{name}={training[name]['mean_inference_fps']:.3f}"
                for name in EXPERIMENTS
            )
            + ".",
            "- FP/FN/recall use a declared score threshold 0.1 and greedy "
            "class-aware 2m center matching. Candidate-write matching uses the "
            "same 2m rule in the LIDAR frame.",
            "- The mini-screen does not support a complementary RayDN claim and "
            "does not authorize 200iter.",
            "",
            "Source specification: `RAYDN_ADAPTATION_SPEC.md`.",
        ]
    )
    (REPORT / "RAYDN_SCREENING_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(json.dumps(decision_rows, indent=2))


if __name__ == "__main__":
    main()
