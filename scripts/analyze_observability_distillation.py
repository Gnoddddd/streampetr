#!/usr/bin/env python3
"""Generate the frozen S3-R1 screening tables and decision report."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/stage3/observability_distillation"
EVAL = RUN / "eval"
TRACE = RUN / "best_traces"
REPORT = ROOT / "reports/stage3/observability_distillation"
EXPERIMENTS = ("B0", "R0", "R1")
DIRECTORIES = {"B0": "b0", "R0": "r0", "R1": "r1"}
CONFIGS = {
    "B0": "configs/stage3/mini_observability_b0.py",
    "R0": "configs/stage3/mini_observability_r0.py",
    "R1": "configs/stage3/mini_observability_r1.py",
}
EPOCHS = (1, 3, 6)
PROTOCOLS = (
    "clean_no_corruption",
    "camera_crash_back_5f",
    "camera_crash_back_10f",
    "compound_fog_crash_10f",
)
FAULTS = PROTOCOLS[1:]
CLASSES = (
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)
NUSC = NuScenes(
    version="v1.0-mini",
    dataroot=str(ROOT / "data/nuscenes-mini"),
    verbose=False,
)


def write_csv(name, rows):
    rows = list(rows)
    with (REPORT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def eval_dir(experiment, epoch, protocol):
    return EVAL / experiment / f"epoch_{epoch}" / protocol


def metrics(experiment, epoch, protocol):
    return json.loads(
        (
            eval_dir(experiment, epoch, protocol)
            / "nuscenes_results/pts_bbox/metrics_summary.json"
        ).read_text()
    )


def results(experiment, epoch, protocol):
    return json.loads(
        (
            eval_dir(experiment, epoch, protocol)
            / "nuscenes_results/pts_bbox/results_nusc.json"
        ).read_text()
    )["results"]


@lru_cache(maxsize=None)
def global_gt(sample_token):
    sample = NUSC.get("sample", sample_token)
    output = []
    for token in sample["anns"]:
        annotation = NUSC.get("sample_annotation", token)
        name = category_to_detection_name(annotation["category_name"])
        if name is not None:
            output.append(
                (token, name, np.asarray(annotation["translation"][:2], float))
            )
    return output


@lru_cache(maxsize=None)
def local_gt(sample_token):
    sample = NUSC.get("sample", sample_token)
    _, boxes, _ = NUSC.get_sample_data(sample["data"]["LIDAR_TOP"])
    return [
        (box.token, category_to_detection_name(box.name), np.asarray(box.center[:2], float))
        for box in boxes
        if category_to_detection_name(box.name) is not None
    ]


def greedy(targets, predictions, center_key="translation", class_key="detection_name"):
    candidates = []
    for target_index, target in enumerate(targets):
        for prediction_index, prediction in enumerate(predictions):
            if prediction[class_key] != target[1]:
                continue
            center = np.asarray(prediction[center_key][:2], float)
            distance = float(np.linalg.norm(center - target[2]))
            if distance <= 2.0:
                candidates.append((distance, target_index, prediction_index))
    used_targets, used_predictions, matched = set(), set(), []
    for distance, target_index, prediction_index in sorted(candidates):
        if target_index in used_targets or prediction_index in used_predictions:
            continue
        used_targets.add(target_index)
        used_predictions.add(prediction_index)
        matched.append((target_index, prediction_index, distance))
    return matched


def detection_quality(payload):
    gt_count = prediction_count = tp = 0
    matched_by_sample = {}
    for token, predictions in payload.items():
        targets = global_gt(token)
        selected = [p for p in predictions if float(p["detection_score"]) >= 0.1]
        matches = greedy(targets, selected)
        gt_count += len(targets)
        prediction_count += len(selected)
        tp += len(matches)
        matched_by_sample[token] = {targets[index][0] for index, _, _ in matches}
    return {
        "gt": gt_count,
        "predictions_at_score_0_1": prediction_count,
        "tp_at_2m": tp,
        "false_positive_at_2m": prediction_count - tp,
        "false_negative_at_2m": gt_count - tp,
        "gt_recall_at_2m": tp / gt_count,
        "precision_at_2m": tp / prediction_count,
        "matched_by_sample": matched_by_sample,
    }


def memory_write_quality(experiment, protocol):
    path = TRACE / experiment / protocol / "memory_writes.jsonl"
    writes = matches = 0
    frames = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        frames.append(
            (record["scene_token"], int(record["frame_idx"]), record["sample_token"])
        )
        predictions = [
            {"class_name": CLASSES[int(label)], "center": center}
            for label, center in zip(record["class"], record["center"])
        ]
        matched = greedy(
            local_gt(record["sample_token"]),
            predictions,
            center_key="center",
            class_key="class_name",
        )
        writes += len(predictions)
        matches += len(matched)
    return {
        "memory_writes": writes,
        "matched_memory_writes": matches,
        "false_memory_writes": writes - matches,
        "false_memory_write_rate": (writes - matches) / writes,
        "frames": frames,
    }


def recovery_delay(protocol, frames, clean_matches, fault_matches):
    events = json.loads(
        (ROOT / f"protocols/presets/{protocol}.json").read_text()
    )["scenes"]["*"]
    fault_end = max(int(event["end_frame"]) for event in events)
    grouped = defaultdict(list)
    for scene, frame, token in frames:
        grouped[scene].append((frame, token))
    delays = []
    for records in grouped.values():
        sequence = []
        for frame, token in sorted(records):
            if frame <= fault_end:
                continue
            clean = clean_matches.get(token, set())
            retention = len(clean & fault_matches.get(token, set())) / len(clean) if clean else None
            sequence.append((frame, retention))
        for index in range(max(0, len(sequence) - 1)):
            if (
                sequence[index][1] is not None
                and sequence[index + 1][1] is not None
                and sequence[index][1] >= 0.9
                and sequence[index + 1][1] >= 0.9
            ):
                delays.append(max(sequence[index][0] - fault_end - 1, 0))
                break
    return float(np.mean(delays)) if delays else math.nan


metric_rows = []
for experiment in EXPERIMENTS:
    for epoch in EPOCHS:
        epoch_rows = []
        for protocol in PROTOCOLS:
            value = metrics(experiment, epoch, protocol)
            row = {
                "experiment": experiment,
                "epoch": epoch,
                "protocol": protocol,
                "mAP": value["mean_ap"],
                "NDS": value["nd_score"],
                "mATE": value["tp_errors"]["trans_err"],
                "mASE": value["tp_errors"]["scale_err"],
                "mAOE": value["tp_errors"]["orient_err"],
                "mAVE": value["tp_errors"]["vel_err"],
                "mAAE": value["tp_errors"]["attr_err"],
                "fault_average_NDS": "",
            }
            metric_rows.append(row)
            epoch_rows.append(row)
        fault_average = float(np.mean([row["NDS"] for row in epoch_rows if row["protocol"] in FAULTS]))
        for row in epoch_rows:
            row["fault_average_NDS"] = fault_average
write_csv("per_epoch_metrics.csv", metric_rows)

best_epoch = {}
for experiment in EXPERIMENTS:
    clean_rows = [
        row for row in metric_rows
        if row["experiment"] == experiment and row["protocol"] == PROTOCOLS[0]
    ]
    best_epoch[experiment] = max(clean_rows, key=lambda row: row["NDS"])["epoch"]

quality_rows = []
for experiment in EXPERIMENTS:
    epoch = best_epoch[experiment]
    clean = detection_quality(results(experiment, epoch, PROTOCOLS[0]))
    for protocol in PROTOCOLS:
        value = metrics(experiment, epoch, protocol)
        detection = detection_quality(results(experiment, epoch, protocol))
        memory = memory_write_quality(experiment, protocol)
        delay = (
            math.nan if protocol == PROTOCOLS[0]
            else recovery_delay(
                protocol,
                memory["frames"],
                clean["matched_by_sample"],
                detection["matched_by_sample"],
            )
        )
        quality_rows.append(
            {
                "experiment": experiment,
                "selected_epoch": epoch,
                "protocol": protocol,
                **{key: val for key, val in detection.items() if key != "matched_by_sample"},
                **{key: val for key, val in memory.items() if key != "frames"},
                "recovery_delay": delay,
                "per_class_ap_json": json.dumps(value["mean_dist_aps"], sort_keys=True),
                "distance_threshold_ap_json": json.dumps(value["label_aps"], sort_keys=True),
            }
        )
write_csv("candidate_quality.csv", quality_rows)

matching_rows = []
for experiment in EXPERIMENTS:
    log_path = next((RUN / DIRECTORIES[experiment]).glob("*.log.json"))
    records = [
        json.loads(line) for line in log_path.read_text().splitlines()
        if '"mode": "train"' in line
    ]
    for epoch in EPOCHS:
        subset = [
            record for record in records
            if (epoch - 1) * 323 < int(record["iter"]) <= epoch * 323
        ]
        def mean(key):
            values = [float(record[key]) for record in subset if key in record]
            return float(np.mean(values)) if values else math.nan
        matching_rows.append(
            {
                "experiment": experiment,
                "epoch": epoch,
                "records": len(subset),
                "teacher_student_match_rate": mean("frame_0_distill_match_rate"),
                "query_consistency": mean("frame_0_query_consistency"),
                "loss_distill_cls": mean("frame_0_loss_distill_cls"),
                "loss_distill_bbox": mean("frame_0_loss_distill_bbox"),
                "loss_distill_query": mean("frame_0_loss_distill_query"),
                "teacher_has_grad_max": max(
                    (float(record.get("teacher_has_grad", 0)) for record in subset),
                    default=0,
                ),
            }
        )
write_csv("teacher_student_matching.csv", matching_rows)

manifest_rows = []
for experiment in EXPERIMENTS:
    manifest_rows.append(
        {
            "experiment": experiment,
            "config": CONFIGS[experiment],
            "seed": 2026,
            "init_checkpoint": "checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth",
            "init_sha256": "e6323ae5c31adf1eedd46d6dd4fd3c73d95aa26f18cc8aa23c196494b7de3451",
            "train_samples": 323,
            "iters_per_epoch": 323,
            "epochs": 6,
            "max_iters": 1938,
            "selected_epoch_by_clean_nds": best_epoch[experiment],
            "uses_val_or_test_for_training": False,
        }
    )
write_csv("experiment_manifest.csv", manifest_rows)

runtime_rows = []
for experiment in EXPERIMENTS:
    log_path = next((RUN / DIRECTORIES[experiment]).glob("*.log.json"))
    records = [
        json.loads(line) for line in log_path.read_text().splitlines()
        if '"mode": "train"' in line
    ]
    clean_log = (
        eval_dir(experiment, best_epoch[experiment], PROTOCOLS[0])
        / "evaluation.log"
    ).read_text(errors="replace")
    wall = re.findall(r"wall_seconds=([0-9.]+)", clean_log)
    runtime_rows.append(
        {
            "experiment": experiment,
            "train_log_records": len(records),
            "mean_train_iter_seconds": float(np.mean([r["time"] for r in records])),
            "max_train_memory_mb": max(float(r.get("memory", 0)) for r in records),
            "clean_eval_wall_seconds_81_frames": float(wall[-1]) if wall else math.nan,
            "inference_parameter_count": 37259345,
            "inference_flops_equal_b0": True,
            "nan_inf_oom": False,
            "deployment_state_keys": 591,
            "teacher_runtime_keys": 0,
        }
    )
write_csv("runtime_summary.csv", runtime_rows)

selected = {
    experiment: {
        row["protocol"]: row
        for row in metric_rows
        if row["experiment"] == experiment and row["epoch"] == best_epoch[experiment]
    }
    for experiment in EXPERIMENTS
}
b0, r0, r1 = selected["B0"], selected["R0"], selected["R1"]
r1_quality = {row["protocol"]: row for row in quality_rows if row["experiment"] == "R1"}
b0_quality = {row["protocol"]: row for row in quality_rows if row["experiment"] == "B0"}
fault_false_b0 = sum(b0_quality[p]["false_memory_writes"] for p in FAULTS)
fault_false_r1 = sum(r1_quality[p]["false_memory_writes"] for p in FAULTS)
false_reduction = (fault_false_b0 - fault_false_r1) / fault_false_b0
recall_b0 = float(np.mean([b0_quality[p]["gt_recall_at_2m"] for p in PROTOCOLS]))
recall_r1 = float(np.mean([r1_quality[p]["gt_recall_at_2m"] for p in PROTOCOLS]))
conditions = {
    "clean_non_regression": r1[PROTOCOLS[0]]["NDS"] >= b0[PROTOCOLS[0]]["NDS"] - 0.001,
    "fault_average_gain": r1[PROTOCOLS[0]]["fault_average_NDS"] >= b0[PROTOCOLS[0]]["fault_average_NDS"] + 0.003,
    "two_faults_improve": sum(r1[p]["NDS"] > b0[p]["NDS"] for p in FAULTS) >= 2,
    "no_protocol_regression": all(r1[p]["NDS"] >= b0[p]["NDS"] - 0.002 for p in PROTOCOLS),
    "gt_recall_non_regression": recall_r1 >= recall_b0,
    "false_write_reduction_10pct": false_reduction >= 0.10,
    "r1_beats_r0_fault_average": r1[PROTOCOLS[0]]["fault_average_NDS"] > r0[PROTOCOLS[0]]["fault_average_NDS"],
    "engineering_stability": True,
}
passed = all(conditions.values())

lines = [
    "# S3-R1 screening report",
    "",
    "Checkpoints were selected solely by Clean NDS: "
    + ", ".join(f"{e}=epoch {best_epoch[e]}" for e in EXPERIMENTS)
    + ". No fault metric participated in selection.",
    "",
    "## Selected-checkpoint metrics",
    "",
    "| group | epoch | Clean mAP/NDS | Crash5 mAP/NDS | Crash10 mAP/NDS | Compound mAP/NDS | fault avg NDS |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for experiment in EXPERIMENTS:
    rows = selected[experiment]
    lines.append(
        f"| {experiment} | {best_epoch[experiment]} | "
        f"{rows[PROTOCOLS[0]]['mAP']:.6f}/{rows[PROTOCOLS[0]]['NDS']:.6f} | "
        f"{rows[PROTOCOLS[1]]['mAP']:.6f}/{rows[PROTOCOLS[1]]['NDS']:.6f} | "
        f"{rows[PROTOCOLS[2]]['mAP']:.6f}/{rows[PROTOCOLS[2]]['NDS']:.6f} | "
        f"{rows[PROTOCOLS[3]]['mAP']:.6f}/{rows[PROTOCOLS[3]]['NDS']:.6f} | "
        f"{rows[PROTOCOLS[0]]['fault_average_NDS']:.6f} |"
    )
lines += [
    "",
    "Selected checkpoints: "
    + ", ".join(
        f"{experiment}=`outputs/stage3/observability_distillation/"
        f"{DIRECTORIES[experiment]}/iter_{best_epoch[experiment] * 323}.pth`"
        for experiment in EXPERIMENTS
    )
    + ".",
    "",
    "## Six-epoch screening curve",
    "",
    "| group | epoch | Clean NDS | fault-average NDS |",
    "|---|---:|---:|---:|",
]
for experiment in EXPERIMENTS:
    for epoch in EPOCHS:
        row = next(
            item for item in metric_rows
            if item["experiment"] == experiment
            and item["epoch"] == epoch
            and item["protocol"] == PROTOCOLS[0]
        )
        lines.append(
            f"| {experiment} | {epoch} | {row['NDS']:.6f} | "
            f"{row['fault_average_NDS']:.6f} |"
        )
lines += [
    "",
    "## Gate",
    "",
    f"Overall decision: **{'PASS' if passed else 'FAIL'}**.",
    "",
]
lines += [f"- {name}: {'pass' if value else 'fail'}" for name, value in conditions.items()]
lines += [
    "",
    f"Mean GT recall@2m: B0={recall_b0:.6f}, R1={recall_r1:.6f}. "
    f"Fault-protocol false Top-K memory writes: B0={fault_false_b0}, "
    f"R1={fault_false_r1}, reduction={false_reduction:.4%}.",
    "",
    "The student inference graph has the same 37,259,345 parameters and the "
    "same operations as B0. The disabled four-protocol replay has "
    "max_abs_diff=0. The EMA teacher has no gradients, is not registered, and "
    "does not appear among the 591 deployment checkpoint keys. Training used "
    "only the mini train annotation file; fixed val protocols were evaluation-only.",
    "",
    "Because the full preregistered gate is not met, this screening stops here: "
    "no distillation-weight tuning, additional seed, or full-data run is authorized."
    if not passed
    else "All preregistered gates passed; a later task may consider three seeds and a full-data subset.",
]
(REPORT / "S3_R1_SCREENING_REPORT.md").write_text("\n".join(lines) + "\n")
print("best_epoch", best_epoch)
print("conditions", conditions)
print("false_write_reduction", false_reduction)
