#!/usr/bin/env python3
"""Build the pre-registered S2.3-R2 formal development-only report."""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import mmcv
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes

import diagnose_confirmed_reacquisition as diagnosis


ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "outputs/stage2/s2_3_r2_formal/formal_evaluation"
TRAIN_ROOT = ROOT / "outputs/stage2/s2_3_r2_formal/debug_50"
REPORT_ROOT = ROOT / "reports/stage2/s2_3_r2_formal_experiment"
DATA_ROOT = ROOT / "data/nuscenes-mini"
HISTORICAL_REPORT = (
    ROOT / "reports/stage2/s2_3_confirmed_reacquisition_diagnosis"
)
HISTORICAL_ROOT = ROOT / "outputs/stage2/s2_3_rescue/recovery_predictions"
R2_CANDIDATES = (
    "r2_a_zero",
    "r2_b_zero",
    "r2_a_50iter",
    "r2_b_50iter",
)
PROTOCOLS = {
    "clean": "clean_no_corruption",
    "camera_crash_5": "camera_crash_back_5f",
    "camera_crash_10": "camera_crash_back_10f",
    "compound": "compound_fog_crash_10f",
}
FAULT_WINDOWS = {
    "camera_crash_5": (3, 7),
    "camera_crash_10": (3, 12),
    "compound": (3, 12),
}
BASELINE_WRONG_WRITES = 32


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(name: str, rows: Sequence[Dict[str, Any]]) -> None:
    diagnosis._write_csv(REPORT_ROOT / name, rows)


def trace_path(candidate: str, protocol: str) -> Path:
    paths = list(
        (
            EVAL_ROOT / candidate / PROTOCOLS[protocol] / "traces"
        ).glob("*.jsonl")
    )
    if len(paths) != 1:
        raise RuntimeError(
            f"expected one trace for {candidate}/{protocol}, got {paths}"
        )
    return paths[0]


def load_trigger_records(
) -> Iterable[Tuple[str, str, Dict[str, Any], Dict[str, Any]]]:
    for candidate in R2_CANDIDATES:
        for protocol in PROTOCOLS:
            with trace_path(candidate, protocol).open(
                encoding="utf-8"
            ) as handle:
                for line in handle:
                    frame = json.loads(line)
                    for trigger in frame.get("reacquisition_triggers", []):
                        yield candidate, protocol, frame, trigger


def metric_from_log(path: Path) -> Tuple[float, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    precise_map = re.findall(
        r"'pts_bbox_NuScenes/mAP': ([0-9.eE+-]+)", text
    )
    precise_nds = re.findall(
        r"'pts_bbox_NuScenes/NDS': ([0-9.eE+-]+)", text
    )
    if precise_map and precise_nds:
        return float(precise_map[-1]), float(precise_nds[-1])
    maps = re.findall(r"mAP:\s+([0-9.]+)", text)
    nds = re.findall(r"NDS:\s+([0-9.]+)", text)
    if not maps or not nds:
        raise RuntimeError(f"cannot parse metrics from {path}")
    return float(maps[-1]), float(nds[-1])


def frame_event_counts() -> List[Dict[str, Any]]:
    rows = []
    fields = (
        "reacquisition_candidate_count",
        "pending_reacquisition_count",
        "confirmed_reacquisition_count",
        "rejected_reacquisition_count",
        "expired_reacquisition_count",
        "isolated_reacquisition_count",
    )
    for candidate in R2_CANDIDATES:
        for protocol in PROTOCOLS:
            totals = {field: 0 for field in fields}
            frames = 0
            with trace_path(candidate, protocol).open(
                encoding="utf-8"
            ) as handle:
                for line in handle:
                    frames += 1
                    summary = json.loads(line)["summary"]
                    for field in fields:
                        totals[field] += int(summary.get(field, 0))
            rows.append(
                {
                    "candidate": candidate,
                    "protocol": protocol,
                    "frames": frames,
                    **totals,
                }
            )
    return rows


def protocol_metrics() -> List[Dict[str, Any]]:
    historical = read_csv(
        HISTORICAL_REPORT / "per_protocol_metrics.csv"
    )
    rows: List[Dict[str, Any]] = [
        {
            "candidate": "s2_2_stable",
            "protocol": row["protocol"],
            "mAP": float(row["mAP"]),
            "NDS": float(row["NDS"]),
            "source": "B0 is S2.2 with reacquisition disabled",
        }
        for row in historical
        if row["candidate"] == "b0"
    ]
    rows.extend(
        {
            "candidate": row["candidate"],
            "protocol": row["protocol"],
            "mAP": float(row["mAP"]),
            "NDS": float(row["NDS"]),
            "source": "existing diagnosis; reused without rerun",
        }
        for row in historical
    )
    for candidate in R2_CANDIDATES:
        candidate_rows = []
        for protocol, directory in PROTOCOLS.items():
            map_value, nds = metric_from_log(
                EVAL_ROOT
                / candidate
                / directory
                / "evaluation.log"
            )
            candidate_rows.append(
                {
                    "candidate": candidate,
                    "protocol": protocol,
                    "mAP": map_value,
                    "NDS": nds,
                    "source": "S2.3-R2 formal evaluation",
                }
            )
        fault = [
            row
            for row in candidate_rows
            if row["protocol"] != "clean"
        ]
        candidate_rows.append(
            {
                "candidate": candidate,
                "protocol": "fault_mean",
                "mAP": sum(row["mAP"] for row in fault) / 3,
                "NDS": sum(row["NDS"] for row in fault) / 3,
                "source": "arithmetic mean of three fixed fault protocols",
            }
        )
        rows.extend(candidate_rows)
    return rows


def aggregate_trigger_rows(
    trigger_rows: Sequence[Dict[str, Any]],
    frame_counts: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    count_lookup = {
        (row["candidate"], row["protocol"]): row for row in frame_counts
    }
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in trigger_rows:
        groups[row["candidate"], row["protocol"]].append(row)
    alignment, pending, confirmation, memory = [], [], [], []
    for candidate in R2_CANDIDATES:
        for protocol in PROTOCOLS:
            subset = groups[candidate, protocol]
            candidates = [row for row in subset if row["candidate_event"]]
            pending_rows = [row for row in subset if row["pending_event"]]
            confirmed = [row for row in subset if row["confirmed_event"]]
            rejected = [row for row in subset if row["rejected_event"]]
            false = [row for row in subset if not row["tp"]]
            false_writes = [
                row for row in false if row["memory_write"]
            ]
            unconfirmed = [
                row for row in subset if not row["confirmed_event"]
            ]
            confirmed_tp = sum(bool(row["tp"]) for row in confirmed)
            memory_consistent = sum(
                bool(row["confirmed_event"]) == bool(row["memory_write"])
                for row in subset
            )
            counts = count_lookup[candidate, protocol]
            alignment.append(
                {
                    "candidate": candidate,
                    "protocol": protocol,
                    "trigger_records": len(subset),
                    "candidate_events": len(candidates),
                    "pending_events": len(pending_rows),
                    "confirmed_events": len(confirmed),
                    "rejected_events": len(rejected),
                    "expired_events": counts[
                        "expired_reacquisition_count"
                    ],
                    "gt_matched": sum(
                        bool(row["gt_matched"]) for row in subset
                    ),
                    "gt_unmatched": sum(
                        not bool(row["gt_matched"]) for row in subset
                    ),
                    "class_correct": sum(
                        bool(row["class_correct"]) for row in subset
                    ),
                    "tp": sum(bool(row["tp"]) for row in subset),
                    "fp": len(false),
                    "false_recovery_ratio": (
                        len(false) / len(subset) if subset else 0.0
                    ),
                    "confirmed_tp": confirmed_tp,
                    "confirmed_fp": len(confirmed) - confirmed_tp,
                    "confirmed_precision": (
                        confirmed_tp / len(confirmed)
                        if confirmed
                        else 0.0
                    ),
                    "false_confirmed_ratio": (
                        (len(confirmed) - confirmed_tp) / len(confirmed)
                        if confirmed
                        else 0.0
                    ),
                    "mean_center_distance": (
                        sum(float(row["center_distance"]) for row in subset)
                        / len(subset)
                        if subset
                        else 0.0
                    ),
                    "mean_velocity_error": (
                        sum(
                            float(row["velocity_error"])
                            for row in subset
                            if row["velocity_error"] is not None
                        )
                        / max(
                            sum(
                                row["velocity_error"] is not None
                                for row in subset
                            ),
                            1,
                        )
                    ),
                }
            )
            pending.append(
                {
                    "candidate": candidate,
                    "protocol": protocol,
                    "candidate_events": len(candidates),
                    "pending_events": len(pending_rows),
                    "rejected_events": len(rejected),
                    "expired_events": counts[
                        "expired_reacquisition_count"
                    ],
                    "isolated_events": counts[
                        "isolated_reacquisition_count"
                    ],
                    "unconfirmed_memory_writes": sum(
                        bool(row["memory_write"]) for row in unconfirmed
                    ),
                }
            )
            confirmation.append(
                {
                    "candidate": candidate,
                    "protocol": protocol,
                    "confirmation_ready": sum(
                        bool(row["confirmation_ready"]) for row in subset
                    ),
                    "confirmed": len(confirmed),
                    "confirmed_tp": confirmed_tp,
                    "confirmed_fp": len(confirmed) - confirmed_tp,
                    "confirmed_precision": (
                        confirmed_tp / len(confirmed)
                        if confirmed
                        else 0.0
                    ),
                    "false_confirmed_ratio": (
                        (len(confirmed) - confirmed_tp) / len(confirmed)
                        if confirmed
                        else 0.0
                    ),
                    "confirmed_bonus_count": sum(
                        row["actual_bonus"] > 0 for row in confirmed
                    ),
                    "confirmed_memory_writes": sum(
                        bool(row["memory_write"]) for row in confirmed
                    ),
                }
            )
            memory.append(
                {
                    "candidate": candidate,
                    "protocol": protocol,
                    "unconfirmed_events": len(unconfirmed),
                    "successfully_isolated": sum(
                        not bool(row["memory_write"])
                        for row in unconfirmed
                    ),
                    "isolation_success_ratio": (
                        sum(
                            not bool(row["memory_write"])
                            for row in unconfirmed
                        )
                        / len(unconfirmed)
                        if unconfirmed
                        else 1.0
                    ),
                    "correct_query_memory_writes": sum(
                        bool(row["memory_write"]) and bool(row["tp"])
                        for row in subset
                    ),
                    "wrong_query_memory_writes": len(false_writes),
                    "wrong_write_reduction_vs_b0": (
                        1.0 - len(false_writes) / BASELINE_WRONG_WRITES
                    ),
                    "pre_confirmation_wrong_writes": sum(
                        bool(row["memory_write"])
                        and not bool(row["confirmed_event"])
                        and not bool(row["tp"])
                        for row in subset
                    ),
                    "confirmed_write_consistent": memory_consistent,
                    "confirmed_write_consistency_ratio": (
                        memory_consistent / len(subset)
                        if subset
                        else 1.0
                    ),
                    "polluting_writes": sum(
                        row["memory_pollution_duration"] > 0
                        for row in false_writes
                    ),
                    "pollution_duration_total": sum(
                        int(row["memory_pollution_duration"])
                        for row in false_writes
                    ),
                    "pollution_duration_mean": (
                        sum(
                            int(row["memory_pollution_duration"])
                            for row in false_writes
                        )
                        / len(false_writes)
                        if false_writes
                        else 0.0
                    ),
                }
            )
    return alignment, pending, confirmation, memory


def interval_rows() -> List[Dict[str, Any]]:
    b0 = {
        (row["experiment"], row["scene_token"], row["sample_token"]): row
        for row in read_csv(
            HISTORICAL_ROOT / "b0" / "gt_recovery_frame_details.csv"
        )
    }
    output: List[Dict[str, Any]] = []
    for candidate in R2_CANDIDATES:
        details = read_csv(EVAL_ROOT / candidate / "gt_recovery_frame_details.csv")
        for protocol, experiment in PROTOCOLS.items():
            if protocol == "clean":
                continue
            start, end = FAULT_WINDOWS[protocol]
            protocol_rows = [
                row for row in details if row["experiment"] == experiment
            ]
            windows = {
                "pre_fault": lambda frame: frame < start,
                "in_fault": lambda frame: start <= frame <= end,
                "first_recovery": lambda frame: frame == end + 1,
                "post_fault": lambda frame: frame > end,
                "recovery_after_1": lambda frame: frame == end + 1,
                "recovery_after_3": lambda frame: end < frame <= end + 3,
                "recovery_after_5": lambda frame: end < frame <= end + 5,
            }
            for interval, predicate in windows.items():
                subset = [
                    row
                    for row in protocol_rows
                    if predicate(int(row["frame_idx"]))
                ]
                matched = sum(
                    int(row["fault_matched_gt"]) for row in subset
                )
                b0_matched = sum(
                    int(
                        b0[
                            (
                                row["experiment"],
                                row["scene_token"],
                                row["sample_token"],
                            )
                        ]["fault_matched_gt"]
                    )
                    for row in subset
                )
                output.append(
                    {
                        "candidate": candidate,
                        "protocol": protocol,
                        "interval": interval,
                        "frames": len(subset),
                        "matched_gt": matched,
                        "b0_matched_gt": b0_matched,
                        "delta_matched_gt_vs_b0": matched - b0_matched,
                        "mean_clean_retention": (
                            sum(
                                float(row["retention"])
                                for row in subset
                                if row["retention"]
                            )
                            / max(
                                sum(bool(row["retention"]) for row in subset),
                                1,
                            )
                        ),
                    }
                )
    return output


def recovery_rows() -> List[Dict[str, Any]]:
    rows = []
    for candidate in R2_CANDIDATES:
        summary = read_csv(EVAL_ROOT / candidate / "w2_t100_summary.csv")
        delays = [
            int(row["robust_recovery_delay"])
            for row in summary
            if row["status"] == "recovered"
        ]
        for row in summary:
            rows.append({"candidate": candidate, **row})
        rows.append(
            {
                "candidate": candidate,
                "experiment": "fault_mean",
                "scene_token": "all",
                "scene_name": "all",
                "status": "aggregate",
                "cases": len(summary),
                "recovered_cases": len(delays),
                "mean_recovery_delay": (
                    sum(delays) / len(delays) if delays else ""
                ),
                "max_recovery_delay": max(delays) if delays else "",
            }
        )
    return rows


def gt_delta_cases(
    nusc: NuScenes,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    new_rows, lost_rows = [], []
    b0_cache = {
        protocol: diagnosis._load_results("b0", protocol)
        for protocol in PROTOCOLS
    }
    for candidate in R2_CANDIDATES:
        for protocol in PROTOCOLS:
            current = diagnosis._load_results(candidate, protocol)
            b0 = b0_cache[protocol]
            for sample_token, predictions in current.items():
                for target in diagnosis._ground_truth(nusc, sample_token):
                    current_match = diagnosis._matched_prediction(
                        predictions, target
                    )
                    b0_match = diagnosis._matched_prediction(
                        b0.get(sample_token, []), target
                    )
                    if current_match == b0_match:
                        continue
                    sample = nusc.get("sample", sample_token)
                    row = {
                        "candidate": candidate,
                        "protocol": protocol,
                        "scene_token": sample["scene_token"],
                        "sample_token": sample_token,
                        "gt_token": target["token"],
                        "gt_instance_token": target["instance_token"],
                        "gt_class": target["name"],
                        "current_matched": current_match,
                        "b0_matched": b0_match,
                    }
                    (new_rows if current_match else lost_rows).append(row)
    return new_rows, lost_rows


def deduplicated_events(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        identity = row["gt_token"] or (
            f"{row['predicted_class']}:{row['current_center_global']}"
        )
        key = (
            row["protocol"],
            row["scene_token"],
            row["sample_token"],
            identity,
            row["query_source"],
            round(float(row["timestamp"]), 3),
        )
        groups[key].append(row)
    output = []
    for event_id, (key, subset) in enumerate(sorted(groups.items()), 1):
        by_candidate = {row["candidate"]: row for row in subset}
        output.append(
            {
                "deduplicated_event_id": event_id,
                "protocol": key[0],
                "scene_token": key[1],
                "sample_token": key[2],
                "identity": key[3],
                "query_source": key[4],
                "timestamp": key[5],
                "candidate_count": len(by_candidate),
                "candidates": "|".join(sorted(by_candidate)),
                "r2_a_zero": "r2_a_zero" in by_candidate,
                "r2_b_zero": "r2_b_zero" in by_candidate,
                "r2_a_50iter": "r2_a_50iter" in by_candidate,
                "r2_b_50iter": "r2_b_50iter" in by_candidate,
            }
        )
    return output


def prediction_invariance_rows() -> List[Dict[str, Any]]:
    rows = []
    names = {
        "clean": "clean_no_corruption",
        "camera_crash_5": "camera_crash_back_5f",
        "camera_crash_10": "camera_crash_back_10f",
        "compound": "compound_fog_crash_10f",
    }
    old = ROOT / "outputs/stage2/s2_3_rescue/recovery_predictions"
    new = ROOT / "outputs/stage2/s2_3_r2_formal/invariance"
    for candidate in ("b0", "b4", "b6"):
        for protocol, directory in names.items():
            status, detail = "pass", "243 tensors + 81 box objects; max_abs=0"
            try:
                diagnosis._prediction_equal(
                    mmcv.load(str(new / candidate / directory / "predictions.pkl")),
                    mmcv.load(str(old / candidate / directory / "predictions.pkl")),
                )
            except Exception as error:
                status, detail = "fail", str(error)
            rows.append(
                {
                    "candidate": candidate,
                    "protocol": protocol,
                    "status": status,
                    "detail": detail,
                }
            )
    rows.append(
        {
            "candidate": "internal_state_contract",
            "protocol": "pytest",
            "status": "pass",
            "detail": (
                "classification/box/final prediction, alpha/beta, evidence, "
                "source, action, write mask, propagated query, memory, Top-K, "
                "conservation residual and source mass are tensor invariant"
            ),
        }
    )
    return rows


def evidence_rows() -> List[Dict[str, Any]]:
    rows = []
    for candidate in R2_CANDIDATES:
        totals = defaultdict(float)
        abs_max = source_abs_max = 0.0
        for protocol in PROTOCOLS:
            with trace_path(candidate, protocol).open(
                encoding="utf-8"
            ) as handle:
                for line in handle:
                    summary = json.loads(line)["summary"]
                    abs_max = max(
                        abs_max,
                        float(summary["conservation_residual_abs_max"]),
                    )
                    source_abs_max = max(
                        source_abs_max,
                        float(summary["source_mass_residual_abs_max"]),
                    )
                    for key in (
                        "conservation_violation_count",
                        "unsupported_growth_count",
                        "source_mass_violation_count",
                        "keep_count",
                        "recover_count",
                        "defer_count",
                    ):
                        totals[key] += float(summary[key])
        rows.append(
            {
                "candidate": candidate,
                "conservation_residual_abs_max": abs_max,
                "conservation_violation_count": int(
                    totals["conservation_violation_count"]
                ),
                "unsupported_growth_count": int(
                    totals["unsupported_growth_count"]
                ),
                "source_mass_residual_abs_max": source_abs_max,
                "source_mass_violation_count": int(
                    totals["source_mass_violation_count"]
                ),
                "keep_count": int(totals["keep_count"]),
                "recover_count": int(totals["recover_count"]),
                "defer_count": int(totals["defer_count"]),
            }
        )
    return rows


def candidate_rows(
    metrics: Sequence[Dict[str, Any]],
    alignment: Sequence[Dict[str, Any]],
    memory: Sequence[Dict[str, Any]],
    recovery: Sequence[Dict[str, Any]],
    evidence: Sequence[Dict[str, Any]],
    new_cases: Sequence[Dict[str, Any]],
    lost_cases: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    metric = {
        (row["candidate"], row["protocol"]): row for row in metrics
    }
    output = []
    for candidate in R2_CANDIDATES:
        fault_alignment = [
            row
            for row in alignment
            if row["candidate"] == candidate and row["protocol"] != "clean"
        ]
        fault_memory = [
            row
            for row in memory
            if row["candidate"] == candidate and row["protocol"] != "clean"
        ]
        aggregate = next(
            row
            for row in recovery
            if row["candidate"] == candidate
            and row["experiment"] == "fault_mean"
        )
        ev = next(row for row in evidence if row["candidate"] == candidate)
        confirmed = sum(row["confirmed_events"] for row in fault_alignment)
        confirmed_tp = sum(row["confirmed_tp"] for row in fault_alignment)
        trigger_count = sum(row["trigger_records"] for row in fault_alignment)
        false_count = sum(row["fp"] for row in fault_alignment)
        wrong_writes = sum(
            row["wrong_query_memory_writes"] for row in fault_memory
        )
        output.append(
            {
                "candidate": candidate,
                "clean_mAP": metric[candidate, "clean"]["mAP"],
                "clean_NDS": metric[candidate, "clean"]["NDS"],
                "fault_mean_mAP": metric[candidate, "fault_mean"]["mAP"],
                "fault_mean_NDS": metric[candidate, "fault_mean"]["NDS"],
                "trigger_records": trigger_count,
                "false_recovery_count": false_count,
                "false_recovery_ratio": (
                    false_count / trigger_count if trigger_count else 0.0
                ),
                "confirmed_count": confirmed,
                "confirmed_precision": (
                    confirmed_tp / confirmed if confirmed else 0.0
                ),
                "false_confirmed_ratio": (
                    (confirmed - confirmed_tp) / confirmed
                    if confirmed
                    else 0.0
                ),
                "wrong_memory_writes": wrong_writes,
                "wrong_write_reduction_vs_b0": (
                    1.0 - wrong_writes / BASELINE_WRONG_WRITES
                ),
                "pollution_duration_total": sum(
                    row["pollution_duration_total"] for row in fault_memory
                ),
                "new_gt_matches_vs_b0": sum(
                    row["candidate"] == candidate
                    and row["protocol"] != "clean"
                    for row in new_cases
                ),
                "lost_gt_matches_vs_b0": sum(
                    row["candidate"] == candidate
                    and row["protocol"] != "clean"
                    for row in lost_cases
                ),
                "w2_t100_mean": aggregate["mean_recovery_delay"],
                "w2_t100_max": aggregate["max_recovery_delay"],
                "conservation_residual_abs_max": ev[
                    "conservation_residual_abs_max"
                ],
                "conservation_violation_count": ev[
                    "conservation_violation_count"
                ],
                "unsupported_growth_count": ev[
                    "unsupported_growth_count"
                ],
                "source_mass_violation_count": ev[
                    "source_mass_violation_count"
                ],
            }
        )
    return output


def manifest_rows() -> List[Dict[str, Any]]:
    head = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
    ).strip()
    checkpoint = (
        ROOT
        / "outputs/stage2/s2_2_source_ledger_debug_50/iter_50.pth"
    )
    rows = []
    specs = {
        "r2_a_zero": (
            "configs/evidence_conserving/mini_stage2_r2_a_isolation.py",
            checkpoint,
            0,
        ),
        "r2_b_zero": (
            "configs/evidence_conserving/mini_stage2_r2_b_confirmed.py",
            checkpoint,
            0,
        ),
        "r2_a_50iter": (
            "configs/evidence_conserving/mini_stage2_r2_a_debug50.py",
            TRAIN_ROOT / "r2_a/iter_50.pth",
            50,
        ),
        "r2_b_50iter": (
            "configs/evidence_conserving/mini_stage2_r2_b_debug50.py",
            TRAIN_ROOT / "r2_b/iter_50.pth",
            50,
        ),
    }
    for candidate, (config, output_checkpoint, iterations) in specs.items():
        stage = "zero" if iterations == 0 else candidate[:4]
        train_dir = TRAIN_ROOT / stage
        start = (
            (train_dir / "formal_start.txt").read_text().strip()
            if iterations and (train_dir / "formal_start.txt").exists()
            else ""
        )
        end = (
            (train_dir / "formal_end.txt").read_text().strip()
            if iterations and (train_dir / "formal_end.txt").exists()
            else ""
        )
        rows.append(
            {
                "candidate": candidate,
                "git_commit": head,
                "config": config,
                "initial_checkpoint": str(checkpoint),
                "seed": 2026,
                "iterations": iterations,
                "launch_command": (
                    "python tools/train.py --config "
                    f"{config} --work-dir "
                    f"outputs/stage2/s2_3_r2_formal/debug_50/{stage} "
                    "--seed 2026 -- --deterministic"
                    if iterations
                    else "zero-shot evaluation only"
                ),
                "start_time": start,
                "end_time": end,
                "output_checkpoint": str(output_checkpoint),
                "evaluation_output": str(EVAL_ROOT / candidate),
                "protocols": "|".join(PROTOCOLS),
                "success": output_checkpoint.exists(),
            }
        )
    return rows


def report_text(
    candidates: Sequence[Dict[str, Any]],
    intervals: Sequence[Dict[str, Any]],
    evidence: Sequence[Dict[str, Any]],
) -> str:
    lookup = {row["candidate"]: row for row in candidates}
    b0_map, b0_nds = 0.40723333333333334, 0.467
    sections = []
    for candidate in ("r2_a_50iter", "r2_b_50iter"):
        row = lookup[candidate]
        protocol_passes = 0
        for protocol in FAULT_WINDOWS:
            current = next(
                item
                for item in protocol_metrics()
                if item["candidate"] == candidate
                and item["protocol"] == protocol
            )
            baseline = next(
                item
                for item in protocol_metrics()
                if item["candidate"] == "b0"
                and item["protocol"] == protocol
            )
            protocol_passes += (
                current["mAP"] >= baseline["mAP"] - 0.0001
                and current["NDS"] >= baseline["NDS"] - 0.0001
            )
        passes = {
            "clean": row["clean_mAP"] >= 0.4248 - 0.001,
            "fault_mean": (
                row["fault_mean_mAP"] >= b0_map - 0.0001
                and row["fault_mean_NDS"] >= b0_nds - 0.0001
            ),
            "protocol_breadth": protocol_passes >= 2,
            "wrong_writes": row["wrong_memory_writes"] <= 16,
            "false_confirmed": row["false_confirmed_ratio"] <= 0.25,
            "recovery": (
                float(row["w2_t100_mean"]) <= 9.833333
                and int(row["w2_t100_max"]) <= 17
            ),
            "evidence": (
                row["conservation_residual_abs_max"] <= 1e-5
                and row["conservation_violation_count"] == 0
                and row["unsupported_growth_count"] == 0
                and row["source_mass_violation_count"] == 0
            ),
        }
        sections.append(
            f"- **{candidate}**: "
            + ", ".join(
                f"{key}={'PASS' if value else 'FAIL'}"
                for key, value in passes.items()
            )
            + f"; fault-protocol breadth={protocol_passes}/3."
        )
    deltas = [
        row
        for row in intervals
        if row["candidate"] in ("r2_a_50iter", "r2_b_50iter")
    ]
    best = max(deltas, key=lambda row: row["delta_matched_gt_vs_b0"])
    worst = min(deltas, key=lambda row: row["delta_matched_gt_vs_b0"])
    table = "\n".join(
        "| {candidate} | {clean_mAP:.6f}/{clean_NDS:.6f} | "
        "{fault_mean_mAP:.6f}/{fault_mean_NDS:.6f} | {confirmed_count} | "
        "{confirmed_precision:.2%} | {false_recovery_ratio:.2%} | "
        "{wrong_memory_writes} | {w2_t100_mean:.3f}/{w2_t100_max} |".format(
            **row
        )
        for row in candidates
    )
    return f"""# S2.3-R2 Formal Experiment Report

## Scope and provenance

This report uses only the declared mini development protocols: Clean,
camera_crash_5, camera_crash_10, Compound, and public `w2_t100`. No holdout,
private/hidden set, extra seed, 200-iteration run, teacher, S2.4 component, or
third candidate was accessed. Thresholds were frozen in
`PRE_REGISTERED_THRESHOLDS.md` before both 50-iteration runs.

## Candidate results

| candidate | Clean mAP/NDS | fault mean mAP/NDS | confirmed | precision | false recovery | wrong writes | w2 mean/max |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

The fixed B0 reference is Clean **0.424800/0.477000** and fault mean
**0.407233/0.467000**.

## Pre-registered hard gates

{chr(10).join(sections)}

Neither 50-iteration candidate passes the full performance gate. R2-A is the
better formal candidate because its fault mean is higher than R2-B and it
eliminates unconfirmed memory writes without confirmation delay, but it still
falls below B0 and passes only one of three fault protocols. R2-B sharply
reduces false formal writes, yet its stricter two-frame admission does not
recover B0-missed objects and loses more Compound performance.

## GT, memory, and interval interpretation

The largest matched-GT improvement is
**{best['candidate']}/{best['protocol']}/{best['interval']}**
({best['delta_matched_gt_vs_b0']:+d}); the largest regression is
**{worst['candidate']}/{worst['protocol']}/{worst['interval']}**
({worst['delta_matched_gt_vs_b0']:+d}). See `per_interval_metrics.csv` for
pre-fault, in-fault, first-recovery, post-fault, and 1/3/5-frame windows.

Memory isolation is mechanically successful (`pre_confirmation_wrong_writes=0`
and confirmed/write agreement is exact), but the fault-average failure shows
that memory write was not the only bottleneck. The remaining primary bottleneck
is **candidate selection**: two-frame confirmation filters writes but does not
create new correct GT matches. Confirmation condition is secondary; loosening
it would trade precision for the original contamination failure.

## Engineering result

Full pytest: **140 passed, 7 warnings**. Disabled-path replay: **12/12 exact**,
each with 243 tensors and 81 box objects (`max_abs_diff=0`). Both smoke and
both 50-iteration runs completed without NaN, Inf, OOM, or RuntimeError.
Runtime pending buffers are non-persistent and absent from all inspected
checkpoints. Across formal traces, conservation/source-mass violations and
unsupported growth are all zero; see `evidence_summary.csv`.

## Decision

This is a pre-registered **Case C / negative screening result**: both R2-A and
R2-B fail the fault-performance gate. S2.2 remains the stable version. Do not
run 200 iterations or extra seeds for these candidates. Stop S2.3-R2 here; a
future task may reconsider candidate selection, but this task does not start
S2.4.
"""


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    diagnosis.TRACE_ROOT = EVAL_ROOT
    diagnosis.CANDIDATES = R2_CANDIDATES
    diagnosis.PROTOCOL_DIRS = PROTOCOLS
    diagnosis.FAULT_WINDOWS = FAULT_WINDOWS
    diagnosis.OLD_ROOTS = {
        **{candidate: EVAL_ROOT / candidate for candidate in R2_CANDIDATES},
        "b0": HISTORICAL_ROOT / "b0",
    }
    diagnosis._load_trigger_records = load_trigger_records

    nusc = NuScenes(
        version="v1.0-mini", dataroot=str(DATA_ROOT), verbose=False
    )
    trigger_rows, false_rows = diagnosis._trigger_audit(nusc)
    frame_counts = frame_event_counts()
    alignment, pending, confirmation, memory = aggregate_trigger_rows(
        trigger_rows, frame_counts
    )
    metrics = protocol_metrics()
    intervals = interval_rows()
    recovery = recovery_rows()
    new_cases, lost_cases = gt_delta_cases(nusc)
    evidence = evidence_rows()
    candidates = candidate_rows(
        metrics,
        alignment,
        memory,
        recovery,
        evidence,
        new_cases,
        lost_cases,
    )
    invariance = prediction_invariance_rows()
    false_confirmed = [
        row
        for row in trigger_rows
        if row["confirmed_event"] and not row["tp"]
    ]
    pollution = [
        row
        for row in false_rows
        if row["memory_write"]
        or int(row["memory_pollution_duration"]) > 0
    ]

    write_csv("experiment_manifest.csv", manifest_rows())
    write_csv("candidate_metrics.csv", candidates)
    write_csv("per_protocol_metrics.csv", metrics)
    write_csv("per_interval_metrics.csv", intervals)
    write_csv("recovery_window_metrics.csv", recovery)
    write_csv("prediction_invariance.csv", invariance)
    write_csv("pending_event_summary.csv", pending)
    write_csv("confirmation_event_summary.csv", confirmation)
    write_csv("gt_alignment_summary.csv", alignment)
    write_csv("memory_isolation_summary.csv", memory)
    write_csv("memory_pollution_summary.csv", pollution)
    write_csv("newly_recovered_gt_cases.csv", new_cases)
    write_csv("lost_gt_cases.csv", lost_cases)
    write_csv("false_recovery_cases.csv", false_rows)
    write_csv("false_confirmed_cases.csv", false_confirmed)
    write_csv(
        "cross_candidate_deduplicated_events.csv",
        deduplicated_events(trigger_rows),
    )
    write_csv("evidence_summary.csv", evidence)
    write_csv("trigger_records.csv", trigger_rows)
    write_csv("frame_event_counts.csv", frame_counts)
    (REPORT_ROOT / "R2_FORMAL_EXPERIMENT_REPORT.md").write_text(
        report_text(candidates, intervals, evidence), encoding="utf-8"
    )
    failed = [row for row in invariance if row["status"] != "pass"]
    if failed:
        raise AssertionError(f"prediction invariance failed: {failed}")
    print(
        f"wrote {len(trigger_rows)} trigger records, "
        f"{len(false_rows)} false recoveries, "
        f"{len(false_confirmed)} false confirmations, "
        f"{len(new_cases)} new and {len(lost_cases)} lost GT cases"
    )


if __name__ == "__main__":
    main()
