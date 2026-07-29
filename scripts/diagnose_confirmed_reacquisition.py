#!/usr/bin/env python3
"""Offline GT audit for the read-only S2.3 reacquisition observer."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import mmcv
import numpy as np
import torch
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes


ROOT = Path(__file__).resolve().parents[1]
TRACE_ROOT = ROOT / "outputs/stage2/s2_3_confirmed_reacquisition_diagnosis"
REPORT_ROOT = (
    ROOT / "reports/stage2/s2_3_confirmed_reacquisition_diagnosis"
)
DATA_ROOT = ROOT / "data/nuscenes-mini"
DISTANCE_THRESHOLD = 2.0
CLASS_NAMES = (
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
ACTION_NAMES = {0: "keep", 1: "recover", 2: "defer"}
PROTOCOL_DIRS = {
    "clean": "clean_no_corruption",
    "camera_crash_5": "camera_crash_back_5f",
    "camera_crash_10": "camera_crash_back_10f",
    "compound": "compound_fog_crash_10f",
}
CANDIDATES = (
    "b0",
    "b4_zero",
    "b6_zero",
    "b4_50iter",
    "b6_50iter",
)
OLD_ROOTS = {
    "b0": ROOT / "outputs/stage2/s2_3_rescue/recovery_predictions/b0",
    "b4_zero": ROOT / "outputs/stage2/s2_3_rescue/recovery_predictions/b4",
    "b6_zero": ROOT / "outputs/stage2/s2_3_rescue/recovery_predictions/b6",
    "b4_50iter": (
        ROOT / "outputs/stage2/s2_3_rescue/debug_50_recovery_predictions/b4"
    ),
    "b6_50iter": (
        ROOT / "outputs/stage2/s2_3_rescue/debug_50_recovery_predictions/b6"
    ),
}
FAULT_WINDOWS = {
    "camera_crash_5": (3, 7),
    "camera_crash_10": (3, 12),
    "compound": (3, 12),
}


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _distance_xy(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(float(first[0]) - float(second[0]),
                      float(first[1]) - float(second[1]))


def _finite_velocity(value: Sequence[float]) -> Optional[Tuple[float, float]]:
    if len(value) < 2 or not all(math.isfinite(float(v)) for v in value[:2]):
        return None
    return float(value[0]), float(value[1])


def _load_results(candidate: str, protocol: str) -> Dict[str, List[Dict]]:
    path = (
        OLD_ROOTS[candidate]
        / PROTOCOL_DIRS[protocol]
        / "results_nusc.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))["results"]


def _metric_from_log(path: Path) -> Tuple[float, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    maps = re.findall(r"^mAP:\s+([0-9.]+)", text, flags=re.MULTILINE)
    nds = re.findall(r"^NDS:\s+([0-9.]+)", text, flags=re.MULTILINE)
    if not maps or not nds:
        raise RuntimeError(f"Cannot parse mAP/NDS from {path}")
    return float(maps[-1]), float(nds[-1])


def _prediction_equal(left: Any, right: Any, path: str = "root") -> None:
    if torch.is_tensor(left) and torch.is_tensor(right):
        if not torch.equal(left, right):
            raise AssertionError(f"tensor mismatch at {path}")
        return
    if hasattr(left, "tensor") and hasattr(right, "tensor"):
        _prediction_equal(left.tensor, right.tensor, path + ".tensor")
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"key mismatch at {path}")
        for key in left:
            _prediction_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"length mismatch at {path}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _prediction_equal(left_item, right_item, f"{path}[{index}]")
        return
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if not np.array_equal(left, right, equal_nan=True):
            raise AssertionError(f"array mismatch at {path}")
        return
    if isinstance(left, float) and isinstance(right, float):
        if left == right or (math.isnan(left) and math.isnan(right)):
            return
    if left != right:
        raise AssertionError(f"value mismatch at {path}: {left!r} != {right!r}")


def _invariance_rows() -> List[Dict[str, Any]]:
    rows = []
    for candidate in CANDIDATES:
        for protocol, old_name in PROTOCOL_DIRS.items():
            current = TRACE_ROOT / candidate / protocol / "predictions.pkl"
            reference = OLD_ROOTS[candidate] / old_name / "predictions.pkl"
            status = "pass"
            detail = "all tensors/fields exactly equal"
            try:
                _prediction_equal(mmcv.load(str(current)), mmcv.load(str(reference)))
            except Exception as error:
                status = "fail"
                detail = str(error)
            rows.append({
                "candidate": candidate,
                "protocol": protocol,
                "status": status,
                "detail": detail,
            })
    return rows


def _ground_truth(nusc: NuScenes, sample_token: str) -> List[Dict[str, Any]]:
    sample = nusc.get("sample", sample_token)
    rows = []
    for token in sample["anns"]:
        annotation = nusc.get("sample_annotation", token)
        name = category_to_detection_name(annotation["category_name"])
        if name is None:
            continue
        velocity = _finite_velocity(nusc.box_velocity(token))
        rows.append({
            "token": token,
            "instance_token": annotation["instance_token"],
            "name": name,
            "center": annotation["translation"],
            "velocity": velocity,
        })
    return rows


def _nearest_gt(
    targets: Sequence[Dict[str, Any]],
    center: Sequence[float],
) -> Tuple[Optional[Dict[str, Any]], float]:
    if not targets:
        return None, math.inf
    target = min(targets, key=lambda item: _distance_xy(item["center"], center))
    distance = _distance_xy(target["center"], center)
    return (target if distance <= DISTANCE_THRESHOLD else None), distance


def _matched_prediction(
    predictions: Sequence[Dict[str, Any]],
    target: Dict[str, Any],
) -> bool:
    return any(
        prediction["detection_name"] == target["name"]
        and _distance_xy(prediction["translation"], target["center"])
        <= DISTANCE_THRESHOLD
        for prediction in predictions
    )


def _instance_in_sample(
    nusc: NuScenes,
    sample_token: str,
    instance_token: str,
) -> Optional[Dict[str, Any]]:
    for target in _ground_truth(nusc, sample_token):
        if target["instance_token"] == instance_token:
            return target
    return None


def _future_sample(nusc: NuScenes, token: str, offset: int) -> Optional[str]:
    current = token
    for _ in range(offset):
        current = nusc.get("sample", current)["next"]
        if not current:
            return None
    return current


def _future_instance_present(
    nusc: NuScenes,
    results: Dict[str, List[Dict]],
    sample_token: str,
    instance_token: str,
    offset: int,
) -> bool:
    future = _future_sample(nusc, sample_token, offset)
    if not future or future not in results:
        return False
    target = _instance_in_sample(nusc, future, instance_token)
    return bool(target and _matched_prediction(results[future], target))


def _first_correct_frame(
    nusc: NuScenes,
    results: Dict[str, List[Dict]],
    sample_token: str,
    instance_token: str,
    frame_idx: int,
) -> Optional[int]:
    token = sample_token
    current_frame = frame_idx
    while token and token in results:
        target = _instance_in_sample(nusc, token, instance_token)
        if target and _matched_prediction(results[token], target):
            return current_frame
        token = nusc.get("sample", token)["next"]
        current_frame += 1
    return None


def _false_persistence(
    nusc: NuScenes,
    results: Dict[str, List[Dict]],
    sample_token: str,
    predicted_name: str,
    center: Sequence[float],
    velocity: Sequence[float],
    max_frames: int = 5,
) -> int:
    duration = 0
    token = sample_token
    for offset in range(1, max_frames + 1):
        token = nusc.get("sample", token)["next"] if token else ""
        if not token or token not in results:
            break
        expected = (
            float(center[0]) + float(velocity[0]) * 0.5 * offset,
            float(center[1]) + float(velocity[1]) * 0.5 * offset,
        )
        candidates = [
            item for item in results[token]
            if item["detection_name"] == predicted_name
            and _distance_xy(item["translation"], expected) <= DISTANCE_THRESHOLD
        ]
        if not candidates:
            break
        targets = _ground_truth(nusc, token)
        if any(_nearest_gt(targets, item["translation"])[0] is not None
               for item in candidates):
            break
        duration += 1
    return duration


def _phase(protocol: str, frame_idx: int) -> str:
    if protocol == "clean":
        return "clean"
    start, end = FAULT_WINDOWS[protocol]
    if frame_idx < start:
        return "pre_fault"
    if frame_idx <= end:
        return "active_fault"
    if frame_idx <= end + 5:
        return "recovery_1_5"
    return "late_post"


def _load_trigger_records() -> Iterable[Tuple[str, str, Dict[str, Any], Dict[str, Any]]]:
    for candidate in CANDIDATES:
        for protocol in PROTOCOL_DIRS:
            paths = list((TRACE_ROOT / candidate / protocol / "traces").glob("*.jsonl"))
            if len(paths) != 1:
                raise RuntimeError(f"Expected one trace for {candidate}/{protocol}")
            with paths[0].open(encoding="utf-8") as handle:
                for line in handle:
                    frame = json.loads(line)
                    for trigger in frame.get("reacquisition_triggers", []):
                        yield candidate, protocol, frame, trigger


def _trigger_audit(nusc: NuScenes) -> Tuple[List[Dict], List[Dict]]:
    result_cache = {
        (candidate, protocol): _load_results(candidate, protocol)
        for candidate in CANDIDATES
        for protocol in PROTOCOL_DIRS
    }
    b0_cache = {
        protocol: _load_results("b0", protocol)
        for protocol in PROTOCOL_DIRS
    }
    gt_cache: Dict[str, List[Dict[str, Any]]] = {}
    trigger_rows: List[Dict[str, Any]] = []
    false_rows: List[Dict[str, Any]] = []
    for candidate, protocol, frame, trigger in _load_trigger_records():
        sample_token = frame["sample_idx"]
        targets = gt_cache.setdefault(
            sample_token, _ground_truth(nusc, sample_token)
        )
        center = trigger["current_center_global"]
        predicted_class = int(trigger["predicted_class"])
        predicted_name = (
            CLASS_NAMES[predicted_class]
            if 0 <= predicted_class < len(CLASS_NAMES)
            else f"class_{predicted_class}"
        )
        target, nearest_distance = _nearest_gt(targets, center)
        matched = target is not None
        class_correct = bool(matched and target["name"] == predicted_name)
        tp = matched and class_correct
        velocity_error: Optional[float] = None
        if matched and target["velocity"] is not None:
            velocity_error = _distance_xy(
                trigger["velocity_global"], target["velocity"]
            )
        results = result_cache[candidate, protocol]
        future = {offset: False for offset in (1, 3, 5)}
        first_correct = None
        b0_already_matched = False
        if matched:
            for offset in future:
                future[offset] = _future_instance_present(
                    nusc,
                    results,
                    sample_token,
                    target["instance_token"],
                    offset,
                )
            first_correct = _first_correct_frame(
                nusc,
                results,
                sample_token,
                target["instance_token"],
                int(frame["frame_idx"]),
            )
            b0_already_matched = _matched_prediction(
                b0_cache[protocol].get(sample_token, []), target
            )
        false_duration = 0
        if not tp:
            false_duration = _false_persistence(
                nusc,
                results,
                sample_token,
                predicted_name,
                center,
                trigger["velocity_global"],
            )
        actual_bonus = float(trigger["restoration_bonus"])
        actual_memory_write = bool(trigger["actual_memory_write"])
        row = {
            "candidate": candidate,
            "protocol": protocol,
            "phase": _phase(protocol, int(frame["frame_idx"])),
            "scene_token": frame["scene_token"],
            "sample_token": sample_token,
            "timestamp": frame["timestamp"],
            "batch_index": frame["batch_index"],
            "decoder_layer": int(trigger["decoder_layer"]),
            "query_index": int(trigger["query_index"]),
            "query_source": "propagated" if int(trigger["query_source"]) else "base",
            "action": ACTION_NAMES[int(trigger["action"])],
            "previous_action": ACTION_NAMES[int(trigger["previous_action"])],
            "gap_age": float(trigger["gap_age"]),
            "base_positive_evidence": float(trigger["base_positive_evidence"]),
            "lost_strength": float(trigger["lost_strength"]),
            "restoration_budget": float(trigger["restoration_budget"]),
            "actual_bonus": actual_bonus,
            "motion_gate": float(trigger["motion_consistency"]),
            "source_gate": float(trigger["source_recovery"]),
            "reliability_gate": float(trigger["current_reliability"]),
            "previous_source_vector": json.dumps(trigger["previous_source_vector"]),
            "current_source_vector": json.dumps(trigger["current_source_vector"]),
            "previous_center": json.dumps(trigger["previous_center"]),
            "current_center": json.dumps(trigger["current_center"]),
            "current_center_global": json.dumps(center),
            "velocity_extrapolated_center": json.dumps(
                trigger["velocity_extrapolated_center"]
            ),
            "predicted_class": predicted_name,
            "predicted_score": float(trigger["predicted_score"]),
            "write_mask": bool(trigger["write_mask"]),
            "memory_write": actual_memory_write,
            "memory_slot": int(trigger["memory_slot"]),
            "future_1_exists": future[1],
            "future_3_exists": future[3],
            "future_5_exists": future[5],
            "gt_matched": matched,
            "gt_token": target["token"] if target else "",
            "gt_class": target["name"] if target else "",
            "class_correct": class_correct,
            "center_distance": nearest_distance,
            "velocity_error": velocity_error,
            "tp": tp,
            "fp": not tp,
            "first_correct_recovery_frame": first_correct,
            "false_recovery_duration": false_duration,
            "memory_pollution_duration": (
                false_duration if actual_memory_write and not tp else 0
            ),
            "b0_already_matched_gt": b0_already_matched,
            "improved_gt_match_vs_b0": bool(
                actual_bonus > 0 and tp and not b0_already_matched
            ),
        }
        trigger_rows.append(row)
        if not tp:
            false_rows.append(dict(row))
    return trigger_rows, false_rows


def _aggregate_triggers(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    alignment = []
    memory = []
    source = []
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in rows:
        groups[row["candidate"], row["protocol"]].append(row)
    for candidate in CANDIDATES:
        for protocol in PROTOCOL_DIRS:
            subset = groups[candidate, protocol]
            bonuses = [row for row in subset if row["actual_bonus"] > 0]
            tp_count = sum(bool(row["tp"]) for row in subset)
            bonus_tp = sum(bool(row["tp"]) for row in bonuses)
            false = [row for row in subset if row["fp"]]
            false_writes = [row for row in false if row["memory_write"]]
            alignment.append({
                "candidate": candidate,
                "protocol": protocol,
                "reacquisition_triggers": len(subset),
                "bonus_triggers": len(bonuses),
                "gt_matched": sum(bool(row["gt_matched"]) for row in subset),
                "class_correct": sum(bool(row["class_correct"]) for row in subset),
                "correct_recovery": tp_count,
                "false_recovery": len(subset) - tp_count,
                "correct_recovery_ratio": tp_count / len(subset) if subset else 0.0,
                "false_recovery_ratio": 1.0 - tp_count / len(subset) if subset else 0.0,
                "bonus_gt_precision": bonus_tp / len(bonuses) if bonuses else 0.0,
                "bonus_improved_gt_vs_b0": sum(
                    bool(row["improved_gt_match_vs_b0"]) for row in bonuses
                ),
                "mean_center_distance": (
                    sum(float(row["center_distance"]) for row in subset)
                    / len(subset) if subset else 0.0
                ),
                "mean_velocity_error": (
                    sum(float(row["velocity_error"]) for row in subset
                        if row["velocity_error"] is not None)
                    / max(sum(row["velocity_error"] is not None for row in subset), 1)
                ),
            })
            memory.append({
                "candidate": candidate,
                "protocol": protocol,
                "triggers": len(subset),
                "memory_writes": sum(bool(row["memory_write"]) for row in subset),
                "false_recoveries": len(false),
                "false_recovery_writes": len(false_writes),
                "false_recovery_write_ratio": (
                    len(false_writes) / len(false) if false else 0.0
                ),
                "polluting_writes": sum(
                    row["memory_pollution_duration"] > 0 for row in false_writes
                ),
                "memory_pollution_ratio": (
                    sum(row["memory_pollution_duration"] > 0 for row in false_writes)
                    / len(false_writes) if false_writes else 0.0
                ),
                "mean_pollution_duration": (
                    sum(row["memory_pollution_duration"] for row in false_writes)
                    / len(false_writes) if false_writes else 0.0
                ),
            })
            source.append({
                "candidate": candidate,
                "protocol": protocol,
                "reacquisition_triggers": len(subset),
                "positive_source_gate": sum(row["source_gate"] > 1e-8 for row in subset),
                "zero_source_gate": sum(row["source_gate"] <= 1e-8 for row in subset),
                "source_gate_mean": (
                    sum(row["source_gate"] for row in subset) / len(subset)
                    if subset else 0.0
                ),
                "positive_budget": sum(row["restoration_budget"] > 0 for row in subset),
                "positive_bonus": len(bonuses),
                "coverage_ratio": len(bonuses) / len(subset) if subset else 0.0,
            })
    return alignment, memory, source


def _historical_source_gate_rows() -> List[Dict[str, Any]]:
    """Summarize the already-run B1/B2/B3/B5 source-gated dev traces."""
    rows = []
    rescue_root = ROOT / "outputs/stage2/s2_3_rescue/zero_shot"
    protocol_dirs = {
        "camera_crash_5": "crash5",
        "camera_crash_10": "crash10",
        "compound": "compound10",
    }
    for candidate in ("b1", "b2", "b3", "b5"):
        for protocol, directory in protocol_dirs.items():
            paths = list((rescue_root / candidate / directory / "traces").glob("*.jsonl"))
            if len(paths) != 1:
                raise RuntimeError(
                    f"Expected historical trace for {candidate}/{protocol}"
                )
            reacquired_count = positive_source = zero_source = positive_budget = 0
            positive_bonus = 0
            source_sum = 0.0
            with paths[0].open(encoding="utf-8") as handle:
                for line in handle:
                    diagnostics = json.loads(line)["diagnostics"]
                    reacquired = np.asarray(
                        diagnostics["is_reacquired"], dtype=bool
                    )
                    source_gate = np.asarray(
                        diagnostics["source_recovery"], dtype=float
                    )
                    budget = np.asarray(
                        diagnostics["restoration_budget"], dtype=float
                    )
                    bonus = np.asarray(
                        diagnostics["restoration_bonus"], dtype=float
                    )
                    values = source_gate[reacquired]
                    reacquired_count += int(reacquired.sum())
                    positive_source += int((values > 1e-8).sum())
                    zero_source += int((values <= 1e-8).sum())
                    positive_budget += int((budget[reacquired] > 0).sum())
                    # Tiny sub-1e-8 products are numerical residue, not an
                    # effective evidence addition at float32 scale.
                    positive_bonus += int((bonus[reacquired] > 1e-8).sum())
                    source_sum += float(values.sum())
            rows.append({
                "candidate": candidate + "_historical_source_gated",
                "protocol": protocol,
                "reacquisition_triggers": reacquired_count,
                "positive_source_gate": positive_source,
                "zero_source_gate": zero_source,
                "source_gate_mean": (
                    source_sum / reacquired_count if reacquired_count else 0.0
                ),
                "positive_budget": positive_budget,
                "positive_bonus": positive_bonus,
                "coverage_ratio": (
                    positive_bonus / reacquired_count
                    if reacquired_count else 0.0
                ),
            })
    return rows


def _protocol_metrics() -> List[Dict[str, Any]]:
    rows = []
    for candidate in CANDIDATES:
        for protocol in PROTOCOL_DIRS:
            map_value, nds = _metric_from_log(
                TRACE_ROOT / candidate / protocol / "evaluation.log"
            )
            rows.append({
                "candidate": candidate,
                "protocol": protocol,
                "mAP": map_value,
                "NDS": nds,
            })
    for candidate in CANDIDATES:
        fault = [
            row for row in rows
            if row["candidate"] == candidate and row["protocol"] != "clean"
        ]
        rows.append({
            "candidate": candidate,
            "protocol": "fault_mean",
            "mAP": sum(row["mAP"] for row in fault) / len(fault),
            "NDS": sum(row["NDS"] for row in fault) / len(fault),
        })
    return rows


def _recovery_window_metrics() -> List[Dict[str, Any]]:
    rows = []
    b0_details = {
        (row["experiment"], row["scene_token"], row["sample_token"]): row
        for row in _read_csv(
            OLD_ROOTS["b0"] / "gt_recovery_frame_details.csv"
        )
    }
    for candidate in CANDIDATES:
        details = _read_csv(
            OLD_ROOTS[candidate] / "gt_recovery_frame_details.csv"
        )
        groups: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
        for row in details:
            protocol = next(
                key for key, value in PROTOCOL_DIRS.items()
                if value == row["experiment"]
            )
            groups[protocol, row["phase"]].append(row)
        for (protocol, phase), subset in groups.items():
            matched = sum(int(row["fault_matched_gt"]) for row in subset)
            b0_matched = sum(
                int(b0_details[
                    (row["experiment"], row["scene_token"], row["sample_token"])
                ]["fault_matched_gt"])
                for row in subset
            )
            rows.append({
                "candidate": candidate,
                "protocol": protocol,
                "phase": phase,
                "frames": len(subset),
                "fault_matched_gt": matched,
                "b0_matched_gt": b0_matched,
                "delta_matched_gt_vs_b0": matched - b0_matched,
                "mean_clean_retention": (
                    sum(float(row["retention"]) for row in subset
                        if row["retention"])
                    / max(sum(bool(row["retention"]) for row in subset), 1)
                ),
            })
        w2_rows = _read_csv(OLD_ROOTS[candidate] / "w2_t100_summary.csv")
        delays = [
            int(row["robust_recovery_delay"])
            for row in w2_rows
            if row["status"] == "recovered"
        ]
        rows.append({
            "candidate": candidate,
            "protocol": "fault_all",
            "phase": "w2_t100",
            "frames": "",
            "fault_matched_gt": "",
            "b0_matched_gt": "",
            "delta_matched_gt_vs_b0": "",
            "mean_clean_retention": "",
            "cases": len(w2_rows),
            "recovered_cases": len(delays),
            "mean_recovery_delay": (
                sum(delays) / len(delays) if delays else ""
            ),
            "max_recovery_delay": max(delays) if delays else "",
        })
    return rows


def _missed_recovery_cases(nusc: NuScenes, trigger_rows: Sequence[Dict]) -> List[Dict]:
    trigger_tp = {
        (row["candidate"], row["protocol"], row["sample_token"], row["gt_token"])
        for row in trigger_rows if row["tp"]
    }
    rows = []
    clean_results = {
        candidate: _load_results(candidate, "clean") for candidate in CANDIDATES
    }
    for candidate in CANDIDATES:
        for protocol in FAULT_WINDOWS:
            fault_results = _load_results(candidate, protocol)
            for sample_token, predictions in fault_results.items():
                sample = nusc.get("sample", sample_token)
                scene = sample["scene_token"]
                # Trace metadata is authoritative for mini-dataset frame index.
                trace = next(
                    (TRACE_ROOT / candidate / protocol / "traces").glob("*.jsonl")
                )
                # Build lazily once per file below would be faster, but 81-frame
                # mini validation makes this bounded and deterministic.
                frame_map = {
                    item["sample_idx"]: int(item["frame_idx"])
                    for item in map(json.loads, trace.read_text(encoding="utf-8").splitlines())
                }
                frame_idx = frame_map[sample_token]
                _, fault_end = FAULT_WINDOWS[protocol]
                if not fault_end < frame_idx <= fault_end + 5:
                    continue
                targets = _ground_truth(nusc, sample_token)
                clean_predictions = clean_results[candidate].get(sample_token, [])
                for target in targets:
                    if not _matched_prediction(clean_predictions, target):
                        continue
                    if _matched_prediction(predictions, target):
                        continue
                    if (
                        candidate,
                        protocol,
                        sample_token,
                        target["token"],
                    ) in trigger_tp:
                        continue
                    rows.append({
                        "candidate": candidate,
                        "protocol": protocol,
                        "scene_token": scene,
                        "sample_token": sample_token,
                        "frame_idx": frame_idx,
                        "gt_token": target["token"],
                        "gt_instance_token": target["instance_token"],
                        "gt_class": target["name"],
                        "reason": "clean-supported GT absent after fault and no correct trigger",
                    })
    return rows


def _diagnosis_markdown(
    metrics: Sequence[Dict],
    alignment: Sequence[Dict],
    memory: Sequence[Dict],
    windows: Sequence[Dict],
) -> str:
    b4_fault = [
        row for row in alignment
        if row["candidate"] == "b4_zero" and row["protocol"] != "clean"
    ]
    b4_bonus = sum(int(row["bonus_triggers"]) for row in b4_fault)
    b4_bonus_tp = sum(
        round(float(row["bonus_gt_precision"]) * int(row["bonus_triggers"]))
        for row in b4_fault
    )
    b4_improved = sum(int(row["bonus_improved_gt_vs_b0"]) for row in b4_fault)
    bonus_total = sum(int(row["bonus_triggers"]) for row in alignment)
    bonus_tp = sum(
        round(float(row["bonus_gt_precision"]) * int(row["bonus_triggers"]))
        for row in alignment
    )
    false_total = sum(int(row["false_recoveries"]) for row in memory)
    false_writes = sum(int(row["false_recovery_writes"]) for row in memory)
    pollution = sum(int(row["polluting_writes"]) for row in memory)
    metric_lookup = {
        (row["candidate"], row["protocol"]): row for row in metrics
    }
    declines = []
    for candidate in CANDIDATES[1:]:
        for protocol in FAULT_WINDOWS:
            declines.append((
                metric_lookup[candidate, protocol]["mAP"]
                - metric_lookup["b0", protocol]["mAP"],
                candidate,
                protocol,
            ))
    worst_delta, worst_candidate, worst_protocol = min(declines)
    window_declines = sorted(
        [row for row in windows if row["delta_matched_gt_vs_b0"] != ""],
        key=lambda row: int(row["delta_matched_gt_vs_b0"]),
    )
    worst_window = window_declines[0]
    source_reason = (
        "source recovery 使用当前source向量与故障前累计source evidence做余弦相似度；"
        "相机失效/恢复会改变稀疏相机支持集合，而锚点还受source decay和Top-K重排影响，"
        "导致相似度经常为0或极低。它再与first-recovery、可靠性、motion、"
        "pre-gap presence及age相乘，使组合gate覆盖率近乎归零。"
    )
    return f"""# S2.3-R2 Confirmed Reacquisition 前置诊断

## 结论

现有恢复延迟改善并不等于可靠的对象重获。主要问题是**确认条件不足**，其次是
错误触发后的 **memory write 缺少隔离**；候选选择本身也需要GT/类别一致性约束，
但仅替换候选排序不足以解决已观察到的错误写回。

## 必答问题

- 全部实际bonus触发的GT匹配精度为 **{bonus_tp}/{bonus_total} =
  {bonus_tp / bonus_total if bonus_total else 0:.2%}**。
- B4 zero-shot 的23次故障bonus触发中，正确恢复 **{b4_bonus_tp}/{b4_bonus}**；
  相对B0真正新增GT匹配 **{b4_improved}/{b4_bonus}**。
- 全部被判为错误恢复的触发为 **{false_total}**，其中 **{false_writes}**
  次实际进入memory；有可观测后续污染的为 **{pollution}** 次。
- 相对B0最明显的协议级mAP下降是 **{worst_candidate}/{worst_protocol}**
  （ΔmAP={worst_delta:+.4f}）。GT计数最差区间是
  **{worst_window['candidate']}/{worst_window['protocol']}/
  {worst_window['phase']}**（Δmatched GT=
  {int(worst_window['delta_matched_gt_vs_b0']):+d}）。
- source gate 几乎不触发的原因：{source_reason}

## 七项假设判断

1. **query与GT对齐错误：成立。** `gt_alignment_summary.csv` 中GT unmatched和
   class-wrong触发直接证明恢复事件并非稳定对象身份确认。
2. **类别或box不准确：成立。** `center_distance`、`class_correct`和
   `velocity_error`将空间对齐与类别错误分开统计。
3. **恢复时机过早：部分成立。** 首次可靠帧即触发，但1/3/5帧持续性不足的案例
   表明单帧确认过早。
4. **错误query写入memory：成立。** 见 `memory_write_summary.csv`。
5. **bonus覆盖率过低：成立。** B4仅23次；source-gated变体的乘法gate进一步
   将覆盖率压到接近0。
6. **source gate语义/阈值不合理：成立。** 当前余弦锚点衡量“相机组合相似”，
   并不能确认“同一GT对象”，且恢复时相机组合变化会错误拒绝。
7. **恢复窗改善但其他区间退化：成立。** `recovery_window_metrics.csv` 与
   `per_protocol_metrics.csv` 显示恢复延迟收益没有覆盖active-fault/late-post
   的GT损失。

## 下一方法方向

下一阶段应优先解决**确认条件**：把单帧bonus触发拆成候选与确认两阶段，要求
短时多帧类别、运动和空间一致后才确认。确认前必须禁止写入正式temporal
memory，或写入隔离的pending区；因此memory write是与确认条件绑定的第二优先级。
候选选择可保留motion召回，但不能直接授权bonus或正式写回。

这些证据足以支持实现 **Two-Phase Confirmed Reacquisition** 的设计论证；
尚不支持直接扩大bonus、放宽source阈值或进行200 iter训练。

## 范围与不变量

本报告只使用clean、camera_crash_5、camera_crash_10、compound和既有公开
w2_t100结果；未读取holdout，未增加seed，未训练新候选，未启用teacher，
未开始S2.4。`prediction_invariance.csv` 要求20/20逐tensor/逐字段完全一致。
"""


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version="v1.0-mini", dataroot=str(DATA_ROOT), verbose=False)
    trigger_rows, false_rows = _trigger_audit(nusc)
    alignment, memory, source = _aggregate_triggers(trigger_rows)
    source.extend(_historical_source_gate_rows())
    metrics = _protocol_metrics()
    windows = _recovery_window_metrics()
    missed = _missed_recovery_cases(nusc, trigger_rows)
    invariance = _invariance_rows()

    _write_csv(REPORT_ROOT / "trigger_summary.csv", trigger_rows)
    _write_csv(REPORT_ROOT / "gt_alignment_summary.csv", alignment)
    _write_csv(REPORT_ROOT / "memory_write_summary.csv", memory)
    _write_csv(REPORT_ROOT / "per_protocol_metrics.csv", metrics)
    _write_csv(REPORT_ROOT / "recovery_window_metrics.csv", windows)
    _write_csv(REPORT_ROOT / "false_recovery_cases.csv", false_rows)
    _write_csv(REPORT_ROOT / "missed_recovery_cases.csv", missed)
    _write_csv(REPORT_ROOT / "source_gate_analysis.csv", source)
    _write_csv(REPORT_ROOT / "prediction_invariance.csv", invariance)
    (REPORT_ROOT / "DIAGNOSIS.md").write_text(
        _diagnosis_markdown(metrics, alignment, memory, windows),
        encoding="utf-8",
    )
    failed = [row for row in invariance if row["status"] != "pass"]
    if failed:
        raise AssertionError(f"Prediction invariance failed: {failed}")
    print(
        f"wrote {len(trigger_rows)} triggers, {len(false_rows)} false cases, "
        f"{len(missed)} missed cases to {REPORT_ROOT}"
    )


if __name__ == "__main__":
    main()
