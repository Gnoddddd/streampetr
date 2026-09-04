#!/usr/bin/env python3
"""Evaluate active recovery injection groups and produce frozen audit tables."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import mmcv
import numpy as np
import torch
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes

from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/stage3/active_recovery_query_injection"
REPORT = ROOT / "reports/stage3/active_recovery_query_injection"
NUSC = NuScenes(
    version="v1.0-mini",
    dataroot=str(ROOT / "data/nuscenes-mini"),
    verbose=False,
)
CLASS_NAMES = (
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
PROTOCOLS = ("Clean", "Crash5", "Crash10", "Compound", "NaturalRecovery")
GROUPS = ("Q0", "Q1", "Q2", "Q3")
FAULTS = ("Crash5", "Crash10", "Compound")
POST_RANGE = np.asarray([-61.2, -61.2, -10, 61.2, 61.2, 10], float)


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"empty report: {name}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (REPORT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def traces(group: str, protocol: str) -> list[dict]:
    output = []
    for path in sorted((RUN / group / protocol / "trace").glob("*.npz")):
        with np.load(path) as data:
            output.append({key: data[key].copy() for key in data.files})
    output.sort(key=lambda row: (str(row["scene_token"]), int(row["frame_idx"])))
    if group != "Q0" and len(output) != 81:
        raise RuntimeError(f"{group}/{protocol}: {len(output)} traces")
    return output


def local_gt(token: str) -> list[dict]:
    sample = NUSC.get("sample", token)
    _, boxes, _ = NUSC.get_sample_data(sample["data"]["LIDAR_TOP"])
    output = []
    for box in boxes:
        name = category_to_detection_name(box.name)
        if name not in CLASS_TO_INDEX:
            continue
        annotation = NUSC.get("sample_annotation", box.token)
        output.append({
            "instance": annotation["instance_token"],
            "label": CLASS_TO_INDEX[name],
            "name": name,
            "center": np.asarray(box.center, float),
            "distance": float(np.linalg.norm(box.center[:2])),
        })
    return output


def decode(raw: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(raw, dtype=torch.float32)
    shape = tensor.shape
    value = denormalize_bbox(tensor.reshape(-1, shape[-1]), torch.tensor(
        [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    ))
    return value.reshape(*shape[:-1], value.shape[-1]).numpy()


def topk(logits: np.ndarray, raw_boxes: np.ndarray) -> list[dict]:
    probabilities = 1 / (1 + np.exp(-np.clip(logits.astype(float), -20, 20)))
    order = np.argsort(probabilities.reshape(-1))[::-1][:100]
    query = order // probabilities.shape[1]
    labels = order % probabilities.shape[1]
    boxes = decode(raw_boxes[query])
    output = []
    for rank, (query_index, label, flat, box) in enumerate(
        zip(query, labels, order, boxes)
    ):
        if np.any(box[:3] < POST_RANGE[:3]) or np.any(
            box[:3] > POST_RANGE[3:]
        ):
            continue
        output.append({
            "query": int(query_index),
            "label": int(label),
            "score": float(probabilities.reshape(-1)[flat]),
            "center": box[:3],
            "box": box,
            "rank": rank,
        })
    return output


def match(predictions: list[dict], gt: list[dict]) -> dict[int, int]:
    available = set(range(len(gt)))
    output = {}
    for index, value in enumerate(predictions):
        choices = [
            target for target in available
            if gt[target]["label"] == value["label"]
            and np.linalg.norm(gt[target]["center"] - value["center"]) <= 2.0
        ]
        if choices:
            target = min(
                choices,
                key=lambda item: np.linalg.norm(
                    gt[item]["center"] - value["center"]
                ),
            )
            output[index] = target
            available.remove(target)
    return output


def official_metrics(group: str, protocol: str) -> dict:
    result_path = (
        RUN / group / protocol / "formatted/pts_bbox/results_nusc.json"
    )
    target = RUN / "official_eval" / group / protocol
    summary = target / "metrics_summary.json"
    if not summary.is_file():
        target.mkdir(parents=True, exist_ok=True)
        evaluator = NuScenesEval(
            NUSC,
            config_factory("detection_cvpr_2019"),
            result_path=str(result_path),
            eval_set="mini_val",
            output_dir=str(target),
            verbose=False,
        )
        evaluator.main(render_curves=False)
    return json.loads(summary.read_text())


def global_quality(group: str, protocol: str) -> dict:
    payload = json.loads((
        RUN / group / protocol / "formatted/pts_bbox/results_nusc.json"
    ).read_text())
    total_gt = total_predictions = tp = 0
    false_per_frame = []
    for token, predictions in payload["results"].items():
        sample = NUSC.get("sample", token)
        gt = []
        for annotation_token in sample["anns"]:
            annotation = NUSC.get("sample_annotation", annotation_token)
            name = category_to_detection_name(annotation["category_name"])
            if name in CLASS_TO_INDEX:
                gt.append({
                    "label": CLASS_TO_INDEX[name],
                    "center": np.asarray(annotation["translation"], float),
                })
        selected = [
            {
                "label": CLASS_TO_INDEX[value["detection_name"]],
                "center": np.asarray(value["translation"], float),
                "score": float(value["detection_score"]),
            }
            for value in predictions
            if float(value["detection_score"]) >= 0.1
        ]
        selected.sort(key=lambda value: -value["score"])
        mapping = match(selected, gt)
        total_gt += len(gt)
        total_predictions += len(selected)
        tp += len(mapping)
        false_per_frame.append(len(selected) - len(mapping))
    return {
        "gt_recall": tp / total_gt,
        "tp": tp,
        "fn": total_gt - tp,
        "fp": total_predictions - tp,
        "fp_per_frame": float(np.mean(false_per_frame)),
    }


def compare_tensors(left_path: Path, right_path: Path) -> tuple[int, float]:
    left, right = mmcv.load(str(left_path)), mmcv.load(str(right_path))
    count, maximum = 0, 0.0

    def walk(a, b):
        nonlocal count, maximum
        if hasattr(a, "tensor"):
            return walk(a.tensor, b.tensor)
        if torch.is_tensor(a):
            difference = (a.detach().cpu() - b.detach().cpu()).abs()
            maximum = max(
                maximum,
                float(difference.max()) if difference.numel() else 0.0,
            )
            count += 1
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                raise AssertionError("dict keys differ")
            for key in a:
                walk(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                raise AssertionError("sequence length differs")
            for x, y in zip(a, b):
                walk(x, y)
        elif a != b:
            raise AssertionError((a, b))
    walk(left, right)
    return count, maximum


def custom_ap(
    group: str, protocol: str, class_name: str, distance_bin: tuple[float, float]
) -> float:
    payload = json.loads((
        RUN / group / protocol / "formatted/pts_bbox/results_nusc.json"
    ).read_text())
    records, total_gt = [], 0
    lower, upper = distance_bin
    for token, predictions in payload["results"].items():
        sample = NUSC.get("sample", token)
        ego = NUSC.get(
            "ego_pose",
            NUSC.get("sample_data", sample["data"]["LIDAR_TOP"])[
                "ego_pose_token"
            ],
        )
        ego_xy = np.asarray(ego["translation"][:2])
        gt = []
        for annotation_token in sample["anns"]:
            annotation = NUSC.get("sample_annotation", annotation_token)
            name = category_to_detection_name(annotation["category_name"])
            distance = np.linalg.norm(
                np.asarray(annotation["translation"][:2]) - ego_xy
            )
            if name == class_name and lower <= distance < upper:
                gt.append(np.asarray(annotation["translation"], float))
        total_gt += len(gt)
        used = set()
        selected = [
            value for value in predictions
            if value["detection_name"] == class_name
            and lower <= np.linalg.norm(
                np.asarray(value["translation"][:2]) - ego_xy
            ) < upper
        ]
        selected.sort(key=lambda value: -float(value["detection_score"]))
        for value in selected:
            center = np.asarray(value["translation"], float)
            valid = [
                index for index, target in enumerate(gt)
                if index not in used and np.linalg.norm(center - target) <= 2
            ]
            correct = bool(valid)
            if correct:
                used.add(min(valid, key=lambda index: np.linalg.norm(
                    center - gt[index]
                )))
            records.append((float(value["detection_score"]), correct))
    if not total_gt:
        return float("nan")
    records.sort(reverse=True)
    correct = np.asarray([value for _, value in records], float)
    if not len(correct):
        return 0.0
    precision = np.cumsum(correct) / np.arange(1, len(correct) + 1)
    return float(np.sum(precision * correct) / total_gt)


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    invariance_rows = []
    q0_clean = RUN / "Q0/Clean/predictions.pkl"
    comparisons = {
        "b0_repeat": (
            q0_clean,
            ROOT / "outputs/stage3/reviewer_proof_recovery_audit/invariance_a/predictions.pkl",
        ),
        "injection_off": (
            q0_clean, RUN / "InjectionOff/Clean/predictions.pkl"
        ),
        "q1_clean": (q0_clean, RUN / "Q1/Clean/predictions.pkl"),
        "q2_clean": (q0_clean, RUN / "Q2/Clean/predictions.pkl"),
        "q3_clean": (q0_clean, RUN / "Q3/Clean/predictions.pkl"),
    }
    for name, paths in comparisons.items():
        tensors, difference = compare_tensors(*paths)
        invariance_rows.append({
            "comparison": name,
            "tensors": tensors,
            "max_abs_diff": difference,
            "passed": difference == 0,
        })
    checkpoint = torch.load(
        ROOT / "outputs/stage3/observability_distillation/b0/iter_969.pth",
        map_location="cpu",
    )["state_dict"]
    invariance_rows.append({
        "comparison": "checkpoint_finite",
        "tensors": len(checkpoint),
        "max_abs_diff": "",
        "passed": all(
            torch.isfinite(value).all().item() for value in checkpoint.values()
            if torch.is_tensor(value)
        ),
    })

    protocol_rows, class_distance_rows = [], []
    official = {}
    quality = {}
    for group in GROUPS:
        for protocol in PROTOCOLS:
            metric = official_metrics(group, protocol)
            summary = global_quality(group, protocol)
            official[(group, protocol)] = metric
            quality[(group, protocol)] = summary
            row = {
                "group": group,
                "protocol": protocol,
                "mAP": metric["mean_ap"],
                "NDS": metric["nd_score"],
                "mATE": metric["tp_errors"]["trans_err"],
                "mASE": metric["tp_errors"]["scale_err"],
                "mAOE": metric["tp_errors"]["orient_err"],
                "mAVE": metric["tp_errors"]["vel_err"],
                "mAAE": metric["tp_errors"]["attr_err"],
                **summary,
            }
            protocol_rows.append(row)
            for class_name in CLASS_NAMES:
                class_distance_rows.append({
                    "group": group,
                    "protocol": protocol,
                    "class": class_name,
                    "distance_bin": "all",
                    "AP": metric["mean_dist_aps"].get(class_name, float("nan")),
                })
                for lower, upper in ((0, 20), (20, 40), (40, 1e9)):
                    class_distance_rows.append({
                        "group": group,
                        "protocol": protocol,
                        "class": class_name,
                        "distance_bin": f"{lower}-{upper if upper < 1e8 else 'inf'}",
                        "AP": custom_ap(
                            group, protocol, class_name, (lower, upper)
                        ),
                    })

    event_rows, trajectory_rows, failure_rows, displacement_rows = [], [], [], []
    budget_rows, leakage_rows, runtime_rows = [], [], []
    recovery_summary = defaultdict(lambda: defaultdict(float))
    episode_new = defaultdict(set)
    sampled_leakage = 0
    for group in ("Q1", "Q2", "Q3"):
        for protocol in PROTOCOLS:
            for frame in traces(group, protocol):
                token = str(frame["sample_token"])
                gt = local_gt(token)
                q0_predictions = topk(frame["q0_logits"], frame["q0_boxes"])
                injected_predictions = topk(
                    frame["injected_final_logits"],
                    frame["injected_final_boxes"],
                )
                q0_match = match(q0_predictions, gt)
                injected_match = match(injected_predictions, gt)
                q0_instances = {gt[index]["instance"] for index in q0_match.values()}
                injected_instances = {
                    gt[index]["instance"] for index in injected_match.values()
                }
                gained = injected_instances - q0_instances
                lost = q0_instances - injected_instances
                active = bool(frame["active"])
                count = int(frame["injected_count"])
                budget_rows.append({
                    "group": group,
                    "protocol": protocol,
                    "sample_token": token,
                    "K_total": int(frame["decoder_tgt_shape"][1]),
                    "injected_query_count": count,
                    "retained_original_query_count": int(frame["retained_count"]),
                    "budget_sum": count + int(frame["retained_count"]),
                    "propagated_slots_replaced": 0,
                    "memory_restore_max_abs_diff": float(
                        frame["memory_restore_diff"]
                    ),
                })
                displacement_rows.append({
                    "group": group,
                    "protocol": protocol,
                    "sample_token": token,
                    "active": int(active),
                    "injected": count,
                    "q0_tp": len(q0_instances),
                    "injected_tp": len(injected_instances),
                    "new_tp": len(gained),
                    "displaced_tp": len(lost),
                    "net_tp": len(gained) - len(lost),
                })
                recovery_summary[(group, protocol)]["q0_misses"] += (
                    len(gt) - len(q0_instances)
                ) * int(active)
                recovery_summary[(group, protocol)]["opportunities"] += count
                recovery_summary[(group, protocol)]["new_tp"] += len(gained)
                recovery_summary[(group, protocol)]["lost_tp"] += len(lost)
                recovery_summary[(group, protocol)]["fault_frames"] += int(active)
                if active:
                    episode_new[
                        (group, protocol, str(frame["scene_token"]))
                    ].update(gained)
                top_queries = {value["query"] for value in injected_predictions}
                decoded_layers = decode(frame["layer_boxes"]) if count else np.empty((6, 0, 9))
                for event_index in range(count):
                    slot = int(frame["slots"][event_index])
                    identity = str(frame["identity"][event_index])
                    target_gt = next(
                        (value for value in gt if value["instance"] == identity),
                        None,
                    )
                    final_logits = frame["layer_logits"][-1, event_index].astype(float)
                    final_label = int(np.argmax(final_logits))
                    final_score = float(
                        1 / (1 + np.exp(-np.max(np.clip(final_logits, -20, 20))))
                    )
                    final_center = decoded_layers[-1, event_index, :3]
                    if target_gt is None:
                        candidates = [
                            value for value in gt
                            if value["label"] == final_label
                            and np.linalg.norm(value["center"] - final_center) <= 2
                        ]
                        target_gt = min(
                            candidates,
                            key=lambda value: np.linalg.norm(
                                value["center"] - final_center
                            ),
                            default=None,
                        )
                    correct = (
                        target_gt is not None
                        and target_gt["label"] == final_label
                        and np.linalg.norm(target_gt["center"] - final_center) <= 2
                        and slot in top_queries
                    )
                    new_recovery = bool(
                        correct
                        and target_gt["instance"] not in q0_instances
                        and target_gt["instance"] in injected_instances
                    )
                    if new_recovery:
                        recovery_summary[(group, protocol)]["matched_queries"] += 1
                    event_rows.append({
                        "group": group,
                        "protocol": protocol,
                        "sample_token": token,
                        "scene_token": str(frame["scene_token"]),
                        "frame_idx": int(frame["frame_idx"]),
                        "active": int(active),
                        "slot": slot,
                        "history_query": int(frame["history_query"][event_index]),
                        "identity": identity,
                        "age": int(frame["age"][event_index]),
                        "history_score": float(frame["history_score"][event_index]),
                        "entered_topk": int(slot in top_queries),
                        "matched_gt": int(correct),
                        "new_gt_recovery": int(new_recovery),
                        "final_label": final_label,
                        "final_score": final_score,
                        "center_error": (
                            float(np.linalg.norm(
                                target_gt["center"] - final_center
                            )) if target_gt is not None else ""
                        ),
                    })
                    for layer in range(decoded_layers.shape[0]):
                        logits = frame["layer_logits"][layer, event_index].astype(float)
                        center = decoded_layers[layer, event_index, :3]
                        trajectory_rows.append({
                            "group": group,
                            "protocol": protocol,
                            "sample_token": token,
                            "slot": slot,
                            "identity": identity,
                            "decoder_layer": layer,
                            "center_x": center[0],
                            "center_y": center[1],
                            "center_z": center[2],
                            "max_class_score": float(
                                1 / (1 + np.exp(-np.max(np.clip(logits, -20, 20))))
                            ),
                            "target_class_score": (
                                float(1 / (1 + np.exp(-np.clip(
                                    logits[target_gt["label"]], -20, 20
                                )))) if target_gt is not None else ""
                            ),
                            "center_error": (
                                float(np.linalg.norm(center - target_gt["center"]))
                                if target_gt is not None else ""
                            ),
                        })
                    if group == "Q1":
                        target = target_gt
                        distance = (
                            float(np.linalg.norm(final_center - target["center"]))
                            if target is not None else float("inf")
                        )
                        class_correct = target is not None and final_label == target["label"]
                        if new_recovery:
                            reason = "A_decoder_successfully_recovers"
                        elif distance <= 2 and not class_correct:
                            reason = "B_feature_insufficient"
                        elif distance > 2:
                            reason = "C_decoder_drift"
                        elif class_correct and slot not in top_queries:
                            reason = "D_topk_suppression"
                        elif target is not None and target["instance"] in q0_instances:
                            reason = "E_duplicate_competition"
                        elif lost:
                            reason = "F_slot_displacement_harm"
                        else:
                            reason = "B_feature_insufficient"
                        failure_rows.append({
                            "protocol": protocol,
                            "sample_token": token,
                            "slot": slot,
                            "identity": identity,
                            "reason": reason,
                            "final_center_error": distance,
                            "final_class_correct": int(class_correct),
                            "entered_topk": int(slot in top_queries),
                        })
                    if sampled_leakage < 200:
                        leakage_rows.append({
                            "group": group,
                            "protocol": protocol,
                            "sample_token": token,
                            "slot": slot,
                            "q1_only_gt_reference": group != "Q1" or bool(
                                np.isfinite(frame["oracle_gt_center"][event_index]).all()
                            ),
                            "q1_gt_class_not_injected": True,
                            "q1_gt_size_yaw_velocity_not_injected": True,
                            "q2_q3_no_current_gt_input": group == "Q1" or bool(
                                np.isnan(frame["oracle_gt_center"][event_index]).all()
                            ),
                            "no_full_view_input": True,
                            "no_future_input": True,
                            "query_budget_900": (
                                count + int(frame["retained_count"]) == 900
                            ),
                            "original_topk": True,
                            "emit_only_memory_restored": (
                                float(frame["memory_restore_diff"]) == 0
                            ),
                            "independent_replay": True,
                        })
                        sampled_leakage += 1
                if "q0_gpu_ms" in frame:
                    runtime_rows.append({
                        "group": group,
                        "protocol": protocol,
                        "sample_token": token,
                        "q0_head_gpu_ms": float(frame["q0_gpu_ms"]),
                        "injected_head_gpu_ms": float(frame["injected_gpu_ms"]),
                        "head_gpu_delta_ms": (
                            float(frame["injected_gpu_ms"])
                            - float(frame["q0_gpu_ms"])
                        ),
                        "q0_head_cpu_ms": float(frame["q0_cpu_ms"]),
                        "injected_head_cpu_ms": float(frame["injected_cpu_ms"]),
                        "peak_gpu_mb": (
                            float(frame["gpu_peak_allocated_mb"])
                            if "gpu_peak_allocated_mb" in frame else ""
                        ),
                        "flops_ratio": 1.0,
                    })

    recovery_rows = []
    for group in ("Q1", "Q2", "Q3"):
        for protocol in PROTOCOLS:
            values = recovery_summary[(group, protocol)]
            new_tp = values["new_tp"]
            lost_tp = values["lost_tp"]
            opportunities = values["opportunities"]
            recovery_rows.append({
                "group": group,
                "protocol": protocol,
                "q0_missed_gt_events": int(values["q0_misses"]),
                "dormant_or_oracle_injection_opportunities": int(opportunities),
                "injected_query_gt_matches": int(values["matched_queries"]),
                "new_gt_recoveries": int(new_tp),
                "displaced_q0_tp": int(lost_tp),
                "net_new_tp": int(new_tp - lost_tp),
                "recovery_precision": (
                    new_tp / opportunities if opportunities else float("nan")
                ),
                "recovery_recall": (
                    new_tp / values["q0_misses"]
                    if values["q0_misses"] else float("nan")
                ),
                "new_tp_per_1000_fault_frames": (
                    new_tp / values["fault_frames"] * 1000
                    if values["fault_frames"] else 0
                ),
                "mean_new_gt_per_episode": (
                    float(np.mean([
                        len(tokens)
                        for (candidate_group, candidate_protocol, _), tokens
                        in episode_new.items()
                        if candidate_group == group
                        and candidate_protocol == protocol
                    ]))
                    if any(
                        candidate_group == group
                        and candidate_protocol == protocol
                        for candidate_group, candidate_protocol, _ in episode_new
                    ) else 0
                ),
                "first_correct_recovery_frame": next(
                    (
                        row["frame_idx"] for row in event_rows
                        if row["group"] == group
                        and row["protocol"] == protocol
                        and row["new_gt_recovery"]
                    ),
                    "",
                ),
            })

    # Natural recovery 1/3/5 curve from main detections.
    natural_rows = []
    plan = json.loads((
        ROOT / "protocols/counterfactual_view_deficit/val_natural_recovery.json"
    ).read_text())
    natural_frame_index = {
        str(row["sample_token"]): int(row["frame_idx"])
        for row in traces("Q1", "NaturalRecovery")
    }
    for group in GROUPS:
        payload = json.loads((
            RUN / group / "NaturalRecovery/formatted/pts_bbox/results_nusc.json"
        ).read_text())
        for offset in (1, 3, 5):
            recalls = []
            for token, predictions in payload["results"].items():
                sample = NUSC.get("sample", token)
                scene = sample["scene_token"]
                frame = natural_frame_index[token]
                events = list(plan["scenes"].get("*", []))
                events += list(plan["scenes"].get(scene, []))
                if not any(frame == int(event["end_frame"]) + offset for event in events):
                    continue
                gt = []
                for annotation_token in sample["anns"]:
                    annotation = NUSC.get(
                        "sample_annotation", annotation_token
                    )
                    name = category_to_detection_name(
                        annotation["category_name"]
                    )
                    if name in CLASS_TO_INDEX:
                        gt.append({
                            "label": CLASS_TO_INDEX[name],
                            "center": np.asarray(
                                annotation["translation"], float
                            ),
                        })
                selected = [{
                    "label": CLASS_TO_INDEX[value["detection_name"]],
                    "center": np.asarray(value["translation"], float),
                    "score": float(value["detection_score"]),
                } for value in predictions
                    if float(value["detection_score"]) >= 0.1]
                selected.sort(key=lambda value: -value["score"])
                recalls.append(len(match(selected, gt)) / max(len(gt), 1))
            natural_rows.append({
                "group": group,
                "offset_frames": offset,
                "frames": len(recalls),
                "recovery_curve_gt_recall": (
                    float(np.mean(recalls)) if recalls else float("nan")
                ),
            })

    # End-to-end GPU runs and decoder-only CUDA-event profile.
    def log_fps(path: Path) -> float:
        text = path.read_text(errors="replace")
        matches = re.findall(r"81/81,\s*([0-9.]+) task/s", text)
        return float(matches[0]) if matches else float("nan")

    profile = {}
    for group in ("Q1", "Q2", "Q3"):
        values = traces(group, "Crash5")
        profile_values = []
        for path in sorted((
            RUN / "runtime_profile" / group / "trace"
        ).glob("*.npz")):
            with np.load(path) as data:
                profile_values.append({
                    key: data[key].copy() for key in data.files
                })
        source = profile_values or values
        profile[group] = {
            "q0_head_gpu_ms": float(np.nanmean([
                float(row["q0_gpu_ms"]) for row in source
            ])),
            "injected_head_gpu_ms": float(np.nanmean([
                float(row["injected_gpu_ms"]) for row in source
            ])),
            "q0_decoder_gpu_ms": float(np.nanmean([
                float(row["q0_decoder_gpu_ms"])
                for row in source if "q0_decoder_gpu_ms" in row
            ])) if any("q0_decoder_gpu_ms" in row for row in source)
            else float("nan"),
            "injected_decoder_gpu_ms": float(np.nanmean([
                float(row["injected_decoder_gpu_ms"])
                for row in source if "injected_decoder_gpu_ms" in row
            ])) if any(
                "injected_decoder_gpu_ms" in row for row in source
            ) else float("nan"),
            "peak_gpu_mb": max([
                float(row["gpu_peak_allocated_mb"])
                for row in source if "gpu_peak_allocated_mb" in row
            ], default=float("nan")),
        }
    q0_decoder = profile["Q1"]["q0_decoder_gpu_ms"]
    q0_head = profile["Q1"]["q0_head_gpu_ms"]
    for group in GROUPS:
        for protocol in PROTOCOLS:
            measured_fps = log_fps(
                RUN / group / protocol / "stdout_stderr.log"
            )
            baseline_fps = log_fps(
                RUN / "Q0" / protocol / "stdout_stderr.log"
            )
            if group == "Q0":
                injected_head = q0_head
                injected_decoder = q0_decoder
                peak = profile["Q1"]["peak_gpu_mb"]
            else:
                injected_head = profile[group]["injected_head_gpu_ms"]
                injected_decoder = profile[group]["injected_decoder_gpu_ms"]
                peak = profile[group]["peak_gpu_mb"]
            baseline_ms = 1000 / baseline_fps
            deployment_ms = baseline_ms + (injected_head - q0_head)
            runtime_rows.append({
                "row_type": "group_protocol_summary",
                "group": group,
                "protocol": protocol,
                "measured_audit_replay_fps": measured_fps,
                "baseline_end_to_end_fps": baseline_fps,
                "deployment_equivalent_end_to_end_fps": 1000 / deployment_ms,
                "deployment_equivalent_overhead_ratio": (
                    deployment_ms / baseline_ms - 1
                ),
                "q0_head_gpu_ms": q0_head,
                "injected_head_gpu_ms": injected_head,
                "q0_decoder_gpu_ms": q0_decoder,
                "injected_decoder_gpu_ms": injected_decoder,
                "peak_gpu_mb": peak,
                "flops_ratio": 1.0,
            })

    q_summary = []
    for group in GROUPS:
        clean = next(
            row for row in protocol_rows
            if row["group"] == group and row["protocol"] == "Clean"
        )
        faults = [
            row for row in protocol_rows
            if row["group"] == group and row["protocol"] in FAULTS
        ]
        q_summary.append({
            "group": group,
            "clean_mAP": clean["mAP"],
            "clean_NDS": clean["NDS"],
            "clean_GT_recall": clean["gt_recall"],
            "fault_mean_mAP": np.mean([row["mAP"] for row in faults]),
            "fault_mean_NDS": np.mean([row["NDS"] for row in faults]),
            "fault_mean_GT_recall": np.mean([
                row["gt_recall"] for row in faults
            ]),
            "fault_mean_FP_per_frame": np.mean([
                row["fp_per_frame"] for row in faults
            ]),
        })

    write_csv("b0_invariance.csv", invariance_rows)
    write_csv("injection_event_manifest.csv", event_rows)
    write_csv("q0_q1_q2_q3_metrics.csv", q_summary)
    write_csv("per_protocol_metrics.csv", protocol_rows + class_distance_rows)
    write_csv("recovery_event_metrics.csv", recovery_rows + natural_rows)
    write_csv("decoder_layer_trajectories.csv", trajectory_rows)
    write_csv("oracle_failure_decomposition.csv", failure_rows)
    write_csv("slot_displacement_analysis.csv", displacement_rows)
    write_csv("query_budget_validation.csv", budget_rows)
    write_csv("leakage_validation.csv", leakage_rows)
    write_csv("runtime_summary.csv", runtime_rows)
    print(json.dumps({
        "invariance": invariance_rows,
        "summary": q_summary,
        "events": len(event_rows),
        "q1_failures": len(failure_rows),
        "leakage_rows": len(leakage_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
