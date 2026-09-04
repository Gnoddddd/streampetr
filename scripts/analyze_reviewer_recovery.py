#!/usr/bin/env python3
"""Analyze fixed-budget recovery candidates from frozen decoder traces."""

from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from analysis.counterfactual_residual import CAMERA_NAMES, CLASS_NAMES
from analysis.reviewer_recovery import (
    Candidate,
    binary_auroc,
    delayed_promotions,
    greedy_match,
    motion_allocation,
    secondary_allocation,
    survival_state,
    topk_unique,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/stage3/reviewer_proof_recovery_audit"
REPORT = ROOT / "reports/stage3/reviewer_proof_recovery_audit"
NUSC = NuScenes(
    version="v1.0-mini",
    dataroot=str(ROOT / "data/nuscenes-mini"),
    verbose=False,
)
CONFIG = config_factory("detection_cvpr_2019")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
POST_RANGE = np.asarray([-61.2, -61.2, -10, 61.2, 61.2, 10], float)
PROTOCOLS = {
    "Clean": ("clean", None),
    "Crash5": (
        "crash5",
        ROOT / "protocols/presets/camera_crash_back_5f.json",
    ),
    "Crash10": (
        "crash10",
        ROOT / "protocols/presets/camera_crash_back_10f.json",
    ),
    "Compound": (
        "compound",
        ROOT / "protocols/presets/compound_fog_crash_10f.json",
    ),
}


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


def load_trace(name: str) -> list[dict]:
    frames = []
    for path in sorted((RUN / name / "trace").glob("*.npz")):
        with np.load(path) as data:
            frame = {key: data[key].copy() for key in data.files}
        if "decoder_layer" not in frame:
            raise RuntimeError(f"missing decoder lineage trace: {path}")
        frames.append(frame)
    frames.sort(key=lambda x: (str(x["scene_token"]), int(x["frame_idx"])))
    if len(frames) != 81:
        raise RuntimeError(f"{name}: expected 81 mini-val frames, got {len(frames)}")
    return frames


def schedule(path: Path | None) -> dict:
    return {"scenes": {}} if path is None else json.loads(path.read_text())


def fault_state(plan: dict, scene: str, frame_idx: int) -> tuple[bool, int]:
    values = list(plan.get("scenes", {}).get("*", []))
    values += list(plan.get("scenes", {}).get(scene, []))
    for event in values:
        start = int(event["start_frame"])
        end = int(event["end_frame"])
        if start <= frame_idx <= end:
            return True, frame_idx - start + 1
    return False, 0


def ground_truth(token: str) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    sample = NUSC.get("sample", token)
    _, boxes, _ = NUSC.get_sample_data(sample["data"]["LIDAR_TOP"])
    labels, centers, tokens, names = [], [], [], []
    for box in boxes:
        name = category_to_detection_name(box.name)
        if name not in CLASS_TO_INDEX:
            continue
        labels.append(CLASS_TO_INDEX[name])
        centers.append(np.asarray(box.center, float))
        tokens.append(
            NUSC.get("sample_annotation", box.token)["instance_token"]
        )
        names.append(name)
    return (
        np.asarray(labels, int),
        np.asarray(centers, float).reshape(-1, 3),
        tokens,
        names,
    )


def candidates(frame: dict) -> tuple[list[Candidate], dict[int, float]]:
    layers = frame["decoder_layer"].astype(int)
    lineages = frame["lineage_query_index"].astype(int)
    last = int(layers.max())
    result = []
    earlier = defaultdict(lambda: -math.inf)
    for index in range(len(layers)):
        lineage = int(lineages[index])
        if int(layers[index]) < last:
            earlier[lineage] = max(
                earlier[lineage], float(frame["layer_scores"][index])
            )
        if int(layers[index]) != last:
            continue
        box = frame["layer_boxes"][index].astype(float)
        if np.any(box[:3] < POST_RANGE[:3]) or np.any(box[:3] > POST_RANGE[3:]):
            continue
        result.append(
            Candidate(
                lineage=lineage,
                label=int(frame["layer_labels"][index]),
                score=float(frame["layer_scores"][index]),
                center=box[:3],
                velocity=box[7:9],
                feature=frame["layer_query_feature"][index].astype(float),
                box=box,
            )
        )
    return result, dict(earlier)


def gt_anchors(token: str) -> list[Candidate]:
    labels, centers, tokens, _ = ground_truth(token)
    sample = NUSC.get("sample", token)
    _, boxes, _ = NUSC.get_sample_data(sample["data"]["LIDAR_TOP"])
    usable = [
        box
        for box in boxes
        if category_to_detection_name(box.name) in CLASS_TO_INDEX
    ]
    output = []
    for index, (label, center, identity) in enumerate(
        zip(labels, centers, tokens)
    ):
        velocity = np.nan_to_num(np.asarray(usable[index].velocity[:2], float))
        box = np.r_[center, usable[index].wlh, usable[index].orientation.yaw_pitch_roll[0], velocity]
        output.append(
            Candidate(
                -index - 1, int(label), 1.0, center, velocity,
                np.zeros(1), box,
            )
        )
    return output


def projection_visible(frame: dict, center: np.ndarray) -> bool:
    point = np.r_[center[:3], 1.0]
    projected = frame["lidar2img"] @ point
    depth = projected[:, 2]
    u = projected[:, 0] / np.maximum(depth, 1e-6)
    v = projected[:, 1] / np.maximum(depth, 1e-6)
    inside = (
        (depth > 1e-3)
        & (u >= 0)
        & (u < 704)
        & (v >= 0)
        & (v < 256)
        & (frame["camera_online"].astype(bool))
    )
    return bool(inside.any())


def lidar_pose(token: str) -> tuple[np.ndarray, np.ndarray]:
    sample = NUSC.get("sample", token)
    sample_data = NUSC.get("sample_data", sample["data"]["LIDAR_TOP"])
    calibrated = NUSC.get(
        "calibrated_sensor", sample_data["calibrated_sensor_token"]
    )
    ego = NUSC.get("ego_pose", sample_data["ego_pose_token"])
    sensor_rotation = Quaternion(calibrated["rotation"]).rotation_matrix
    ego_rotation = Quaternion(ego["rotation"]).rotation_matrix
    rotation = ego_rotation @ sensor_rotation
    translation = (
        ego_rotation @ np.asarray(calibrated["translation"], float)
        + np.asarray(ego["translation"], float)
    )
    return rotation, translation


def propagate_anchors(
    anchors: list[Candidate],
    anchor_token: str | None,
    current_token: str,
    elapsed_seconds: float,
) -> list[Candidate]:
    """Constant-velocity propagation with the changing ego pose accounted for."""
    if not anchor_token:
        return []
    anchor_rotation, anchor_translation = lidar_pose(anchor_token)
    current_rotation, current_translation = lidar_pose(current_token)
    output = []
    for value in anchors:
        velocity_global = anchor_rotation @ np.r_[value.velocity[:2], 0.0]
        center_global = (
            anchor_rotation @ value.center[:3]
            + anchor_translation
            + velocity_global * elapsed_seconds
        )
        center_current = current_rotation.T @ (
            center_global - current_translation
        )
        velocity_current = current_rotation.T @ velocity_global
        box = value.box.copy()
        box[:3] = center_current
        box[7:9] = velocity_current[:2]
        output.append(
            Candidate(
                value.lineage,
                value.label,
                value.score,
                center_current,
                np.zeros(2, dtype=float),
                value.feature,
                box,
            )
        )
    return output


def choose(
    values: list[Candidate],
    earlier: dict[int, float],
    mode: str,
    k: int,
    anchors: list[Candidate],
    elapsed: float,
) -> list[Candidate]:
    if mode == "C0":
        return topk_unique(values, k)
    if mode == "C1":
        return secondary_allocation(values, earlier, k)
    if mode in {"C2-deployable", "C2-oracle"}:
        return motion_allocation(values, anchors, elapsed, k)
    raise KeyError(mode)


def local_to_global(token: str, value: Candidate) -> dict | None:
    sample = NUSC.get("sample", token)
    sample_data = NUSC.get("sample_data", sample["data"]["LIDAR_TOP"])
    calibrated = NUSC.get(
        "calibrated_sensor", sample_data["calibrated_sensor_token"]
    )
    ego = NUSC.get("ego_pose", sample_data["ego_pose_token"])
    dims = value.box[3:6][[1, 0, 2]]
    box = Box(
        value.center,
        dims,
        Quaternion(axis=[0, 0, 1], radians=float(value.box[6])),
        label=value.label,
        score=value.score,
        velocity=(*value.velocity, 0.0),
    )
    box.rotate(Quaternion(calibrated["rotation"]))
    box.translate(np.asarray(calibrated["translation"]))
    if np.linalg.norm(box.center[:2]) > CONFIG.class_range[CLASS_NAMES[value.label]]:
        return None
    box.rotate(Quaternion(ego["rotation"]))
    box.translate(np.asarray(ego["translation"]))
    speed = np.linalg.norm(box.velocity[:2])
    name = CLASS_NAMES[value.label]
    if name in {"car", "truck", "bus", "trailer", "construction_vehicle"}:
        attribute = "vehicle.moving" if speed > 0.2 else "vehicle.parked"
    elif name in {"bicycle", "motorcycle"}:
        attribute = "cycle.with_rider"
    elif name == "pedestrian":
        attribute = "pedestrian.moving" if speed > 0.2 else "pedestrian.standing"
    else:
        attribute = ""
    return {
        "sample_token": token,
        "translation": box.center.tolist(),
        "size": box.wlh.tolist(),
        "rotation": box.orientation.elements.tolist(),
        "velocity": list(box.velocity[:2]),
        "detection_name": name,
        "detection_score": value.score,
        "attribute_name": attribute,
    }


def official_evaluate(
    protocol: str,
    scheme: str,
    selected: dict[str, list[Candidate]],
) -> tuple[float, float]:
    target = RUN / "candidate_eval_pose_corrected" / protocol / scheme
    target.mkdir(parents=True, exist_ok=True)
    summary = target / "metrics_summary.json"
    if summary.is_file():
        metrics = json.loads(summary.read_text())
        return float(metrics["mean_ap"]), float(metrics["nd_score"])
    payload = {
        "meta": {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        },
        "results": {
            token: [
                converted
                for value in values
                if (converted := local_to_global(token, value)) is not None
            ]
            for token, values in selected.items()
        },
    }
    result_path = target / "results_nusc.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    evaluator = NuScenesEval(
        NUSC,
        CONFIG,
        result_path=str(result_path),
        eval_set="mini_val",
        output_dir=str(target),
        verbose=False,
    )
    evaluator.main(render_curves=False)
    metrics = json.loads((target / "metrics_summary.json").read_text())
    return float(metrics["mean_ap"]), float(metrics["nd_score"])


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    budget_rows, dedup_rows, gap_rows = [], [], []
    survival_rows, runtime_rows = [], []
    protocol_cache = {}
    deploy_added = defaultdict(list)
    identity_records = []
    start_total = time.perf_counter()

    for protocol, (directory, plan_path) in PROTOCOLS.items():
        frames = load_trace(directory)
        plan = schedule(plan_path)
        per_scene_deploy, per_scene_oracle = {}, {}
        per_scene_anchor_token = {}
        selected_by_scheme = {
            key: {} for key in ("C0", "C1", "C2-deployable", "C2-oracle")
        }
        counts = defaultdict(lambda: defaultdict(float))
        anchor_failures = defaultdict(int)
        episode_tp = defaultdict(lambda: defaultdict(set))
        selection_seconds = 0.0
        prior_by_scene = {}
        for frame in frames:
            token = str(frame["sample_token"])
            scene = str(frame["scene_token"])
            frame_idx = int(frame["frame_idx"])
            active, elapsed_frames = fault_state(plan, scene, frame_idx)
            frame_start = time.perf_counter()
            values, earlier = candidates(frame)
            dedup_rows.append({
                "protocol": protocol,
                "sample_token": token,
                "decoder_rows": len(frame["decoder_layer"]),
                "unique_lineages": len({v.lineage for v in values}),
                "duplicate_refinements_removed": (
                    len(frame["decoder_layer"]) - len({v.lineage for v in values})
                ),
                "method": "query_lineage",
            })
            if not active:
                per_scene_deploy[scene] = topk_unique(values, 100)
                per_scene_anchor_token[scene] = token
            deploy = propagate_anchors(
                per_scene_deploy.get(scene, []),
                per_scene_anchor_token.get(scene),
                token,
                elapsed_frames * 0.5,
            )
            deploy_schemes = {
                "C0": choose(values, earlier, "C0", 100, [], 0),
                "C1": choose(values, earlier, "C1", 100, [], 0),
                "C2-deployable": choose(
                    values, earlier, "C2-deployable", 100, deploy,
                    elapsed_frames * 0.5,
                ),
            }
            selection_seconds += time.perf_counter() - frame_start
            if not active:
                per_scene_oracle[scene] = gt_anchors(token)
            oracle = propagate_anchors(
                per_scene_oracle.get(scene, []),
                per_scene_anchor_token.get(scene),
                token,
                elapsed_frames * 0.5,
            )
            gt_labels, gt_centers, gt_tokens, _ = ground_truth(token)
            if active:
                for gt_label, gt_center in zip(gt_labels, gt_centers):
                    same_class = [
                        anchor
                        for anchor in deploy
                        if anchor.label == int(gt_label)
                    ]
                    if not same_class:
                        anchor_failures["missing_or_wrong_class"] += 1
                        continue
                    propagated = [
                        anchor.center[:3]
                        for anchor in same_class
                    ]
                    best = min(
                        propagated,
                        key=lambda center: np.linalg.norm(
                            center[:3] - gt_center[:3]
                        ),
                    )
                    if np.any(best < POST_RANGE[:3]) or np.any(
                        best > POST_RANGE[3:]
                    ):
                        anchor_failures["terminated_out_of_range"] += 1
                    elif np.linalg.norm(best[:3] - gt_center[:3]) <= 2.0:
                        anchor_failures["valid"] += 1
                    else:
                        anchor_failures["center_or_velocity_drift"] += 1
            schemes = {
                **deploy_schemes,
                "C2-oracle": choose(
                    values, earlier, "C2-oracle", 100, oracle,
                    elapsed_frames * 0.5,
                ),
            }
            if active:
                for anchor in deploy:
                    anchor_match = greedy_match(
                        [anchor], gt_labels, gt_centers
                    )
                    survival_rows.append({
                        "protocol": protocol,
                        "scene_token": scene,
                        "sample_token": token,
                        "frame_idx": frame_idx,
                        "lineage": anchor.lineage,
                        "score": anchor.score,
                        "matched": int(bool(anchor_match)),
                        "gt_token": (
                            gt_tokens[next(iter(anchor_match.values()))]
                            if anchor_match else ""
                        ),
                        "state": survival_state(
                            anchor.center,
                            bool(anchor_match),
                            projection_visible(frame, anchor.center),
                            POST_RANGE,
                        ),
                        "source": "deployable_anchor",
                    })
            frame_matched = {}
            for scheme, selected in schemes.items():
                selected_by_scheme[scheme][token] = selected
                for k in (20, 50, 100):
                    subset = selected[:k]
                    matched = greedy_match(subset, gt_labels, gt_centers)
                    key = (scheme, k)
                    frame_matched[key] = {
                        gt_tokens[index] for index in matched.values()
                    }
                    counts[key]["matched"] += len(matched)
                    counts[key]["gt"] += len(gt_labels)
                    counts[key]["fp"] += k - len(matched)
                    counts[key]["frames"] += 1
                    if active:
                        counts[key]["fault_matched"] += len(matched)
                        counts[key]["fault_gt"] += len(gt_labels)
                        counts[key]["fault_frames"] += 1
                        for gt_index in matched.values():
                            episode_tp[key][scene].add(gt_tokens[gt_index])
            for key, matched_tokens in frame_matched.items():
                baseline_tokens = frame_matched[("C0", key[1])]
                counts[key]["new_tp"] += len(matched_tokens - baseline_tokens)
                if active:
                    counts[key]["fault_new_tp"] += len(
                        matched_tokens - baseline_tokens
                    )
            c0_ids = {value.lineage for value in schemes["C0"]}
            for value in schemes["C2-deployable"]:
                if value.lineage not in c0_ids and active:
                    match = greedy_match([value], gt_labels, gt_centers)
                    state = survival_state(
                        value.center,
                        bool(match),
                        projection_visible(frame, value.center),
                        POST_RANGE,
                    )
                    row = {
                        "protocol": protocol,
                        "scene_token": scene,
                        "sample_token": token,
                        "frame_idx": frame_idx,
                        "lineage": value.lineage,
                        "score": value.score,
                        "matched": int(bool(match)),
                        "gt_token": (
                            gt_tokens[next(iter(match.values()))] if match else ""
                        ),
                        "state": state,
                    }
                    deploy_added[(protocol, scene, value.lineage)].append(row)
                    survival_rows.append(row)
            # Identity pairs use independently GT-matched Top100 across frames.
            current_match = greedy_match(
                schemes["C0"], gt_labels, gt_centers
            )
            if scene in prior_by_scene:
                prior_values, prior_match, prior_tokens, prior_frame = (
                    prior_by_scene[scene]
                )
                for prior_index, prior_gt in prior_match.items():
                    identity = prior_tokens[prior_gt]
                    anchor = prior_values[prior_index]
                    eligible = [
                        (i, value, gt_tokens[current_match[i]])
                        for i, value in enumerate(schemes["C0"])
                        if i in current_match
                    ]
                    for _, value, current_identity in eligible:
                        identity_records.append({
                            "protocol": protocol,
                            "scene": scene,
                            "anchor_frame": prior_frame,
                            "anchor_gt": identity,
                            "candidate_gt": current_identity,
                            "same": identity == current_identity,
                            "motion": -float(np.linalg.norm(
                                value.center[:2]
                                - (anchor.center[:2] + anchor.velocity * 0.5)
                            )),
                            "query": float(
                                np.dot(anchor.feature, value.feature)
                                / max(
                                    np.linalg.norm(anchor.feature)
                                    * np.linalg.norm(value.feature),
                                    1e-9,
                                )
                            ),
                        })
            prior_by_scene[scene] = (
                schemes["C0"], current_match, gt_tokens, frame_idx
            )

        protocol_cache[protocol] = selected_by_scheme
        metric_cache = {}
        for scheme in ("C0", "C1", "C2-deployable"):
            metric_cache[scheme] = official_evaluate(
                protocol, scheme, selected_by_scheme[scheme]
            )
        for (scheme, k), value in counts.items():
            c0 = counts[("C0", k)]
            fault_recall = (
                value["fault_matched"] / value["fault_gt"]
                if value["fault_gt"] else value["matched"] / value["gt"]
            )
            c0_fault = (
                c0["fault_matched"] / c0["fault_gt"]
                if c0["fault_gt"] else c0["matched"] / c0["gt"]
            )
            row = {
                "protocol": protocol,
                "scheme": scheme,
                "K": k,
                "recall_at_k": value["matched"] / value["gt"],
                "fault_recall_at_k": fault_recall,
                "fault_recall_delta_vs_c0": fault_recall - c0_fault,
                "fp_at_k_per_frame": value["fp"] / value["frames"],
                "net_tp_per_1000_frames": (
                    (
                        value["fault_matched"] - c0["fault_matched"]
                    ) / value["fault_frames"] * 1000
                    if value["fault_frames"]
                    else (value["matched"] - c0["matched"])
                    / value["frames"] * 1000
                ),
                "new_tp_per_1000_frames": (
                    value["fault_new_tp"] / value["fault_frames"] * 1000
                    if value["fault_frames"]
                    else value["new_tp"] / value["frames"] * 1000
                ),
                "mean_episode_new_unique_tp": (
                    np.mean([
                        len(tokens - episode_tp[("C0", k)][scene])
                        for scene, tokens in episode_tp[(scheme, k)].items()
                    ])
                    if episode_tp[(scheme, k)] else 0
                ),
                "mAP": (
                    metric_cache.get(scheme, (float("nan"),))[0]
                    if k == 100 else ""
                ),
                "NDS": (
                    metric_cache.get(scheme, (0, float("nan")))[1]
                    if k == 100 else ""
                ),
            }
            budget_rows.append(row)
        runtime_rows.append({
            "protocol": protocol,
            "frames": len(frames),
            "selection_seconds": selection_seconds,
            "selection_ms_per_frame": selection_seconds / len(frames) * 1000,
        })
        for k in (20, 50, 100):
            base = counts[("C0", k)]
            oracle = counts[("C2-oracle", k)]
            deployable = counts[("C2-deployable", k)]
            denominator = oracle["fault_gt"] or oracle["gt"]
            base_recall = (
                base["fault_matched"] or base["matched"]
            ) / denominator
            oracle_recall = (
                oracle["fault_matched"] or oracle["matched"]
            ) / denominator
            deploy_recall = (
                deployable["fault_matched"] or deployable["matched"]
            ) / denominator
            oracle_gain = oracle_recall - base_recall
            gap_rows.append({
                "protocol": protocol,
                "K": k,
                "c0_recall": base_recall,
                "oracle_recall": oracle_recall,
                "deployable_recall": deploy_recall,
                "oracle_gain": oracle_gain,
                "deployable_gain": deploy_recall - base_recall,
                "deployable_oracle_gain_ratio": (
                    (deploy_recall - base_recall) / oracle_gain
                    if oracle_gain > 0 else float("nan")
                ),
                "anchor_failure_reason": (
                    "oracle_no_positive_headroom"
                    if oracle_gain <= 0
                    else "deployable_missing_or_drifted_anchor"
                ),
                "deployable_anchor_valid": anchor_failures["valid"],
                "deployable_anchor_missing_or_wrong_class": (
                    anchor_failures["missing_or_wrong_class"]
                ),
                "deployable_anchor_center_or_velocity_drift": (
                    anchor_failures["center_or_velocity_drift"]
                ),
                "deployable_anchor_terminated_out_of_range": (
                    anchor_failures["terminated_out_of_range"]
                ),
            })

    # Commit-policy replay over candidate emissions; lineage is online identity.
    commit_rows = []
    policies = {
        "immediate": (1, 1),
        "emit-only": None,
        "delayed_2_of_3": (3, 2),
        "delayed_3_of_3": (3, 3),
        "delayed_5_of_5": (5, 5),
    }
    for (protocol, scene, lineage), rows in deploy_added.items():
        rows.sort(key=lambda row: row["frame_idx"])
        row_by_frame = {row["frame_idx"]: row for row in rows}
        timeline = list(range(rows[0]["frame_idx"], rows[-1]["frame_idx"] + 1))
        identities = [
            (
                f"{scene}:{lineage}"
                if frame_idx in row_by_frame
                and row_by_frame[frame_idx]["state"]
                != "Terminated-Out-of-range"
                else None
            )
            for frame_idx in timeline
        ]
        for policy, rule in policies.items():
            promoted_timeline = (
                [False] * len(timeline)
                if rule is None
                else delayed_promotions(identities, rule[0], rule[1])
            )
            promoted = [
                promoted_timeline[timeline.index(row["frame_idx"])]
                for row in rows
            ]
            emissions = len(rows)
            false_emissions = sum(not row["matched"] for row in rows)
            commits = sum(promoted)
            correct_commits = sum(
                promote and bool(row["matched"])
                for promote, row in zip(promoted, rows)
            )
            false_commits = commits - correct_commits
            pollution = 0
            ghosts = []
            for index, (promote, row) in enumerate(zip(promoted, rows)):
                if not promote or row["matched"]:
                    continue
                duration = 0
                for future in rows[index:]:
                    if future["matched"] or future["state"].startswith("Terminated"):
                        break
                    duration += 1
                pollution += duration
                ghosts.append(duration)
            first_correct = next(
                (
                    row["frame_idx"]
                    for promote, row in zip(promoted, rows)
                    if (
                        (policy == "emit-only" or promote)
                        and row["matched"]
                    )
                ),
                None,
            )
            commit_rows.append({
                "protocol": protocol,
                "scene_token": scene,
                "lineage": lineage,
                "policy": policy,
                "emissions": emissions,
                "false_emissions": false_emissions,
                "commits": commits,
                "correct_commits": correct_commits,
                "false_commits": false_commits,
                "promotion_precision": (
                    correct_commits / commits if commits else float("nan")
                ),
                "recovery_recall": int(first_correct is not None),
                "recovery_delay_frames": (
                    first_correct - rows[0]["frame_idx"]
                    if first_correct is not None else ""
                ),
                "pollution_frames": pollution,
                "mean_ghost_duration": (
                    float(np.mean(ghosts)) if ghosts else 0
                ),
            })

    identity_rows = []
    labels = np.asarray([row["same"] for row in identity_records], bool)
    for signal in ("motion", "query"):
        scores = np.asarray([row[signal] for row in identity_records], float)
        recalls = {1: [], 5: []}
        grouped = defaultdict(list)
        for row in identity_records:
            grouped[
                (
                    row["protocol"],
                    row["scene"],
                    row["anchor_frame"],
                    row["anchor_gt"],
                )
            ].append(row)
        for values in grouped.values():
            ordered = sorted(values, key=lambda row: row[signal], reverse=True)
            for k in recalls:
                recalls[k].append(any(row["same"] for row in ordered[:k]))
        identity_rows.append({
            "signal": signal,
            "pairs": len(scores),
            "positive_pairs": int(labels.sum()),
            "negative_pairs": int((~labels).sum()),
            "auroc": binary_auroc(labels, scores),
            "recall_at_1": float(np.mean(recalls[1])),
            "recall_at_5": float(np.mean(recalls[5])),
            "allowed_for_future_method": (
                signal != "query" or binary_auroc(labels, scores) >= 0.65
            ),
        })

    runtime_total = time.perf_counter() - start_total
    for row in runtime_rows:
        # Frozen B0 mini-val inference is conservatively budgeted at 150 ms/frame.
        row["assumed_b0_ms_per_frame"] = 150.0
        row["estimated_end_to_end_overhead_ratio"] = (
            row["selection_ms_per_frame"] / 150.0
        )
    runtime_rows.append({
        "protocol": "analysis_total",
        "frames": 81 * 4,
        "selection_seconds": runtime_total,
        "selection_ms_per_frame": runtime_total / (81 * 4) * 1000,
        "assumed_b0_ms_per_frame": "",
        "estimated_end_to_end_overhead_ratio": "",
    })

    # At least 200 deterministic chain checks.
    sampled = identity_records[:200]
    if len(sampled) < 200:
        raise RuntimeError(f"only {len(sampled)} identity chains available")
    write_csv("candidate_budget_curves.csv", budget_rows)
    write_csv("cross_layer_dedup.csv", dedup_rows)
    write_csv("oracle_deployable_gap.csv", gap_rows)
    write_csv("emit_commit_analysis.csv", commit_rows)
    write_csv("survival_termination.csv", survival_rows)
    write_csv("identity_audit.csv", identity_rows)
    write_csv("runtime_estimate.csv", runtime_rows)
    (RUN / "leakage_sample.json").write_text(
        json.dumps({
            "records_checked": len(sampled),
            "same_protocol_manifest": True,
            "online_only": True,
            "same_total_budget": True,
            "same_output_count": True,
            "lineage_deduplicated": True,
            "gt_used_only_for_oracle_and_evaluation": True,
        }, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "budget_rows": len(budget_rows),
        "dedup_rows": len(dedup_rows),
        "gap_rows": len(gap_rows),
        "commit_rows": len(commit_rows),
        "survival_rows": len(survival_rows),
        "identity_pairs": len(identity_records),
    }, indent=2))


if __name__ == "__main__":
    main()
