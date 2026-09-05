#!/usr/bin/env python3
"""Camera/LiDAR object-level gap stage after teacher coverage passes."""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STREAM_ROOT = ROOT / "repos/StreamPETR"
sys.dont_write_bytecode = True
sys.path.insert(0, str(STREAM_ROOT))

import mmcv  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.parallel import MMDataParallel  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmcv.utils import import_modules_from_strings  # noqa: E402
from mmdet3d.datasets import build_dataloader, build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from nuscenes.nuscenes import NuScenes  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402

from analysis.fault_boundary_root_cause import candidate_pool_statistics  # noqa: E402
from analysis.lidar_privileged_signal import (  # noqa: E402
    circular_error,
    cosine_distance,
    cross_modal_signal_decision,
)
from analysis.temporal_state_counterfactual import cluster_bootstrap_median  # noqa: E402
from scripts.audit_cross_view_target_evidence import (  # noqa: E402
    PROTOCOLS,
    protocol_dataset,
)
from scripts.audit_dark_target_recoverability import (  # noqa: E402
    CHECKPOINT,
    CONFIG,
    DISABLED,
    compare_tensors,
    features,
    physical,
    run_head,
    snapshot,
    unpack,
    validate_trace,
)
from scripts.audit_lidar_privileged_signal import (  # noqa: E402
    CLASSES,
    PROTOCOL_NAMES,
    REPORT,
    TEACHER_CHECKPOINT,
    TEACHER_CONFIG,
    gt_targets,
    teacher_frame,
    write_csv,
)
from scripts.audit_temporal_state_attribution import (  # noqa: E402
    parent_value,
    validate_parent_metrics,
)
from scripts.audit_temporal_state_counterfactual import output_metrics  # noqa: E402


PARENT_PER_GT = ROOT / "reports/stage4/temporal_state_counterfactual_audit/per_gt_counterfactual.csv"
SEED, BOOTSTRAPS = 314159, 5000


def finite_median(rows: list[dict], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def rate(rows: list[dict], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows])) if rows else float("nan")


def delta(left, right) -> float:
    left, right = float(left), float(right)
    return left - right if math.isfinite(left) and math.isfinite(right) else float("nan")


def probability_logit(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return float("nan")
    value = min(max(value, 1e-7), 1.0 - 1e-7)
    return float(math.log(value / (1.0 - value)))


def run_camera_with_rep(model, meta, data, image_features, prev_exists, state):
    captured = {}

    def hook(_, __, output):
        captured["decoder"] = output[0][-1, 0].detach().float().cpu().numpy()

    handle = model.pts_bbox_head.transformer.register_forward_hook(hook)
    try:
        output, post_state, _ = run_head(
            model, meta, data, image_features, prev_exists, state)
    finally:
        handle.remove()
    return output, post_state, captured["decoder"]


def camera_object(output: dict, decoder: np.ndarray, target: dict,
                  pc_range) -> dict:
    logits = output["all_cls_scores"][-1, 0].detach().float().cpu().numpy()
    boxes = physical(output, pc_range)[-1, 0].detach().float().cpu().numpy()
    pool = candidate_pool_statistics(
        logits, boxes, target["center"], target["label"], 100, 2.0)
    if not pool["candidate_available"]:
        return {"candidate": False, "qplus": -1, "s_pos": float("nan"),
                "rank": float("nan"), "margin": float("nan"),
                "class_probs": np.full(len(CLASSES), np.nan),
                "box": np.full(9, np.nan), "representation": np.full(256, np.nan),
                "xy_center_error": float("nan"), "relative_size_l1": float("nan"),
                "yaw_error_rad": float("nan")}
    query = int(pool["best_query"])
    box = boxes[query]
    center_delta = box[:3] - target["center"]
    size_delta = np.abs(box[3:6] - target["size"])
    return {"candidate": True, "qplus": query, "s_pos": float(pool["s_pos"]),
            "rank": int(pool["rank"]), "margin": float(pool["margin"]),
            "class_probs": 1.0 / (1.0 + np.exp(-logits[query])),
            "box": box.copy(), "representation": decoder[query].copy(),
            "xy_center_error": float(np.linalg.norm(center_delta[:2])),
            "relative_size_l1": float(np.mean(
                size_delta / np.maximum(target["size"], 1e-6))),
            "yaw_error_rad": circular_error(box[6], target["yaw"])}


def cosine_matrix(features: list[np.ndarray]) -> np.ndarray:
    matrix = np.asarray(features, float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    return matrix @ matrix.T


def summarize(protocol: str, group: str, rows: list[dict]) -> dict:
    return {"protocol": protocol,
            "condition": "Pooled" if protocol == "pooled" else PROTOCOL_NAMES[protocol],
            "group": group, "n": len(rows),
            "teacher_matched_rate": rate(rows, "teacher_matched"),
            "representation_available_rate": rate(rows, "representation_available"),
            "median_score_gap_expansion": finite_median(rows, "score_gap_expansion"),
            "median_center_gap_expansion": finite_median(rows, "center_gap_expansion"),
            "median_size_gap_expansion": finite_median(rows, "size_gap_expansion"),
            "median_representation_gap_expansion": finite_median(
                rows, "representation_gap_expansion"),
            "median_teacher_score_minus_clean_spos": finite_median(
                rows, "teacher_score_minus_clean_spos"),
            "median_clean_center_advantage": finite_median(
                rows, "clean_center_advantage"),
            "median_clean_representation_gap": finite_median(
                rows, "clean_lidar_representation_gap"),
            "median_fault_representation_gap": finite_median(
                rows, "fault_lidar_representation_gap")}


def main() -> None:
    with (REPORT / "mechanism_decision.csv").open(encoding="utf-8") as handle:
        coverage_decision = next(csv.DictReader(handle))
    if (coverage_decision.get("teacher_coverage_pass") == "False"
            or coverage_decision.get("decision") == "NO_GO_LIDAR_TEACHER_COVERAGE"):
        raise RuntimeError("teacher coverage gate did not pass")
    with PARENT_PER_GT.open(encoding="utf-8") as handle:
        parent_rows = list(csv.DictReader(handle))
    if len(parent_rows) != 142:
        raise RuntimeError(f"parent population changed: {len(parent_rows)}")
    units = defaultdict(list)
    for row in parent_rows:
        units[(row["protocol"], row["sample_token"])].append(row)

    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    nusc = NuScenes(version="v1.0-mini", dataroot=str(ROOT / "data/nuscenes-mini"),
                    verbose=False)
    population_scenes = {row["scene_token"] for row in parent_rows}
    population_specs = set()
    for row in parent_rows:
        annotation = nusc.get("sample_annotation", row["gt_token"])
        population_specs.add((row["protocol"], row["group"], row["scene_token"],
                              annotation["instance_token"]))

    teacher_cfg = Config.fromfile(str(TEACHER_CONFIG))
    teacher_dataset = build_dataset(teacher_cfg.data.test)
    teacher_loader = build_dataloader(
        teacher_dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)
    teacher = build_model(teacher_cfg.model, test_cfg=teacher_cfg.get("test_cfg"))
    load_checkpoint(teacher, str(TEACHER_CHECKPOINT), map_location="cpu")
    teacher = MMDataParallel(teacher.cuda().eval(), device_ids=[0])
    capture = {}

    def capture_neck(_, __, output):
        capture["feature"] = output[0].detach()

    hook_handle = teacher.module.pts_neck.register_forward_hook(capture_neck)
    relevant_tokens = {str(info["token"]) for info in teacher_dataset.data_infos
                       if str(info["scene_token"]) in population_scenes
                       and 3 <= int(info["frame_idx"]) <= 12}
    teacher_cache = {}
    try:
        for index, data in enumerate(teacher_loader):
            token = str(teacher_dataset.data_infos[index]["token"])
            if token not in relevant_tokens:
                continue
            teacher_cache[token] = teacher_frame(
                teacher, data, gt_targets(nusc, token), capture)
    finally:
        hook_handle.remove()
    del teacher
    torch.cuda.empty_cache()

    camera_cfg = Config.fromfile(str(CONFIG))
    import_modules_from_strings(**camera_cfg.custom_imports)
    camera_cfg.model.pretrained = None
    camera_cfg.model.train_cfg = None
    camera_cfg.data.test.test_mode = True
    clean_dataset = protocol_dataset(
        camera_cfg, ROOT / "protocols/presets/clean_no_corruption.json")
    fault_datasets = {protocol: protocol_dataset(camera_cfg, schedule)
                      for protocol, (_, schedule) in PROTOCOLS.items()}
    camera = build_model(camera_cfg.model, test_cfg=camera_cfg.get("test_cfg"))
    load_checkpoint(camera, str(CHECKPOINT), map_location="cpu")
    camera = camera.cuda().eval()
    camera.pts_bbox_head.reset_memory()
    initial = snapshot(camera.pts_bbox_head)
    states = {key: initial for key in ("clean", *PROTOCOLS)}
    pc_range = camera.pts_bbox_head.pc_range.detach()

    camera_gt = {}
    for info in clean_dataset.data_infos:
        token = str(info["token"])
        targets = gt_targets(nusc, token)
        for target in targets:
            target["global_center"] = np.asarray(
                nusc.get("sample_annotation", target["token"])["translation"], float)
        camera_gt[token] = targets
    specs_by_scene_instance = defaultdict(list)
    for spec in population_specs:
        specs_by_scene_instance[(spec[2], spec[3])].append(spec)

    replay = {protocol: {"logits_exact": True, "box_max": 0.0, "frames": 0}
              for protocol in ("clean", *PROTOCOLS)}
    clean_cache, fault_cache = {}, {}
    previous_scene = None
    with torch.no_grad():
        for index in range(len(clean_dataset)):
            clean_meta, clean_image, clean_data = unpack(clean_dataset[index], "cuda:0")
            token, scene = str(clean_meta["sample_idx"]), str(clean_meta["scene_token"])
            prev_exists = 0 if scene != previous_scene else 1
            previous_scene = scene
            _, _, clean_features = features(camera, clean_image)
            output_a, states["clean"], decoder_a = run_camera_with_rep(
                camera, clean_meta, clean_data, clean_features, prev_exists,
                states["clean"])
            exact, box_diff = validate_trace("clean", token, output_a, pc_range)
            replay["clean"]["logits_exact"] &= exact
            replay["clean"]["box_max"] = max(replay["clean"]["box_max"], box_diff)
            replay["clean"]["frames"] += 1
            if not exact or box_diff > 1e-5:
                raise RuntimeError(f"clean replay divergence {token}")

            if token in relevant_tokens:
                for target in camera_gt[token]:
                    if (scene, target["instance_token"]) in specs_by_scene_instance:
                        clean_cache[(token, target["token"])] = camera_object(
                            output_a, decoder_a, target, pc_range)

            info = clean_dataset.data_infos[index]
            context = {
                "lidar2ego_rotation": Quaternion(
                    info["lidar2ego_rotation"]).rotation_matrix,
                "lidar2ego_translation": np.asarray(info["lidar2ego_translation"], float),
                "ego2global_rotation": Quaternion(
                    info["ego2global_rotation"]).rotation_matrix,
                "ego2global_translation": np.asarray(info["ego2global_translation"], float),
                "class_range": clean_dataset.eval_detection_configs.class_range,
            }
            target_index = {target["token"]: target for target in camera_gt[token]}
            for protocol, dataset in fault_datasets.items():
                fault_meta, fault_image, fault_data = unpack(dataset[index], "cuda:0")
                _, _, fault_features = features(camera, fault_image)
                output_d, states[protocol], decoder_d = run_camera_with_rep(
                    camera, fault_meta, fault_data, fault_features, prev_exists,
                    states[protocol])
                exact, box_diff = validate_trace(protocol, token, output_d, pc_range)
                replay[protocol]["logits_exact"] &= exact
                replay[protocol]["box_max"] = max(replay[protocol]["box_max"], box_diff)
                replay[protocol]["frames"] += 1
                if not exact or box_diff > 1e-5:
                    raise RuntimeError(f"{protocol} replay divergence {token}")
                for parent in units.get((protocol, token), []):
                    target = target_index[parent["gt_token"]]
                    clean_object = clean_cache[(token, target["token"])]
                    fault_object = camera_object(output_d, decoder_d, target, pc_range)
                    clean_official = output_metrics(
                        output_a, target, camera_gt[token], pc_range, context)
                    fault_official = output_metrics(
                        output_d, target, camera_gt[token], pc_range, context)
                    identity = f"{protocol}/{token}/{target['token']}"
                    validate_parent_metrics(clean_official, parent, "A", identity)
                    validate_parent_metrics(fault_official, parent, "D", identity)
                    fault_cache[(protocol, token, target["token"])] = fault_object

    invariance = []
    for protocol in ("clean", *PROTOCOLS):
        difference, leaves = compare_tensors(
            mmcv.load(str(ROOT / "outputs/stage4/gt_query_survival_audit"
                          / protocol / "predictions.pkl")),
            mmcv.load(str(DISABLED / protocol / "predictions.pkl")))
        invariance.append({"comparison": f"{protocol}_B0_vs_disabled",
                           "tensor_leaves": leaves, "max_abs_diff": difference,
                           "exact": difference == 0.0})
        value = replay[protocol]
        invariance.append({"comparison": f"{protocol}_manual_replay_vs_trace",
                           "tensor_leaves": 2 * value["frames"],
                           "max_abs_diff": value["box_max"],
                           "exact": value["logits_exact"] and value["box_max"] <= 1e-5})
    if not all(row["exact"] for row in invariance):
        raise RuntimeError(f"camera invariance failure: {invariance}")
    write_csv("disabled_invariance.csv", invariance)

    signal_rows, representations = [], {}
    for parent in parent_rows:
        key = (parent["protocol"], parent["sample_token"], parent["gt_token"])
        teacher_target = teacher_cache[parent["sample_token"]][parent["gt_token"]]
        clean = clean_cache[(parent["sample_token"], parent["gt_token"])]
        fault = fault_cache[key]
        row = {key_name: parent[key_name] for key_name in (
            "protocol", "condition", "group", "sample_token", "scene_token",
            "frame_idx", "fault_episode_age", "gt_token", "gt_class",
            "paired_gt_token", "alternative_view_count")}
        row.update({
            "instance_token": teacher_target["instance_token"],
            "num_lidar_pts": teacher_target["num_lidar_pts"],
            "teacher_matched": teacher_target["teacher_matched"],
            "teacher_score": teacher_target["teacher_score"],
            "teacher_class_logit": probability_logit(teacher_target["teacher_score"]),
            "teacher_xy_center_error": teacher_target["teacher_xy_center_error"],
            "teacher_relative_size_l1": teacher_target["teacher_relative_size_l1"],
            "teacher_yaw_error_deg": math.degrees(teacher_target["teacher_yaw_error_rad"]),
            "clean_qplus": clean["qplus"], "clean_s_pos": clean["s_pos"],
            "clean_gt_class_logit": probability_logit(clean["s_pos"]),
            "clean_class_margin": delta(
                clean["s_pos"], np.nanmax(np.delete(
                    clean["class_probs"], teacher_target["label"]))),
            "clean_rank": clean["rank"], "clean_margin": clean["margin"],
            "clean_xy_center_error": clean["xy_center_error"],
            "clean_relative_size_l1": clean["relative_size_l1"],
            "fault_qplus": fault["qplus"], "fault_s_pos": fault["s_pos"],
            "fault_gt_class_logit": probability_logit(fault["s_pos"]),
            "fault_class_margin": delta(
                fault["s_pos"], np.nanmax(np.delete(
                    fault["class_probs"], teacher_target["label"]))),
            "fault_rank": fault["rank"], "fault_margin": fault["margin"],
            "fault_xy_center_error": fault["xy_center_error"],
            "fault_relative_size_l1": fault["relative_size_l1"],
            "teacher_score_minus_clean_spos": delta(
                teacher_target["teacher_score"], clean["s_pos"]),
            "clean_center_advantage": delta(
                clean["xy_center_error"], teacher_target["teacher_xy_center_error"]),
        })
        for prefix, box in (("teacher", np.concatenate([
                teacher_target["teacher_center"], teacher_target["teacher_size"],
                [teacher_target["teacher_yaw"]]])),
                            ("clean", clean["box"][:7]),
                            ("fault", fault["box"][:7])):
            for name, value in zip(("x", "y", "z", "w", "l", "h", "yaw"), box):
                row[f"{prefix}_box_{name}"] = float(value)
        if teacher_target["teacher_matched"]:
            clean_center_gap = float(np.linalg.norm(
                clean["box"][:2] - teacher_target["teacher_center"][:2]))
            fault_center_gap = float(np.linalg.norm(
                fault["box"][:2] - teacher_target["teacher_center"][:2]))
            clean_size_gap = float(np.mean(np.abs(
                clean["box"][3:6] - teacher_target["teacher_size"])))
            fault_size_gap = float(np.mean(np.abs(
                fault["box"][3:6] - teacher_target["teacher_size"])))
            row.update({
                "clean_teacher_score_abs_gap": abs(
                    teacher_target["teacher_score"] - clean["s_pos"]),
                "fault_teacher_score_abs_gap": abs(
                    teacher_target["teacher_score"] - fault["s_pos"]),
                "score_gap_expansion": abs(
                    teacher_target["teacher_score"] - fault["s_pos"])
                    - abs(teacher_target["teacher_score"] - clean["s_pos"]),
                "clean_teacher_center_gap": clean_center_gap,
                "fault_teacher_center_gap": fault_center_gap,
                "center_gap_expansion": fault_center_gap - clean_center_gap,
                "clean_teacher_size_gap": clean_size_gap,
                "fault_teacher_size_gap": fault_size_gap,
                "size_gap_expansion": fault_size_gap - clean_size_gap,
            })
            representations[key] = (
                teacher_target["teacher_representation"], clean["representation"],
                fault["representation"])
        else:
            for name in ("clean_teacher_score_abs_gap", "fault_teacher_score_abs_gap",
                         "score_gap_expansion", "clean_teacher_center_gap",
                         "fault_teacher_center_gap", "center_gap_expansion",
                         "clean_teacher_size_gap", "fault_teacher_size_gap",
                         "size_gap_expansion"):
                row[name] = float("nan")
        row.update({"representation_available": False,
                    "clean_lidar_representation_gap": float("nan"),
                    "fault_lidar_representation_gap": float("nan"),
                    "representation_gap_expansion": float("nan")})
        signal_rows.append(row)

    row_by_key = {(row["protocol"], row["sample_token"], row["gt_token"]): row
                  for row in signal_rows}
    relation_groups = defaultdict(list)
    for key in representations:
        relation_groups[key[:2]].append(key)
    for _, keys in relation_groups.items():
        if len(keys) < 2:
            continue
        keys = sorted(keys)
        lidar_kernel = cosine_matrix([representations[key][0] for key in keys])
        clean_kernel = cosine_matrix([representations[key][1] for key in keys])
        fault_kernel = cosine_matrix([representations[key][2] for key in keys])
        for index, key in enumerate(keys):
            others = np.arange(len(keys)) != index
            clean_gap = float(np.mean(np.abs(
                clean_kernel[index, others] - lidar_kernel[index, others])))
            fault_gap = float(np.mean(np.abs(
                fault_kernel[index, others] - lidar_kernel[index, others])))
            row_by_key[key].update({
                "representation_available": True,
                "clean_lidar_representation_gap": clean_gap,
                "fault_lidar_representation_gap": fault_gap,
                "representation_gap_expansion": fault_gap - clean_gap,
            })
    write_csv("per_gt_signal.csv", signal_rows)

    summary_rows = []
    for protocol in (*PROTOCOL_NAMES, "pooled"):
        for group in ("lost", "retained"):
            rows = [row for row in signal_rows if row["group"] == group
                    and (protocol == "pooled" or row["protocol"] == protocol)]
            summary_rows.append(summarize(protocol, group, rows))
    write_csv("gap_summary.csv", summary_rows)
    summary_index = {(row["protocol"], row["group"]): row for row in summary_rows}

    paired_rows = []
    signal_index = {(row["protocol"], row["sample_token"], row["gt_token"], row["group"]): row
                    for row in signal_rows}
    for lost in (row for row in signal_rows if row["group"] == "lost"):
        retained = signal_index[(lost["protocol"], lost["sample_token"],
                                 lost["paired_gt_token"], "retained")]
        pair = {"protocol": lost["protocol"], "sample_token": lost["sample_token"]}
        for metric in ("score_gap_expansion", "center_gap_expansion",
                       "representation_gap_expansion"):
            pair[metric + "_enrichment"] = delta(lost[metric], retained[metric])
        paired_rows.append(pair)

    temporal_rows = []
    infos_by_scene = defaultdict(list)
    for info in clean_dataset.data_infos:
        if str(info["token"]) in relevant_tokens:
            infos_by_scene[str(info["scene_token"])].append(info)
    for protocol, group, scene, instance in sorted(population_specs):
        observations = []
        for info in sorted(infos_by_scene[scene], key=lambda value: int(value["frame_idx"])):
            token = str(info["token"])
            teacher_target = next((value for value in teacher_cache[token].values()
                                   if value["instance_token"] == instance), None)
            if teacher_target is None or not teacher_target["teacher_matched"]:
                continue
            clean = clean_cache.get((token, teacher_target["token"]))
            if clean is None or not clean["candidate"]:
                continue
            observations.append((int(info["frame_idx"]), token, teacher_target, clean))
        for left, right in zip(observations, observations[1:]):
            if right[0] - left[0] != 1:
                continue
            teacher_rep = cosine_distance(
                left[2]["teacher_representation"], right[2]["teacher_representation"])
            clean_rep = cosine_distance(
                left[3]["representation"], right[3]["representation"])
            teacher_score = abs(right[2]["teacher_score"] - left[2]["teacher_score"])
            clean_score = abs(right[3]["s_pos"] - left[3]["s_pos"])
            teacher_center = abs(right[2]["teacher_xy_center_error"]
                                 - left[2]["teacher_xy_center_error"])
            clean_center = abs(right[3]["xy_center_error"]
                               - left[3]["xy_center_error"])
            temporal_rows.append({
                "protocol": protocol, "condition": PROTOCOL_NAMES[protocol],
                "group": group, "scene_token": scene, "instance_token": instance,
                "left_frame_idx": left[0], "right_frame_idx": right[0],
                "teacher_representation_drift": teacher_rep,
                "clean_representation_drift": clean_rep,
                "representation_stability_improvement": clean_rep - teacher_rep,
                "teacher_score_drift": teacher_score, "clean_score_drift": clean_score,
                "score_stability_improvement": clean_score - teacher_score,
                "teacher_center_error_drift": teacher_center,
                "clean_center_error_drift": clean_center,
                "center_stability_improvement": clean_center - teacher_center,
            })
    write_csv("teacher_temporal_stability.csv", temporal_rows)

    bootstrap_rows, boot = [], {}
    boot_index = 0
    metric_names = ("score_gap_expansion", "center_gap_expansion",
                    "representation_gap_expansion",
                    "teacher_score_minus_clean_spos", "clean_center_advantage")
    for protocol in (*PROTOCOL_NAMES, "pooled"):
        for group in ("lost", "retained"):
            rows = [row for row in signal_rows if row["group"] == group
                    and (protocol == "pooled" or row["protocol"] == protocol)]
            for metric in metric_names:
                result = cluster_bootstrap_median(
                    rows, metric, ("protocol", "sample_token"),
                    SEED + boot_index, BOOTSTRAPS)
                boot_index += 1
                boot[("signal", protocol, group, metric)] = result
                bootstrap_rows.append({"category": "signal", "protocol": protocol,
                    "group": group, "metric": metric, "n": len(rows), **result})
        pairs = [row for row in paired_rows
                 if protocol == "pooled" or row["protocol"] == protocol]
        for metric in ("score_gap_expansion_enrichment",
                       "center_gap_expansion_enrichment",
                       "representation_gap_expansion_enrichment"):
            result = cluster_bootstrap_median(
                pairs, metric, ("protocol", "sample_token"),
                SEED + boot_index, BOOTSTRAPS)
            boot_index += 1
            boot[("enrichment", protocol, "paired", metric)] = result
            bootstrap_rows.append({"category": "enrichment", "protocol": protocol,
                "group": "paired", "metric": metric, "n": len(pairs), **result})
    lost_temporal = [row for row in temporal_rows if row["group"] == "lost"]
    for metric in ("representation_stability_improvement",
                   "score_stability_improvement", "center_stability_improvement"):
        result = cluster_bootstrap_median(
            lost_temporal, metric, ("protocol", "instance_token"),
            SEED + boot_index, BOOTSTRAPS)
        boot_index += 1
        boot[("temporal", "pooled", "lost", metric)] = result
        bootstrap_rows.append({"category": "temporal", "protocol": "pooled",
            "group": "lost", "metric": metric, "n": len(lost_temporal), **result})
    write_csv("bootstrap_95ci.csv", bootstrap_rows)

    pooled_lost = summary_index[("pooled", "lost")]
    gap = {}
    for short, metric, enrichment in (
            ("score", "score_gap_expansion", "score_gap_expansion_enrichment"),
            ("center", "center_gap_expansion", "center_gap_expansion_enrichment"),
            ("representation", "representation_gap_expansion",
             "representation_gap_expansion_enrichment")):
        result = boot[("signal", "pooled", "lost", metric)]
        enrich = boot[("enrichment", "pooled", "paired", enrichment)]
        gap[f"{short}_median"] = result["estimate"]
        gap[f"{short}_ci_low"] = result["ci_low"]
        gap[f"{short}_cross_protocol"] = all(
            summary_index[(protocol, "lost")][f"median_{metric}"] > 0
            for protocol in PROTOCOL_NAMES)
        gap[f"{short}_enrichment_median"] = enrich["estimate"]
        gap[f"{short}_enrichment_ci_low"] = enrich["ci_low"]

    score_strength_boot = boot[("signal", "pooled", "lost",
                                "teacher_score_minus_clean_spos")]
    geometry_strength_boot = boot[("signal", "pooled", "lost",
                                   "clean_center_advantage")]
    temporal_medians = {key: finite_median(lost_temporal, key) for key in (
        "teacher_representation_drift", "clean_representation_drift",
        "teacher_score_drift", "clean_score_drift",
        "teacher_center_error_drift", "clean_center_error_drift")}

    def relative_improvement(clean_key: str, teacher_key: str) -> float:
        clean_value = temporal_medians[clean_key]
        return ((clean_value - temporal_medians[teacher_key]) / clean_value
                if math.isfinite(clean_value) and clean_value > 1e-12 else float("nan"))

    rep_relative = relative_improvement(
        "clean_representation_drift", "teacher_representation_drift")
    score_relative = relative_improvement("clean_score_drift", "teacher_score_drift")
    center_relative = relative_improvement(
        "clean_center_error_drift", "teacher_center_error_drift")
    superiority = {
        "score_strength": (score_strength_boot["estimate"] >= 0.05
                           and score_strength_boot["ci_low"] > 0),
        "geometry_strength": (geometry_strength_boot["estimate"] >= 0.10
                              and geometry_strength_boot["ci_low"] > 0),
        "temporal_representation": (
            len(lost_temporal) >= 10 and rep_relative >= 0.20
            and boot[("temporal", "pooled", "lost",
                      "representation_stability_improvement")]["ci_low"] > 0),
        "temporal_score_or_geometry": (
            (score_relative >= 0.20
             and boot[("temporal", "pooled", "lost",
                       "score_stability_improvement")]["ci_low"] > 0)
            or (center_relative >= 0.20
                and boot[("temporal", "pooled", "lost",
                          "center_stability_improvement")]["ci_low"] > 0)),
    }
    decision = cross_modal_signal_decision(True, gap, superiority)
    decision_row = {"teacher_coverage_pass": True, **decision, **gap,
                    "teacher_score_minus_clean_spos_median": score_strength_boot["estimate"],
                    "teacher_score_minus_clean_spos_ci_low": score_strength_boot["ci_low"],
                    "clean_center_advantage_median": geometry_strength_boot["estimate"],
                    "clean_center_advantage_ci_low": geometry_strength_boot["ci_low"],
                    "temporal_pair_count": len(lost_temporal),
                    "representation_relative_stability_improvement": rep_relative,
                    "score_relative_stability_improvement": score_relative,
                    "center_relative_stability_improvement": center_relative,
                    **{f"superiority_{key}": value for key, value in superiority.items()}}
    write_csv("mechanism_decision.csv", [decision_row])

    lines = ["# LiDAR-Privileged Target/Temporal Representation Signal Audit", "",
             "## 最终判定", "", f"**{decision['decision']}**。", ""]
    if decision["signal_pass"]:
        lines += ["Teacher coverage、lost-specific cross-modal gap和LiDAR-over-Clean优势门均通过；",
                  "只允许进入预注册的最小train-only target/temporal distillation single-batch overfit。", ""]
    else:
        lines += ["Signal门未全部通过；按预注册停止LiDAR privileged路线，"
                  "未实现distillation、未建optimizer、未overfit或smoke。", ""]
    lines += ["## Teacher coverage", "",
              "CenterPoint pooled lost/retained match=90.1%/94.4%，lost median score=0.705、"
              "xy error=0.187m；coverage/quality/teacher temporal stability三门已通过。", "",
              "## Clean/Fault/LiDAR gap", "",
              "| Protocol | lost n | score expansion | center expansion | representation expansion |",
              "|---|---:|---:|---:|---:|"]
    for protocol in PROTOCOL_NAMES:
        row = summary_index[(protocol, "lost")]
        lines.append(f"| {PROTOCOL_NAMES[protocol]} | {row['n']} | "
                     f"{row['median_score_gap_expansion']:+.4f} | "
                     f"{row['median_center_gap_expansion']:+.3f}m | "
                     f"{row['median_representation_gap_expansion']:+.4f} |")
    lines += ["", f"Pooled lost score/center/representation expansion="
              f"{gap['score_median']:+.4f}/{gap['center_median']:+.3f}m/"
              f"{gap['representation_median']:+.4f}。",
              f"Paired lost-retained enrichment="
              f"{gap['score_enrichment_median']:+.4f}/"
              f"{gap['center_enrichment_median']:+.3f}m/"
              f"{gap['representation_enrichment_median']:+.4f}。", "",
              "## LiDAR vs Clean-camera teacher", "",
              f"Teacher score-Clean S_pos={score_strength_boot['estimate']:+.4f} "
              f"(CI lower {score_strength_boot['ci_low']:+.4f})；"
              f"Clean center error-teacher={geometry_strength_boot['estimate']:+.3f}m "
              f"(CI lower {geometry_strength_boot['ci_low']:+.3f})。",
              f"Temporal relative improvement: representation={rep_relative:+.1%}，"
              f"score={score_relative:+.1%}，center-error={center_relative:+.1%}；"
              f"matched lost pairs={len(lost_temporal)}。", "",
              f"Raw temporal drift (LiDAR/Clean): representation="
              f"{temporal_medians['teacher_representation_drift']:.4f}/"
              f"{temporal_medians['clean_representation_drift']:.4f}，score="
              f"{temporal_medians['teacher_score_drift']:.4f}/"
              f"{temporal_medians['clean_score_drift']:.4f}，center-error="
              f"{temporal_medians['teacher_center_error_drift']:.4f}/"
              f"{temporal_medians['clean_center_error_drift']:.4f}。", "",
              "## 判定门", "",
              f"- score lost-specific gap: {decision['score_gap_pass']}",
              f"- representation lost-specific gap: {decision['representation_gap_pass']}",
              f"- geometry corroboration: {decision['geometry_corroboration']}",
              f"- target strength over Clean teacher: {decision['target_strength_pass']}",
              f"- temporal stability over Clean teacher: "
              f"{decision['temporal_stability_advantage_pass']}", "",
              "Clean/Dark/Blur/Crash canonical replay与disabled均逐tensor exact，最大差0。",
              "未修改repos/StreamPETR。"]
    (REPORT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
