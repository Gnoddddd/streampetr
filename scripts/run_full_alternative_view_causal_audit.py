#!/usr/bin/env python3
"""Incremental full-nuScenes GT-local alternative-view causal forwards."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import signal
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STREAM = ROOT / "repos/StreamPETR"
sys.path.insert(0, str(STREAM))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmcv.utils import import_modules_from_strings  # noqa: E402
from mmdet3d.datasets import build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from nuscenes.nuscenes import NuScenes  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402

from analysis.cross_view_target_evidence import same_area_background_mask  # noqa: E402
from analysis.dark_target_recoverability import projected_roi, roi_cell_mask  # noqa: E402
from analysis.fault_boundary_root_cause import (  # noqa: E402
    candidate_pool_statistics,
    sigmoid,
    stable_rank,
)
from scripts.audit_dark_target_recoverability import (  # noqa: E402
    CLASSES,
    POST_RANGE,
    features,
    local_gt,
    physical,
    run_head,
    snapshot,
    unpack,
)
from scripts.audit_temporal_state_counterfactual import state_max_difference  # noqa: E402


REPORT = ROOT / "reports/full_nuscenes/alternative_view_causal_audit"
POPULATION = REPORT / "population_forward_units.csv"
POP_MANIFEST = REPORT / "population_manifest.json"
PROGRESS = REPORT / "progress_manifest.json"
PARTIAL = REPORT / "PARTIAL_STATUS.md"
CONFIG = ROOT / "configs/full_nuscenes/stream_petr_r50_90e_mechanism_val.py"
CHECKPOINT = ROOT / "checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth"
DATA = ROOT / "data/nuscenes"
TRACE = ROOT / "reports/full_nuscenes/mechanism_confirmation/paired_inference"
PROTOCOLS = {
    "blur_back": ROOT / "protocols/presets/motion_blur_back_10f_s09.json",
    "crash_back": ROOT / "protocols/presets/camera_crash_back_10f.json",
    "dark_back": ROOT / "protocols/presets/dark_back_10f_s09.json",
}
PROTOCOL_ORDER = tuple(PROTOCOLS)
CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK",
    "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)
CAM_BACK = 3
SCHEMA = 1
STOP_REQUESTED = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("p0_p1", "p2_history"), required=True)
    parser.add_argument("--protocol", choices=PROTOCOL_ORDER, required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def truth(value) -> bool:
    return str(value).lower() == "true"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty scene checkpoint: {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def protocol_dataset(config, schedule: Path):
    data_config = copy.deepcopy(config.data.test)
    nodes = [node for node in data_config.pipeline
             if node.get("type") == "ApplyPartialObservation"]
    if len(nodes) != 1:
        raise RuntimeError(f"expected one partial-observation node, got {len(nodes)}")
    nodes[0]["schedule_file"] = str(schedule)
    data_config.test_mode = True
    return build_dataset(data_config)


def clean_dataset(config):
    data_config = copy.deepcopy(config.data.test)
    nodes = [node for node in data_config.pipeline
             if node.get("type") == "ApplyPartialObservation"]
    if len(nodes) != 1:
        raise RuntimeError(f"expected one partial-observation node, got {len(nodes)}")
    nodes[0]["schedule_file"] = None
    data_config.test_mode = True
    return build_dataset(data_config)


def trace_difference(group: str, token: str, output, pc_range) -> tuple[bool, float]:
    with np.load(TRACE / group / "trace" / f"{token}.npz") as value:
        logits = output["all_cls_scores"][:, 0].detach().cpu().numpy().astype(np.float16)
        exact = bool(np.array_equal(logits, value["layer_logits"]))
        boxes = physical(output, pc_range)[:, 0].detach().float().cpu().numpy()
        difference = float(np.max(np.abs(boxes - value["layer_boxes"])))
    return exact, difference


def match_context(info: dict, dataset) -> dict:
    return {
        "lidar2ego_rotation": Quaternion(info["lidar2ego_rotation"]).rotation_matrix,
        "lidar2ego_translation": np.asarray(info["lidar2ego_translation"], dtype=float),
        "ego2global_rotation": Quaternion(info["ego2global_rotation"]).rotation_matrix,
        "ego2global_translation": np.asarray(info["ego2global_translation"], dtype=float),
        "class_range": dataset.eval_detection_configs.class_range,
    }


def deployed_match_queries(logits, boxes, gt: list[dict], context: dict) -> dict[str, int]:
    scores = torch.as_tensor(logits).float().sigmoid().reshape(-1)
    values, indexes = torch.topk(scores, min(100, scores.numel()))
    predictions = []
    for score, flat in zip(values.tolist(), indexes.tolist()):
        if score < 0.1:
            continue
        query, label = divmod(int(flat), logits.shape[1])
        center = np.asarray(boxes[query, :3], dtype=float)
        if np.any(center < POST_RANGE[:3]) or np.any(center > POST_RANGE[3:]):
            continue
        ego = context["lidar2ego_rotation"] @ center + context["lidar2ego_translation"]
        if np.linalg.norm(ego[:2]) > context["class_range"][CLASSES[label]]:
            continue
        global_center = context["ego2global_rotation"] @ ego + context["ego2global_translation"]
        predictions.append((query, label, global_center))
    pairs = []
    for gt_index, target in enumerate(gt):
        for prediction_index, (_, label, center) in enumerate(predictions):
            if target["label"] != label:
                continue
            distance = float(np.linalg.norm(target["global_center"][:2] - center[:2]))
            if distance <= 2.0:
                pairs.append((distance, gt_index, prediction_index))
    used_gt, used_prediction, matched = set(), set(), {}
    for _, gt_index, prediction_index in sorted(pairs):
        if gt_index in used_gt or prediction_index in used_prediction:
            continue
        used_gt.add(gt_index)
        used_prediction.add(prediction_index)
        matched[gt[gt_index]["token"]] = int(predictions[prediction_index][0])
    return matched


def output_arrays(output, pc_range):
    logits = output["all_cls_scores"][-1, 0].detach().float().cpu().numpy()
    boxes = physical(output, pc_range)[-1, 0].detach().float().cpu().numpy()
    return logits, boxes


def fixed_metrics(output, target: dict, all_gt: list[dict], query: int,
                  pc_range, context: dict) -> dict:
    logits, boxes = output_arrays(output, pc_range)
    scores = sigmoid(logits)
    flat = scores.reshape(-1)
    k = min(100, flat.size)
    boundary = float(np.partition(flat, flat.size - k)[flat.size - k])
    flat_index = int(query) * logits.shape[1] + int(target["label"])
    score = float(scores[int(query), int(target["label"])])
    rank = int(stable_rank(flat, flat_index))
    matched = deployed_match_queries(logits, boxes, all_gt, context)
    return {
        "s_pos": score,
        "s_k": boundary,
        "margin": score - boundary,
        "rank": rank,
        "topk": 0 < rank <= 100,
        "tp": target["token"] in matched,
        "q_center_distance": float(np.linalg.norm(
            boxes[int(query), :3] - np.asarray(target["center"][:3], dtype=float)
        )),
    }


def ablate_masks(feature, target_masks: dict[int, np.ndarray],
                  donor_masks: dict[int, np.ndarray]):
    output = feature.clone()
    for camera in sorted(target_masks):
        targets = np.argwhere(np.asarray(target_masks[camera], dtype=bool))
        donors = np.argwhere(np.asarray(donor_masks[camera], dtype=bool))
        if len(targets) == 0 or len(targets) != len(donors):
            raise RuntimeError(
                f"invalid target/donor budget camera={camera}: {len(targets)}/{len(donors)}"
            )
        target_y = torch.as_tensor(targets[:, 0], device=output.device).long()
        target_x = torch.as_tensor(targets[:, 1], device=output.device).long()
        donor_y = torch.as_tensor(donors[:, 0], device=output.device).long()
        donor_x = torch.as_tensor(donors[:, 1], device=output.device).long()
        output[camera, :, target_y, target_x] = feature[
            camera, :, donor_y, donor_x
        ]
    return output


def make_variants(feature, target_masks: dict[int, np.ndarray],
                  donor_masks: dict[int, np.ndarray], alternatives: list[int]):
    visible = sorted(target_masks)
    back = [CAM_BACK] if CAM_BACK in target_masks else []
    no_back = ablate_masks(
        feature,
        {camera: target_masks[camera] for camera in back},
        {camera: donor_masks[camera] for camera in back},
    ) if back else feature.clone()
    no_alt = ablate_masks(
        feature,
        {camera: target_masks[camera] for camera in alternatives},
        {camera: donor_masks[camera] for camera in alternatives},
    )
    none = ablate_masks(feature, target_masks, donor_masks)
    alt_only = none.clone()
    for camera in alternatives:
        selected = torch.as_tensor(
            target_masks[camera], device=feature.device, dtype=torch.bool
        )
        alt_only[camera, :, selected] = feature[camera, :, selected]
    structural_difference = float((alt_only - no_back).abs().max().item())
    return {
        "remove_cam_back": no_back,
        "remove_alternative": no_alt,
        "all_local_removed": none,
        "alternative_only": alt_only,
    }, structural_difference


def add_metric_columns(row: dict, condition: str, variant: str, value: dict) -> None:
    for key, item in value.items():
        row[f"{condition}_{variant}_{key}"] = item


def add_contributions(row: dict, condition: str) -> None:
    for metric in ("s_pos", "margin"):
        row[f"{condition}_alt_with_back_{metric}"] = (
            float(row[f"{condition}_full_{metric}"])
            - float(row[f"{condition}_remove_alternative_{metric}"])
        )
        row[f"{condition}_alt_addback_{metric}"] = (
            float(row[f"{condition}_alternative_only_{metric}"])
            - float(row[f"{condition}_all_local_removed_{metric}"])
        )
        row[f"{condition}_back_{metric}"] = (
            float(row[f"{condition}_full_{metric}"])
            - float(row[f"{condition}_remove_cam_back_{metric}"])
        )
    for metric in ("topk", "tp"):
        row[f"{condition}_alt_with_back_{metric}"] = (
            int(bool(row[f"{condition}_full_{metric}"]))
            - int(bool(row[f"{condition}_remove_alternative_{metric}"]))
        )
        row[f"{condition}_alt_addback_{metric}"] = (
            int(bool(row[f"{condition}_alternative_only_{metric}"]))
            - int(bool(row[f"{condition}_all_local_removed_{metric}"]))
        )
        row[f"{condition}_back_{metric}"] = (
            int(bool(row[f"{condition}_full_{metric}"]))
            - int(bool(row[f"{condition}_remove_cam_back_{metric}"]))
        )


def target_geometry(all_gt: list[dict], matrices, image_hw, feature_hw):
    rois = {
        target["token"]: [
            projected_roi(target["corners"], matrices[camera], image_hw)
            for camera in range(len(CAMERAS))
        ]
        for target in all_gt
    }
    excluded = []
    for camera in range(len(CAMERAS)):
        mask = np.zeros(feature_hw, dtype=bool)
        for target in all_gt:
            roi = rois[target["token"]][camera]
            if roi is not None:
                mask |= roi_cell_mask(roi, image_hw, feature_hw)
        excluded.append(mask)
    return rois, excluded


def masks_for_target(target: dict, root_row: dict, rois: dict, excluded: list,
                     image_hw, feature_hw):
    target_rois = rois[target["token"]]
    visible = [camera for camera, roi in enumerate(target_rois) if roi is not None]
    alternatives = [camera for camera in visible if camera != CAM_BACK]
    if len(alternatives) != int(root_row["alternative_view_count"]):
        raise RuntimeError(
            f"alternative-view mismatch {target['token']}: "
            f"{len(alternatives)} != {root_row['alternative_view_count']}"
        )
    target_masks, donor_masks, unavailable = {}, {}, []
    for camera in visible:
        target_mask = roi_cell_mask(target_rois[camera], image_hw, feature_hw)
        donor_mask = same_area_background_mask(target_mask, excluded[camera])
        if donor_mask is None:
            unavailable.append(camera)
            donor_mask = np.zeros_like(target_mask)
        target_masks[camera] = target_mask
        donor_masks[camera] = donor_mask
    return visible, alternatives, target_masks, donor_masks, unavailable


def current_feature_differences(clean_feature, fault_feature, target_masks,
                                alternatives, base: dict) -> list[dict]:
    rows = []
    for camera in sorted(target_masks):
        selected = torch.as_tensor(
            target_masks[camera], device=clean_feature.device, dtype=torch.bool
        )
        difference = float((
            clean_feature[camera, :, selected] - fault_feature[camera, :, selected]
        ).abs().max().item())
        rows.append({
            **base,
            "camera_index": camera,
            "camera": CAMERAS[camera],
            "is_cam_back": camera == CAM_BACK,
            "is_alternative": camera in alternatives,
            "target_cells": int(np.count_nonzero(target_masks[camera])),
            "clean_fault_target_feature_max_abs_diff": difference,
        })
    return rows


def build_base_row(root_row: dict, visible: list[int], alternatives: list[int],
                   target_masks: dict, structural: float, qref: int,
                   qref_source: str, state_difference: float) -> dict:
    return {
        "unit_id": root_row["unit_id"],
        "matched_unit_id": root_row["matched_unit_id"],
        "match_cost": root_row["match_cost"],
        "forward_role": root_row["forward_role"],
        "protocol": root_row["protocol"],
        "outcome": root_row["outcome"],
        "sample_token": root_row["sample_token"],
        "scene_token": root_row["scene_token"],
        "frame_idx": int(root_row["frame_idx"]),
        "gt_token": root_row["gt_token"],
        "instance_token": root_row["instance_token"],
        "gt_class": root_row["gt_class"],
        "distance_m": root_row["distance_m"],
        "distance_bin": root_row["distance_bin"],
        "visibility_token": root_row["visibility_token"],
        "cam_back_visible": CAM_BACK in visible,
        "alternative_view_count": len(alternatives),
        "visible_camera_count": len(visible),
        "cam_back_target_cells": (
            int(np.count_nonzero(target_masks[CAM_BACK])) if CAM_BACK in target_masks else 0
        ),
        "alternative_target_cells": sum(
            int(np.count_nonzero(target_masks[camera])) for camera in alternatives
        ),
        "q_reference": qref,
        "q_reference_source": qref_source,
        "state_max_abs_diff": state_difference,
        "alternative_only_remove_cam_back_tensor_max_abs_diff": structural,
        "structural_alias_exact": (
            structural == 0.0 if math.isfinite(float(structural)) else ""
        ),
    }


def unavailable_p0_p1_row(clean_output, fault_output, target, all_gt, root_row,
                          pc_range, context, visible, alternatives, target_masks,
                          clean_pre, fault_pre, unavailable: list[int]):
    qref, source = q_reference(clean_output, target, all_gt, pc_range, context)
    row = build_base_row(
        root_row, visible, alternatives, target_masks, float("nan"), qref, source,
        state_max_difference(clean_pre, fault_pre),
    )
    row["causal_evaluable"] = False
    row["unavailable_reason"] = "donor_unavailable:" + ",".join(
        CAMERAS[camera] for camera in unavailable
    )
    for condition, output in (("A", clean_output), ("D", fault_output)):
        add_metric_columns(
            row, condition, "full",
            fixed_metrics(output, target, all_gt, qref, pc_range, context),
        )
        for variant in (
            "remove_cam_back", "remove_alternative", "alternative_only",
            "all_local_removed",
        ):
            for metric in ("s_pos", "s_k", "margin", "rank", "topk", "tp",
                           "q_center_distance"):
                row[f"{condition}_{variant}_{metric}"] = float("nan")
        for family in ("alt_with_back", "alt_addback", "back"):
            for metric in ("s_pos", "margin", "topk", "tp"):
                row[f"{condition}_{family}_{metric}"] = float("nan")
    for family in ("alt_with_back", "alt_addback"):
        for metric in ("s_pos", "margin", "topk", "tp"):
            row[f"attenuation_AD_{family}_{metric}"] = float("nan")
    expected_fault_tp = root_row["outcome"] == "retained"
    if not row["A_full_tp"] or bool(row["D_full_tp"]) != expected_fault_tp:
        raise RuntimeError(f"canonical population mismatch {root_row['unit_id']}")
    return row


def unavailable_p2_row(root_row: dict, p0_row: dict, visible, alternatives,
                       target_masks, clean_pre, fault_pre, unavailable: list[int]):
    row = build_base_row(
        root_row, visible, alternatives, target_masks, float("nan"),
        int(p0_row["q_reference"]), p0_row["q_reference_source"],
        state_max_difference(clean_pre, fault_pre),
    )
    row["causal_evaluable"] = False
    row["unavailable_reason"] = "donor_unavailable:" + ",".join(
        CAMERAS[camera] for camera in unavailable
    )
    for condition in ("B", "C"):
        for variant in ("full", "remove_cam_back", "remove_alternative",
                        "alternative_only", "all_local_removed"):
            for metric in ("s_pos", "s_k", "margin", "rank", "topk", "tp",
                           "q_center_distance"):
                row[f"{condition}_{variant}_{metric}"] = float("nan")
        for family in ("alt_with_back", "alt_addback", "back"):
            for metric in ("s_pos", "margin", "topk", "tp"):
                row[f"{condition}_{family}_{metric}"] = float("nan")
    for family in ("current_fault_clean_history", "current_fault_fault_history",
                   "fault_history_given_fault_current"):
        for metric in ("s_pos", "margin", "topk", "tp"):
            row[f"{family}_{metric}"] = float("nan")
    return row


def q_reference(clean_output, target, all_gt, pc_range, context):
    logits, boxes = output_arrays(clean_output, pc_range)
    pool = candidate_pool_statistics(
        logits, boxes, target["center"], target["label"], 100, 2.0
    )
    if pool["candidate_available"]:
        return int(pool["best_query"]), "clean_2m_best_query"
    matches = deployed_match_queries(logits, boxes, all_gt, context)
    if target["token"] not in matches:
        raise RuntimeError(f"Clean-correct GT has no q reference: {target['token']}")
    return int(matches[target["token"]]), "clean_deployed_tp_fallback"


def p0_p1_unit(model, clean_output, fault_output, clean_p0, fault_p0,
               clean_meta, fault_meta, clean_data, fault_data, clean_pre, fault_pre,
               prev_exists, target, all_gt, root_row, pc_range, context,
               visible, alternatives, target_masks, donor_masks):
    qref, source = q_reference(clean_output, target, all_gt, pc_range, context)
    clean_variants, clean_structural = make_variants(
        clean_p0, target_masks, donor_masks, alternatives
    )
    fault_variants, fault_structural = make_variants(
        fault_p0, target_masks, donor_masks, alternatives
    )
    if clean_structural != 0.0 or fault_structural != 0.0:
        raise RuntimeError(f"structural alias divergence {root_row['unit_id']}")
    state_difference = state_max_difference(clean_pre, fault_pre)
    row = build_base_row(
        root_row, visible, alternatives, target_masks,
        max(clean_structural, fault_structural), qref, source, state_difference,
    )
    row["causal_evaluable"] = True
    row["unavailable_reason"] = ""
    outputs = {"A": {"full": clean_output}, "D": {"full": fault_output}}
    for condition, variants, meta, data, memory in (
        ("A", clean_variants, clean_meta, clean_data, clean_pre),
        ("D", fault_variants, fault_meta, fault_data, fault_pre),
    ):
        for variant in ("remove_cam_back", "remove_alternative", "all_local_removed"):
            outputs[condition][variant] = run_head(
                model, meta, data, variants[variant].view(1, 6, *variants[variant].shape[1:]),
                prev_exists, memory,
            )[0]
        outputs[condition]["alternative_only"] = outputs[condition]["remove_cam_back"]
        for variant, output in outputs[condition].items():
            add_metric_columns(
                row, condition, variant,
                fixed_metrics(output, target, all_gt, qref, pc_range, context),
            )
        add_contributions(row, condition)
    for metric in ("s_pos", "margin", "topk", "tp"):
        row[f"attenuation_AD_alt_with_back_{metric}"] = (
            float(row[f"D_alt_with_back_{metric}"])
            - float(row[f"A_alt_with_back_{metric}"])
        )
        row[f"attenuation_AD_alt_addback_{metric}"] = (
            float(row[f"D_alt_addback_{metric}"])
            - float(row[f"A_alt_addback_{metric}"])
        )
    expected_fault_tp = root_row["outcome"] == "retained"
    if not row["A_full_tp"] or bool(row["D_full_tp"]) != expected_fault_tp:
        raise RuntimeError(
            f"canonical population mismatch {root_row['unit_id']}: "
            f"A={row['A_full_tp']} D={row['D_full_tp']}"
        )
    return row


def p2_unit(model, clean_output, fault_output, clean_p0, fault_p0,
            clean_meta, fault_meta, clean_data, fault_data, clean_pre, fault_pre,
            prev_exists, target, all_gt, root_row, p0_row, pc_range, context,
            visible, alternatives, target_masks, donor_masks):
    qref = int(p0_row["q_reference"])
    clean_variants, clean_structural = make_variants(
        clean_p0, target_masks, donor_masks, alternatives
    )
    fault_variants, fault_structural = make_variants(
        fault_p0, target_masks, donor_masks, alternatives
    )
    if clean_structural != 0.0 or fault_structural != 0.0:
        raise RuntimeError(f"structural alias divergence {root_row['unit_id']}")
    row = build_base_row(
        root_row, visible, alternatives, target_masks,
        max(clean_structural, fault_structural), qref,
        p0_row["q_reference_source"], state_max_difference(clean_pre, fault_pre),
    )
    row["causal_evaluable"] = True
    row["unavailable_reason"] = ""
    output_b = run_head(
        model, fault_meta, fault_data, fault_p0.view(1, 6, *fault_p0.shape[1:]),
        prev_exists, clean_pre,
    )[0]
    output_c = run_head(
        model, clean_meta, clean_data, clean_p0.view(1, 6, *clean_p0.shape[1:]),
        prev_exists, fault_pre,
    )[0]
    outputs = {"B": {"full": output_b}, "C": {"full": output_c}}
    for condition, variants, meta, data, memory in (
        ("B", fault_variants, fault_meta, fault_data, clean_pre),
        ("C", clean_variants, clean_meta, clean_data, fault_pre),
    ):
        for variant in ("remove_cam_back", "remove_alternative", "all_local_removed"):
            outputs[condition][variant] = run_head(
                model, meta, data, variants[variant].view(1, 6, *variants[variant].shape[1:]),
                prev_exists, memory,
            )[0]
        outputs[condition]["alternative_only"] = outputs[condition]["remove_cam_back"]
        for variant, output in outputs[condition].items():
            add_metric_columns(
                row, condition, variant,
                fixed_metrics(output, target, all_gt, qref, pc_range, context),
            )
        add_contributions(row, condition)
    for metric in ("s_pos", "margin", "topk", "tp"):
        a = float(p0_row[f"A_alt_with_back_{metric}"])
        d = float(p0_row[f"D_alt_with_back_{metric}"])
        b = float(row[f"B_alt_with_back_{metric}"])
        c = float(row[f"C_alt_with_back_{metric}"])
        row[f"current_fault_clean_history_{metric}"] = b - a
        row[f"current_fault_fault_history_{metric}"] = d - c
        row[f"fault_history_given_fault_current_{metric}"] = d - b
    return row


def scene_paths(phase: str, protocol: str, scene: str):
    base = REPORT / "incremental" / phase / protocol
    return {
        "rows": base / f"{scene}.csv",
        "views": base / f"{scene}.views.csv",
        "meta": base / f"{scene}.complete.json",
    }


def complete_scene(phase: str, protocol: str, scene: str, population_hash: str) -> bool:
    paths = scene_paths(phase, protocol, scene)
    if not paths["meta"].is_file() or not paths["rows"].is_file():
        return False
    value = json.loads(paths["meta"].read_text(encoding="utf-8"))
    return bool(value.get("complete") and value.get("schema_version") == SCHEMA
                and value.get("population_sha256") == population_hash
                and value.get("phase") == phase and value.get("protocol") == protocol
                and value.get("scene_token") == scene)


def update_status(population: list[dict], population_hash: str,
                  current_error: str = "") -> None:
    desired = {
        protocol: sorted({row["scene_token"] for row in population
                          if row["protocol"] == protocol})
        for protocol in PROTOCOL_ORDER
    }
    completed = {phase: {} for phase in ("p0_p1", "p2_history")}
    row_counts = {phase: {} for phase in ("p0_p1", "p2_history")}
    for phase in completed:
        for protocol in PROTOCOL_ORDER:
            scenes = [scene for scene in desired[protocol]
                      if complete_scene(phase, protocol, scene, population_hash)]
            completed[phase][protocol] = scenes
            total = 0
            for scene in scenes:
                with scene_paths(phase, protocol, scene)["rows"].open(encoding="utf-8") as handle:
                    total += max(sum(1 for _ in handle) - 1, 0)
            row_counts[phase][protocol] = total
    p0_complete = all(len(completed["p0_p1"][p]) == len(desired[p]) for p in PROTOCOL_ORDER)
    progress = {
        "schema_version": SCHEMA,
        "population_sha256": population_hash,
        "status": ("p0_p1_complete_analysis_pending" if p0_complete
                   else "PARTIAL_INSUFFICIENT_COVERAGE"),
        "completed_scenes": completed,
        "row_counts": row_counts,
        "desired_scenes": {p: len(desired[p]) for p in PROTOCOL_ORDER},
        "current_error": current_error,
        "resume_commands": [
            f"python scripts/run_full_alternative_view_causal_audit.py --phase p0_p1 --protocol {p}"
            for p in PROTOCOL_ORDER
        ] + [
            f"python scripts/run_full_alternative_view_causal_audit.py --phase p2_history --protocol {p}"
            for p in PROTOCOL_ORDER
        ] + ["python scripts/analyze_full_alternative_view_causal_audit.py"],
    }
    atomic_json(PROGRESS, progress)

    prelim = []
    for protocol in PROTOCOL_ORDER:
        rows = []
        for scene in completed["p0_p1"][protocol]:
            rows.extend(read_csv(scene_paths("p0_p1", protocol, scene)["rows"]))
        lost = [row for row in rows if row["outcome"] == "fault_induced_lost"]
        values = np.asarray([
            float(row["attenuation_AD_alt_with_back_s_pos"]) for row in lost
        ], dtype=float)
        values = values[np.isfinite(values)]
        prelim.append({
            "protocol": protocol,
            "completed_scenes": len(completed["p0_p1"][protocol]),
            "desired_scenes": len(desired[protocol]),
            "rows": len(rows),
            "lost": len(lost),
            "preliminary_lost_median_attenuation": (
                float(np.median(values)) if values.size else float("nan")
            ),
        })
    lines = [
        "# PARTIAL STATUS", "",
        ("`P0_P1_COMPLETE_ANALYSIS_PENDING`" if p0_complete
         else "`PARTIAL_INSUFFICIENT_COVERAGE`"), "",
        "部分结果不得用于最终 Go/No-Go。", "",
        "## Progress", "",
        "| Protocol | completed/desired scenes | rows | lost | preliminary median attenuation |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in prelim:
        lines.append(
            f"| {row['protocol']} | {row['completed_scenes']}/{row['desired_scenes']} | "
            f"{row['rows']} | {row['lost']} | "
            f"{row['preliminary_lost_median_attenuation']:+.6f} |"
        )
    if current_error:
        lines += ["", "## Current error", "", f"`{current_error}`"]
    lines += ["", "## Resume", "", "```bash", *progress["resume_commands"], "```", ""]
    temporary = PARTIAL.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(str(temporary), str(PARTIAL))


def signal_handler(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def run_scene(args, scene: str, indices: list[int], scene_units: list[dict],
              clean, fault, model, nusc, pc_range, population_hash: str):
    head = model.pts_bbox_head
    head.reset_memory()
    initial = snapshot(head)
    clean_state, fault_state = initial, initial
    by_frame = defaultdict(list)
    for row in scene_units:
        by_frame[row["sample_token"]].append(row)
    p0_map = {}
    if args.phase == "p2_history":
        p0_path = scene_paths("p0_p1", args.protocol, scene)["rows"]
        if not complete_scene("p0_p1", args.protocol, scene, population_hash):
            raise RuntimeError(f"P2 requires complete P0/P1 scene: {args.protocol}/{scene}")
        p0_map = {row["unit_id"]: row for row in read_csv(p0_path)}

    output_rows, view_rows = [], []
    replay_frames, clean_exact, fault_exact, box_max = 0, True, True, 0.0
    previous_frame = None
    for index in indices:
        info = clean.data_infos[index]
        frame_idx = int(info["frame_idx"])
        if previous_frame is not None and frame_idx != previous_frame + 1:
            raise RuntimeError(f"non-contiguous scene replay {scene}: {previous_frame}->{frame_idx}")
        previous_frame = frame_idx
        prev_exists = int(frame_idx > 0)
        clean_meta, clean_image, clean_data = unpack(clean[index], model.device)
        fault_meta, fault_image, fault_data = unpack(fault[index], model.device)
        token = str(clean_meta["sample_idx"])
        if str(fault_meta["sample_idx"]) != token:
            raise RuntimeError(f"paired token mismatch {args.protocol}/{token}")
        _, clean_pyramid, clean_feats = features(model, clean_image)
        _, fault_pyramid, fault_feats = features(model, fault_image)
        clean_pre, fault_pre = clean_state, fault_state
        clean_output, clean_state, _ = run_head(
            model, clean_meta, clean_data, clean_feats, prev_exists, clean_pre
        )
        fault_output, fault_state, _ = run_head(
            model, fault_meta, fault_data, fault_feats, prev_exists, fault_pre
        )
        clean_ok, clean_box = trace_difference("clean", token, clean_output, pc_range)
        fault_ok, fault_box = trace_difference(args.protocol, token, fault_output, pc_range)
        clean_exact &= clean_ok
        fault_exact &= fault_ok
        box_max = max(box_max, clean_box, fault_box)
        replay_frames += 1
        if not clean_ok or not fault_ok or max(clean_box, fault_box) > 1e-5:
            raise RuntimeError(
                f"canonical replay divergence {args.protocol}/{token}: "
                f"{clean_ok}/{fault_ok}/{clean_box}/{fault_box}"
            )
        frame_units = by_frame.get(token, [])
        if not frame_units:
            continue
        all_gt = local_gt(nusc, token)
        for target in all_gt:
            ann = nusc.get("sample_annotation", target["token"])
            target["global_center"] = np.asarray(ann["translation"], dtype=float)
        targets = {target["token"]: target for target in all_gt}
        context = match_context(info, clean)
        image_hw = clean_image.shape[-2:]
        clean_p0 = clean_pyramid[0]
        fault_p0 = fault_pyramid[0]
        feature_hw = clean_p0.shape[-2:]
        matrices = clean_data["lidar2img"][0].detach().cpu().numpy()
        rois, excluded = target_geometry(all_gt, matrices, image_hw, feature_hw)
        for root_row in frame_units:
            target = targets[root_row["gt_token"]]
            visible, alternatives, target_masks, donor_masks, unavailable = masks_for_target(
                target, root_row, rois, excluded, image_hw, feature_hw
            )
            if not alternatives:
                raise RuntimeError(f"frozen alternative-view unit has none: {root_row['unit_id']}")
            base_view = {
                "unit_id": root_row["unit_id"], "protocol": args.protocol,
                "outcome": root_row["outcome"], "sample_token": token,
                "scene_token": scene, "frame_idx": frame_idx,
                "gt_token": root_row["gt_token"],
                "instance_token": root_row["instance_token"],
                "gt_class": root_row["gt_class"],
            }
            if args.phase == "p0_p1":
                view_rows.extend(current_feature_differences(
                    clean_p0, fault_p0, target_masks, alternatives, base_view
                ))
                if unavailable:
                    row = unavailable_p0_p1_row(
                        clean_output, fault_output, target, all_gt, root_row,
                        pc_range, context, visible, alternatives, target_masks,
                        clean_pre, fault_pre, unavailable,
                    )
                else:
                    row = p0_p1_unit(
                        model, clean_output, fault_output, clean_p0, fault_p0,
                        clean_meta, fault_meta, clean_data, fault_data, clean_pre, fault_pre,
                        prev_exists, target, all_gt, root_row, pc_range, context,
                        visible, alternatives, target_masks, donor_masks,
                    )
            else:
                if unavailable:
                    row = unavailable_p2_row(
                        root_row, p0_map[root_row["unit_id"]], visible, alternatives,
                        target_masks, clean_pre, fault_pre, unavailable,
                    )
                else:
                    row = p2_unit(
                        model, clean_output, fault_output, clean_p0, fault_p0,
                        clean_meta, fault_meta, clean_data, fault_data, clean_pre, fault_pre,
                        prev_exists, target, all_gt, root_row, p0_map[root_row["unit_id"]],
                        pc_range, context, visible, alternatives, target_masks, donor_masks,
                    )
            output_rows.append(row)
    if len(output_rows) != len(scene_units):
        raise RuntimeError(
            f"scene row count mismatch {args.protocol}/{scene}: "
            f"{len(output_rows)} != {len(scene_units)}"
        )
    paths = scene_paths(args.phase, args.protocol, scene)
    atomic_csv(paths["rows"], output_rows)
    if view_rows:
        atomic_csv(paths["views"], view_rows)
    atomic_json(paths["meta"], {
        "schema_version": SCHEMA,
        "population_sha256": population_hash,
        "phase": args.phase,
        "protocol": args.protocol,
        "scene_token": scene,
        "rows": len(output_rows),
        "causal_evaluable_rows": sum(bool(row.get("causal_evaluable"))
                                     for row in output_rows),
        "donor_unavailable_rows": sum(not bool(row.get("causal_evaluable"))
                                      for row in output_rows),
        "view_rows": len(view_rows),
        "replay_frames": replay_frames,
        "clean_logits_exact": clean_exact,
        "fault_logits_exact": fault_exact,
        "box_max_abs_diff": box_max,
        "complete": bool(clean_exact and fault_exact and box_max <= 1e-5),
    })


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not POP_MANIFEST.is_file() or not POPULATION.is_file():
        raise RuntimeError("population must be frozen before any forward")
    manifest = json.loads(POP_MANIFEST.read_text(encoding="utf-8"))
    population_hash = manifest["forward_sha256"]
    population = read_csv(POPULATION)
    protocol_units = [row for row in population if row["protocol"] == args.protocol]
    desired_scenes = sorted({row["scene_token"] for row in protocol_units})
    pending = [scene for scene in desired_scenes
               if not complete_scene(args.phase, args.protocol, scene, population_hash)]
    if args.max_scenes is not None:
        pending = pending[:max(args.max_scenes, 0)]
    if not pending:
        update_status(population, population_hash)
        print(f"no pending scenes for {args.phase}/{args.protocol}")
        return

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    cfg = Config.fromfile(str(CONFIG))
    import_modules_from_strings(**cfg.custom_imports)
    cfg.model.train_cfg = None
    clean = clean_dataset(cfg)
    fault = protocol_dataset(cfg, PROTOCOLS[args.protocol])
    if len(clean) != len(fault):
        raise RuntimeError("paired dataset length mismatch")
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(CHECKPOINT), map_location="cpu")
    model = model.to(device).eval()
    # Existing helpers use model.device; MMDetection models do not define it.
    model.device = device
    pc_range = model.pts_bbox_head.pc_range.detach()
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(DATA), verbose=False)
    scene_indices = defaultdict(list)
    for index, info in enumerate(clean.data_infos):
        scene = str(info["scene_token"])
        if scene in pending and int(info["frame_idx"]) <= 12:
            scene_indices[scene].append(index)
    error = ""
    try:
        with torch.no_grad():
            for number, scene in enumerate(pending, 1):
                units = [row for row in protocol_units if row["scene_token"] == scene]
                run_scene(
                    args, scene, scene_indices[scene], units, clean, fault, model,
                    nusc, pc_range, population_hash,
                )
                update_status(population, population_hash)
                print(
                    f"completed {args.phase}/{args.protocol} scene {number}/{len(pending)} "
                    f"token={scene} units={len(units)}",
                    flush=True,
                )
                if STOP_REQUESTED:
                    break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        update_status(population, population_hash, error)


if __name__ == "__main__":
    main()
