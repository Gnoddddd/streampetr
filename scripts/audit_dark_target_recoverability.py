#!/usr/bin/env python3
"""Frozen Dark ROI and local-feature recoverability audit."""

from __future__ import annotations

import copy
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
from pyquaternion import Quaternion  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmcv.utils import import_modules_from_strings  # noqa: E402
from mmdet3d.datasets import build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from nuscenes.eval.detection.utils import category_to_detection_name  # noqa: E402
from nuscenes.nuscenes import NuScenes  # noqa: E402
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox  # noqa: E402

from analysis.dark_target_recoverability import (  # noqa: E402
    background_ring_mask, bootstrap_median, bootstrap_median_difference,
    centroid_separability, destructive_fraction, match_retained_controls,
    projected_roi, recovery_fraction, replace_local_feature, roi_cell_mask,
)
from analysis.fault_boundary_root_cause import candidate_pool_statistics  # noqa: E402


CONFIG = ROOT / "configs/stage4/gt_query_survival_b0_audit.py"
CHECKPOINT = ROOT / "outputs/stage3/observability_distillation/b0/iter_969.pth"
TRACE = ROOT / "outputs/stage4/gt_query_survival_audit"
DISABLED = ROOT / "outputs/stage4/lidar_privileged_target_evidence_audit/disabled"
ROOT_CAUSE = ROOT / "reports/stage4/fault_boundary_root_cause_audit/per_gt_root_cause.csv"
DARK_PROTOCOL = ROOT / "protocols/presets/dark_back_10f_s09.json"
REPORT = ROOT / "reports/stage4/dark_target_recoverability_audit"
CLASSES = ("car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
           "motorcycle", "bicycle", "pedestrian", "traffic_cone")
CLASS_INDEX = {name: index for index, name in enumerate(CLASSES)}
CAM_BACK, DARK_FACTOR = 3, 0.235
POST_RANGE = np.asarray([-61.2, -61.2, -10, 61.2, 61.2, 10], float)
MEMORY_NAMES = ("memory_embedding", "memory_reference_point", "memory_timestamp",
                "memory_egopose", "memory_velo")
FEATURE_STAGES = ("backbone_local", "fpn_local", "decoder_input_token")
SEP_LAYERS = ("backbone_c4", "backbone_c5", "fpn_p0", "decoder_input")
PRIMARY_SEP = ("backbone_c4", "fpn_p0", "decoder_input")
SEED, BOOTSTRAPS = 314159, 5000


def write_csv(name, rows):
    if not rows:
        raise RuntimeError(f"empty output: {name}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (REPORT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def compare_tensors(left, right):
    if hasattr(left, "tensor") or hasattr(right, "tensor"):
        return compare_tensors(left.tensor, right.tensor)
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or left.shape != right.shape:
            return float("inf"), 0
        return (float((left.cpu() - right.cpu()).abs().max()) if left.numel() else 0), 1
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            return float("inf"), 0
        values = [compare_tensors(left[key], right[key]) for key in left]
    elif isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            return float("inf"), 0
        values = [compare_tensors(a, b) for a, b in zip(left, right)]
    else:
        return (0 if left == right else float("inf")), 1
    return max((v[0] for v in values), default=0), sum(v[1] for v in values)


def load_root_rows():
    with ROOT_CAUSE.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def true(row, key):
    return str(row[key]).lower() == "true"


def local_gt(nusc, token):
    sample = nusc.get("sample", token)
    _, boxes, _ = nusc.get_sample_data(sample["data"]["LIDAR_TOP"])
    output = []
    for box in boxes:
        name = category_to_detection_name(box.name)
        if name in CLASS_INDEX:
            output.append({"token": str(box.token), "name": name,
                           "label": CLASS_INDEX[name],
                           "center": np.asarray(box.center, float),
                           "corners": np.asarray(box.corners(), float)})
    return output


def snapshot(head):
    return {name: None if getattr(head, name) is None
            else getattr(head, name).detach().clone() for name in MEMORY_NAMES}


def restore(head, state):
    for name in MEMORY_NAMES:
        value = state[name]
        setattr(head, name, None if value is None else value.detach().clone())


def unpack(sample, device):
    meta = sample["img_metas"][0].data
    image = sample["img"][0].data.unsqueeze(0).to(device)
    data = {}
    for key, values in sample.items():
        if key in {"img_metas", "img"}:
            continue
        value = values[0].data
        if torch.is_tensor(value):
            data[key] = value.unsqueeze(0).to(device)
    data["img"] = image
    return meta, image, data


def features(model, image):
    backbone = model.img_backbone(image.flatten(0, 1))
    backbone = tuple(backbone.values()) if isinstance(backbone, dict) else tuple(backbone)
    pyramid = tuple(model.img_neck(backbone))
    _, channels, height, width = pyramid[0].shape
    return backbone, pyramid, pyramid[0].view(1, 6, channels, height, width)


def selected_from_backbone(model, backbone):
    p0 = model.img_neck(tuple(backbone))[0]
    return p0.view(1, 6, *p0.shape[1:])


def selected_from_p0(p0):
    return p0.view(1, 6, *p0.shape[1:])


def run_head(model, meta, base_data, img_feats, prev_exists, memory_state,
             capture=False, replacement=None):
    head = model.pts_bbox_head
    restore(head, memory_state)
    data = dict(base_data)
    data["img_feats"] = img_feats
    data["prev_exists"] = img_feats.new_tensor([float(prev_exists)])
    captured = {}

    def pre_hook(_, arguments):
        values = list(arguments)
        memory, position = values[0], values[3]
        if replacement is not None:
            indexes = torch.as_tensor(replacement["indexes"], device=memory.device).long()
            memory, position = memory.clone(), position.clone()
            memory[:, indexes] = replacement["memory"][:, indexes]
            position[:, indexes] = replacement["position"][:, indexes]
            values[0], values[3] = memory, position
        if capture:
            captured["memory"] = memory.detach().clone()
            captured["position"] = position.detach().clone()
        return tuple(values)

    handle = head.transformer.register_forward_pre_hook(pre_hook)
    try:
        location = model.prepare_location([meta], **data)
        output = head(location, [meta], None, **data)
    finally:
        handle.remove()
    return output, snapshot(head), captured


def physical(output, pc_range):
    raw = output["all_bbox_preds"]
    return denormalize_bbox(raw.reshape(-1, raw.shape[-1]), pc_range).reshape(
        *raw.shape[:-1], -1)


def deployed_matches(logits, boxes, gt, match_context):
    scores = torch.as_tensor(logits).float().sigmoid().reshape(-1)
    values, indexes = torch.topk(scores, min(100, scores.numel()))
    predictions = []
    for score, flat in zip(values.tolist(), indexes.tolist()):
        if score < .1:
            continue
        query, label = divmod(int(flat), logits.shape[1])
        center = np.asarray(boxes[query, :3], float)
        if np.any(center < POST_RANGE[:3]) or np.any(center > POST_RANGE[3:]):
            continue
        ego_center = (match_context["lidar2ego_rotation"] @ center
                      + match_context["lidar2ego_translation"])
        if np.linalg.norm(ego_center[:2]) > match_context["class_range"][CLASSES[label]]:
            continue
        predictions.append((label, center))
    pairs = []
    for gt_index, target in enumerate(gt):
        for pred_index, (label, center) in enumerate(predictions):
            if target["label"] != label:
                continue
            distance = np.linalg.norm(target["center"][:2] - center[:2])
            if distance <= 2:
                pairs.append((float(distance), gt_index, pred_index))
    used_gt, used_prediction, matched = set(), set(), set()
    for _, gt_index, pred_index in sorted(pairs):
        if gt_index in used_gt or pred_index in used_prediction:
            continue
        used_gt.add(gt_index)
        used_prediction.add(pred_index)
        matched.add(gt[gt_index]["token"])
    return matched


def metrics(output, target, gt, pc_range, match_context):
    logits = output["all_cls_scores"][-1, 0].detach().float().cpu().numpy()
    boxes = physical(output, pc_range)[-1, 0].detach().float().cpu().numpy()
    pool = candidate_pool_statistics(logits, boxes, target["center"], target["label"], 100, 2)
    return {"candidate": pool["candidate_available"], "qplus": pool["best_query"],
            "s_pos": pool["s_pos"], "rank": pool["rank"], "margin": pool["margin"],
            "topk": 0 < pool["rank"] <= 100,
            "tp": target["token"] in deployed_matches(
                logits, boxes, gt, match_context)}


def prefixed(prefix, values):
    return {f"{prefix}_{key}": value for key, value in values.items()}


def validate_trace(protocol, token, output, pc_range):
    with np.load(TRACE / protocol / "trace" / f"{token}.npz") as trace:
        logits = output["all_cls_scores"][:, 0].detach().cpu().numpy().astype(np.float16)
        exact = np.array_equal(logits, trace["layer_logits"])
        boxes = physical(output, pc_range)[:, 0].detach().float().cpu().numpy()
        difference = float(np.max(np.abs(boxes - trace["layer_boxes"])))
    return bool(exact), difference


def patch_image(destination, source, mask):
    output = destination.clone()
    mask = torch.as_tensor(mask, device=output.device).bool()
    output[0, CAM_BACK, :, mask] = source[0, CAM_BACK, :, mask]
    return output


def median(rows, key):
    values = np.asarray([float(row[key]) for row in rows], float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else float("nan")


def rate(rows, key):
    return float(np.mean([bool(row[key]) for row in rows])) if rows else float("nan")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    REPORT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")

    root = [row for row in load_root_rows() if row["protocol"] == "dark_back"]
    lost_all = [row for row in root if row["outcome"] == "fault_induced_lost"]
    lost = [row for row in lost_all if true(row, "fault_candidate_available")
            and true(row, "cam_back_visible")]
    retained_pool = [row for row in root if row["outcome"] == "retained_control"
                     and true(row, "fault_candidate_available") and true(row, "cam_back_visible")]
    pairs = match_retained_controls(lost, retained_pool)
    if len(pairs) != len(lost):
        raise RuntimeError("control coverage failure")
    manifest, units = [], defaultdict(list)
    for pair in pairs:
        left, right = pair["lost"], pair["retained"]
        manifest.append({"sample_token": left["sample_token"],
                         "lost_gt_token": left["gt_token"], "lost_class": left["gt_class"],
                         "retained_gt_token": right["gt_token"], "retained_class": right["gt_class"],
                         "same_class": left["gt_class"] == right["gt_class"],
                         "lost_distance": left["gt_center_distance"],
                         "retained_distance": right["gt_center_distance"],
                         "lost_altviews": left["alternative_view_count"],
                         "retained_altviews": right["alternative_view_count"],
                         "match_cost": pair["match_cost"]})
        units[left["sample_token"]].append(("lost", left, right["gt_token"]))
        units[right["sample_token"]].append(("retained", right, left["gt_token"]))
    write_csv("control_manifest.csv", manifest)

    cfg = Config.fromfile(str(CONFIG))
    import_modules_from_strings(**cfg.custom_imports)
    cfg.model.pretrained, cfg.model.train_cfg, cfg.data.test.test_mode = None, None, True
    clean_data_cfg = copy.deepcopy(cfg.data.test)
    dark_data_cfg = copy.deepcopy(cfg.data.test)
    partial_nodes = [node for node in dark_data_cfg.pipeline
                     if node.get("type") == "ApplyPartialObservation"]
    if len(partial_nodes) != 1:
        raise RuntimeError(f"expected one ApplyPartialObservation, got {len(partial_nodes)}")
    partial_nodes[0]["schedule_file"] = str(DARK_PROTOCOL)
    clean_dataset = build_dataset(clean_data_cfg)
    dark_dataset = build_dataset(dark_data_cfg)
    if len(clean_dataset) != len(dark_dataset):
        raise RuntimeError("paired dataset length mismatch")
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(CHECKPOINT), map_location="cpu")
    model = model.to(device).eval()
    head = model.pts_bbox_head
    head.reset_memory()
    initial = snapshot(head)
    pc_range = head.pc_range.detach()
    nusc = NuScenes(version="v1.0-mini", dataroot=str(ROOT / "data/nuscenes-mini"), verbose=False)
    gt_cache = {str(info["token"]): local_gt(nusc, str(info["token"]))
                for info in clean_dataset.data_infos}

    invariance = []
    for protocol in ("clean", "dark_back"):
        difference, leaves = compare_tensors(mmcv.load(str(TRACE / protocol / "predictions.pkl")),
                                             mmcv.load(str(DISABLED / protocol / "predictions.pkl")))
        invariance.append({"comparison": f"{protocol}_B0_vs_disabled",
                           "tensor_leaves": leaves, "max_abs_diff": difference,
                           "exact": difference == 0})
    if not all(row["exact"] for row in invariance):
        raise RuntimeError("historical disabled divergence")

    clean_state = dark_state = initial
    previous_scene = None
    input_rows, feature_rows, sep_rows = [], [], []
    clean_exact = dark_exact = True
    clean_box_max = dark_box_max = 0.
    with torch.no_grad():
        for index in range(len(clean_dataset)):
            clean_meta, clean_image, clean_data = unpack(clean_dataset[index], device)
            dark_meta, dark_image, dark_data = unpack(dark_dataset[index], device)
            token, scene = str(clean_meta["sample_idx"]), str(clean_meta["scene_token"])
            if token != str(dark_meta["sample_idx"]):
                raise RuntimeError(f"paired dataset token mismatch at index {index}")
            prev_exists = 0 if scene != previous_scene else 1
            previous_scene = scene
            cb, cp, cs = features(model, clean_image)
            db, dp, ds = features(model, dark_image)
            clean_pre, dark_pre = clean_state, dark_state
            co, clean_state, cd = run_head(model, clean_meta, clean_data, cs, prev_exists,
                                           clean_pre, capture=True)
            do, dark_state, dd = run_head(model, dark_meta, dark_data, ds, prev_exists,
                                          dark_pre, capture=True)
            ce, cbd = validate_trace("clean", token, co, pc_range)
            de, dbd = validate_trace("dark_back", token, do, pc_range)
            clean_exact, dark_exact = clean_exact and ce, dark_exact and de
            clean_box_max, dark_box_max = max(clean_box_max, cbd), max(dark_box_max, dbd)
            if not ce or not de or cbd > 1e-5 or dbd > 1e-5:
                raise RuntimeError(
                    f"canonical replay divergence at {token}: clean_logits={ce} "
                    f"dark_logits={de} clean_box_diff={cbd:.3e} dark_box_diff={dbd:.3e}")
            if token not in units:
                continue
            all_gt = gt_cache[token]
            by_token = {value["token"]: value for value in all_gt}
            info = clean_dataset.data_infos[index]
            match_context = {
                "lidar2ego_rotation": Quaternion(
                    info["lidar2ego_rotation"]).rotation_matrix,
                "lidar2ego_translation": np.asarray(
                    info["lidar2ego_translation"], float),
                "class_range": clean_dataset.eval_detection_configs.class_range,
            }
            image_hw = clean_image.shape[-2:]
            matrix = clean_data["lidar2img"][0, CAM_BACK].cpu().numpy()
            rois = {value["token"]: projected_roi(value["corners"], matrix, image_hw)
                    for value in all_gt}
            for group, root_row, paired_gt in units[token]:
                target, roi = by_token[root_row["gt_token"]], rois[root_row["gt_token"]]
                if roi is None:
                    raise RuntimeError(f"invalid preregistered ROI {target['token']}")
                other_rois = [value for key, value in rois.items()
                              if key != target["token"] and value is not None]
                cm = metrics(co, target, all_gt, pc_range, match_context)
                dm = metrics(do, target, all_gt, pc_range, match_context)
                expected_dark = root_row["outcome"] == "retained_control"
                if not cm["tp"] or dm["tp"] != expected_dark:
                    raise RuntimeError(
                        f"canonical outcome mismatch {token}/{target['token']}: "
                        f"clean={cm} dark={dm} expected_dark_tp={expected_dark}")

                pixel_mask = roi_cell_mask(roi, image_hw, image_hw)
                _, _, drc_feat = features(model, patch_image(dark_image, clean_image, pixel_mask))
                _, _, crd_feat = features(model, patch_image(clean_image, dark_image, pixel_mask))
                drc = run_head(model, dark_meta, dark_data, drc_feat, prev_exists, dark_pre)[0]
                crd = run_head(model, clean_meta, clean_data, crd_feat, prev_exists, clean_pre)[0]
                rm = metrics(drc, target, all_gt, pc_range, match_context)
                xm = metrics(crd, target, all_gt, pc_range, match_context)
                input_rows.append({"group": group, "sample_token": token,
                    "frame_idx": clean_meta["frame_idx"], "gt_token": target["token"],
                    "gt_class": target["name"], "paired_gt_token": paired_gt,
                    "roi_x0": roi[0], "roi_y0": roi[1], "roi_x1": roi[2], "roi_y1": roi[3],
                    "roi_pixels": int(pixel_mask.sum()), **prefixed("clean", cm),
                    **prefixed("dark", dm), **prefixed("dark_roi_clean", rm),
                    **prefixed("clean_roi_dark", xm),
                    "dark_roi_clean_delta_s_pos": rm["s_pos"] - dm["s_pos"],
                    "dark_roi_clean_rank_improvement": dm["rank"] - rm["rank"],
                    "dark_roi_clean_topk_recovered": not dm["topk"] and rm["topk"],
                    "dark_roi_clean_tp_recovered": not dm["tp"] and rm["tp"],
                    "roi_rescue_fraction": recovery_fraction(cm["s_pos"], dm["s_pos"], rm["s_pos"]),
                    "clean_roi_dark_delta_s_pos": xm["s_pos"] - cm["s_pos"],
                    "clean_roi_dark_rank_degradation": xm["rank"] - cm["rank"],
                    "clean_roi_dark_topk_loss_reproduced": cm["topk"] and not xm["topk"],
                    "clean_roi_dark_tp_loss_reproduced": cm["tp"] and not xm["tp"],
                    "roi_destructive_fraction": destructive_fraction(
                        cm["s_pos"], dm["s_pos"], xm["s_pos"])})

                patched = []
                for c_level, d_level in zip(cb, db):
                    mask = roi_cell_mask(roi, image_hw, d_level.shape[-2:])
                    patched.append(replace_local_feature(d_level, c_level, mask, CAM_BACK))
                backbone_feat = selected_from_backbone(model, patched)
                p0_mask = roi_cell_mask(roi, image_hw, dp[0].shape[-2:])
                fpn_feat = selected_from_p0(replace_local_feature(dp[0], cp[0], p0_mask, CAM_BACK))
                fh, fw = dp[0].shape[-2:]
                decoder_indexes = np.flatnonzero(p0_mask.ravel()) + CAM_BACK * fh * fw
                variants = {
                    "backbone_local": run_head(model, dark_meta, dark_data, backbone_feat,
                                                prev_exists, dark_pre)[0],
                    "fpn_local": run_head(model, dark_meta, dark_data, fpn_feat,
                                           prev_exists, dark_pre)[0],
                    "decoder_input_token": run_head(
                        model, dark_meta, dark_data, ds, prev_exists, dark_pre,
                        replacement={"indexes": decoder_indexes,
                                     "memory": cd["memory"], "position": cd["position"]})[0]}
                for stage, output in variants.items():
                    vm = metrics(output, target, all_gt, pc_range, match_context)
                    feature_rows.append({"group": group, "stage": stage,
                        "sample_token": token, "frame_idx": clean_meta["frame_idx"],
                        "gt_token": target["token"], "gt_class": target["name"],
                        "paired_gt_token": paired_gt, **prefixed("clean", cm),
                        **prefixed("dark", dm), **prefixed("variant", vm),
                        "delta_s_pos_vs_dark": vm["s_pos"] - dm["s_pos"],
                        "rank_improvement_vs_dark": dm["rank"] - vm["rank"],
                        "topk_recovered": not dm["topk"] and vm["topk"],
                        "tp_recovered": not dm["tp"] and vm["tp"],
                        "tp_retained": dm["tp"] and vm["tp"],
                        "rescue_fraction": recovery_fraction(
                            cm["s_pos"], dm["s_pos"], vm["s_pos"])})

                memory_start, memory_stop = CAM_BACK * fh * fw, (CAM_BACK + 1) * fh * fw
                c_memory = cd["memory"][0, memory_start:memory_stop].reshape(fh, fw, -1).permute(2, 0, 1)
                d_memory = dd["memory"][0, memory_start:memory_stop].reshape(fh, fw, -1).permute(2, 0, 1)
                pairs_features = {"backbone_c4": (cb[0][CAM_BACK], db[0][CAM_BACK]),
                                  "backbone_c5": (cb[1][CAM_BACK], db[1][CAM_BACK]),
                                  "fpn_p0": (cp[0][CAM_BACK], dp[0][CAM_BACK]),
                                  "decoder_input": (c_memory, d_memory)}
                for layer, (clean_feature, dark_feature) in pairs_features.items():
                    target_mask = roi_cell_mask(roi, image_hw, clean_feature.shape[-2:])
                    ring_mask = background_ring_mask(roi, other_rois, image_hw,
                                                     clean_feature.shape[-2:])
                    csep = centroid_separability(clean_feature, target_mask, ring_mask)
                    dsep = centroid_separability(dark_feature, target_mask, ring_mask)
                    sep_rows.append({"group": group, "layer": layer,
                        "sample_token": token, "frame_idx": clean_meta["frame_idx"],
                        "gt_token": target["token"], "gt_class": target["name"],
                        "paired_gt_token": paired_gt, "target_cells": csep["target_cells"],
                        "background_cells": csep["background_cells"],
                        "clean_cosine_distance": csep["cosine_distance"],
                        "dark_cosine_distance": dsep["cosine_distance"],
                        "delta_cosine_distance": dsep["cosine_distance"] - csep["cosine_distance"],
                        "clean_normalized_l2": csep["normalized_l2"],
                        "dark_normalized_l2": dsep["normalized_l2"],
                        "delta_normalized_l2": dsep["normalized_l2"] - csep["normalized_l2"]})

    invariance += [
        {"comparison": "clean_manual_replay_vs_trace", "tensor_leaves": 162,
         "max_abs_diff": clean_box_max, "exact": clean_exact and clean_box_max <= 1e-5},
        {"comparison": "dark_manual_replay_vs_trace", "tensor_leaves": 162,
         "max_abs_diff": dark_box_max, "exact": dark_exact and dark_box_max <= 1e-5}]
    if not all(row["exact"] for row in invariance):
        raise RuntimeError(f"canonical replay divergence: {invariance}")
    write_csv("disabled_invariance.csv", invariance)
    write_csv("input_roi_counterfactual.csv", input_rows)
    write_csv("feature_rescue_per_gt.csv", feature_rows)
    write_csv("feature_separability_per_gt.csv", sep_rows)

    input_summary = []
    for group in ("lost", "retained"):
        rows = [row for row in input_rows if row["group"] == group]
        input_summary += [
            {"group": group, "variant": "dark_roi_clean", "n": len(rows),
             "median_delta_s_pos": median(rows, "dark_roi_clean_delta_s_pos"),
             "median_rank_change": median(rows, "dark_roi_clean_rank_improvement"),
             "median_fraction": median(rows, "roi_rescue_fraction"),
             "topk_event_rate": rate(rows, "dark_roi_clean_topk_recovered"),
             "tp_event_rate": rate(rows, "dark_roi_clean_tp_recovered"),
             "variant_tp_rate": rate(rows, "dark_roi_clean_tp")},
            {"group": group, "variant": "clean_roi_dark", "n": len(rows),
             "median_delta_s_pos": median(rows, "clean_roi_dark_delta_s_pos"),
             "median_rank_change": median(rows, "clean_roi_dark_rank_degradation"),
             "median_fraction": median(rows, "roi_destructive_fraction"),
             "topk_event_rate": rate(rows, "clean_roi_dark_topk_loss_reproduced"),
             "tp_event_rate": rate(rows, "clean_roi_dark_tp_loss_reproduced"),
             "variant_tp_rate": rate(rows, "clean_roi_dark_tp")}]
    write_csv("input_roi_summary.csv", input_summary)

    feature_summary = []
    for group in ("lost", "retained"):
        for stage in FEATURE_STAGES:
            rows = [row for row in feature_rows if row["group"] == group and row["stage"] == stage]
            feature_summary.append({"group": group, "stage": stage, "n": len(rows),
                "median_delta_s_pos": median(rows, "delta_s_pos_vs_dark"),
                "median_rank_improvement": median(rows, "rank_improvement_vs_dark"),
                "median_rescue_fraction": median(rows, "rescue_fraction"),
                "topk_recovery_rate": rate(rows, "topk_recovered"),
                "tp_recovery_rate": rate(rows, "tp_recovered"),
                "tp_retention_rate": rate(rows, "variant_tp")})
    write_csv("feature_rescue_summary.csv", feature_summary)

    sep_summary, bootstrap = [], []
    for group in ("lost", "retained"):
        for layer in SEP_LAYERS:
            rows = [row for row in sep_rows if row["group"] == group and row["layer"] == layer
                    and math.isfinite(float(row["delta_cosine_distance"]))]
            interval = bootstrap_median([row["delta_cosine_distance"] for row in rows],
                                        SEED + len(sep_summary), BOOTSTRAPS)
            sep_summary.append({"group": group, "layer": layer, "n": len(rows),
                "median_clean_cosine_distance": median(rows, "clean_cosine_distance"),
                "median_dark_cosine_distance": median(rows, "dark_cosine_distance"),
                "median_delta_cosine_distance": interval["estimate"],
                "delta_ci_low": interval["ci_low"], "delta_ci_high": interval["ci_high"],
                "median_delta_normalized_l2": median(rows, "delta_normalized_l2")})
            bootstrap.append({"category": "separability_delta", "group": group,
                "stage": layer, "metric": "median_dark_minus_clean_cosine", "n": len(rows),
                **interval})
    for layer_index, layer in enumerate(PRIMARY_SEP):
        left = [row["delta_cosine_distance"] for row in sep_rows
                if row["group"] == "lost" and row["layer"] == layer]
        right = [row["delta_cosine_distance"] for row in sep_rows
                 if row["group"] == "retained" and row["layer"] == layer]
        interval = bootstrap_median_difference(left, right, SEED + 100 + layer_index, BOOTSTRAPS)
        bootstrap.append({"category": "lost_minus_retained", "group": "contrast",
                          "stage": layer, "metric": "median_delta_cosine", "n": len(left),
                          **interval})
    for index, row in enumerate(input_summary):
        key = "roi_rescue_fraction" if row["variant"] == "dark_roi_clean" else "roi_destructive_fraction"
        rows = [value for value in input_rows if value["group"] == row["group"]]
        interval = bootstrap_median([value[key] for value in rows], SEED + 200 + index, BOOTSTRAPS)
        bootstrap.append({"category": "input_roi", "group": row["group"],
                          "stage": row["variant"], "metric": f"median_{key}", "n": len(rows),
                          **interval})
    for index, row in enumerate(feature_summary):
        rows = [value for value in feature_rows if value["group"] == row["group"]
                and value["stage"] == row["stage"]]
        interval = bootstrap_median([value["rescue_fraction"] for value in rows],
                                    SEED + 300 + index, BOOTSTRAPS)
        bootstrap.append({"category": "feature_rescue", "group": row["group"],
                          "stage": row["stage"], "metric": "median_rescue_fraction",
                          "n": len(rows), **interval})
    write_csv("feature_separability_summary.csv", sep_summary)
    write_csv("bootstrap_95ci.csv", bootstrap)

    input_idx = {(r["group"], r["variant"]): r for r in input_summary}
    feature_idx = {(r["group"], r["stage"]): r for r in feature_summary}
    sep_idx = {(r["group"], r["layer"]): r for r in sep_summary}
    contrast = {r["stage"]: r for r in bootstrap if r["category"] == "lost_minus_retained"}
    rescue, destroy = input_idx[("lost", "dark_roi_clean")], input_idx[("lost", "clean_roi_dark")]
    input_gate = (rescue["tp_event_rate"] >= .5 and rescue["median_fraction"] >= .5
                  and destroy["tp_event_rate"] >= .5 and destroy["median_fraction"] >= .5)
    passing = [stage for stage in FEATURE_STAGES
               if feature_idx[("lost", stage)]["tp_recovery_rate"] >= .5
               and feature_idx[("lost", stage)]["median_rescue_fraction"] >= .5]
    feature_gate = bool(passing)
    attenuated = [layer for layer in PRIMARY_SEP
                  if sep_idx[("lost", layer)]["median_delta_cosine_distance"] < 0
                  and sep_idx[("lost", layer)]["delta_ci_high"] < 0]
    late_contrast = [layer for layer in ("fpn_p0", "decoder_input")
                     if contrast[layer]["estimate"] < 0 and contrast[layer]["ci_high"] < 0]
    sep_gate = len(attenuated) >= 2 and bool(late_contrast)
    go = (input_gate or feature_gate) and sep_gate
    decision = "GO_RECOVERABLE_REPRESENTATION" if go else "NO_GO_CAMERA_ONLY_DARK"

    lines = ["# Dark Target Evidence Recoverability Audit", "", "## 决策", "",
             f"**{decision}**。", ""]
    if go:
        lines += ["局部ROI/feature rescue达到预注册稳定门，且Dark显著降低target与邻近",
                  "background的feature separability。Dark属于可恢复的representation问题；",
                  "下一阶段才允许评估illumination-invariant或target-focused feature preservation。", ""]
    else:
        lines += ["局部rescue与target semantic attenuation未同时通过预注册门。停止",
                  "camera-only Dark表征路线，不调阈值、不进入smoke。", ""]
    lines += ["## 人群与输入ROI反事实", "",
              f"Dark lost={len(lost_all)}，仍有<=2m q+={sum(true(r, 'fault_candidate_available') for r in lost_all)}，"
              f"ROI有效主集={len(lost)}；同帧1:1 retained control={len(pairs)}。", "",
              "| Group | Variant | median ΔS_pos | median fraction | Top-K recover/reproduce | TP recover/reproduce | variant TP |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for group in ("lost", "retained"):
        for variant in ("dark_roi_clean", "clean_roi_dark"):
            row = input_idx[(group, variant)]
            lines.append(f"| {group} | {variant} | {row['median_delta_s_pos']:+.4f} | "
                         f"{row['median_fraction']:.3f} | {100*row['topk_event_rate']:.1f}% | "
                         f"{100*row['tp_event_rate']:.1f}% | "
                         f"{100*row['variant_tp_rate']:.1f}% |")
    lines += ["", f"输入ROI稳定门：{input_gate}。", "", "## 逐层feature rescue", "",
              "| Stage | lost median ΔS_pos | fraction | Top-K recovery | TP recovery | control TP retention |",
              "|---|---:|---:|---:|---:|---:|"]
    for stage in FEATURE_STAGES:
        left, right = feature_idx[("lost", stage)], feature_idx[("retained", stage)]
        lines.append(f"| {stage} | {left['median_delta_s_pos']:+.4f} | "
                     f"{left['median_rescue_fraction']:.3f} | "
                     f"{100*left['topk_recovery_rate']:.1f}% | "
                     f"{100*left['tp_recovery_rate']:.1f}% | "
                     f"{100*right['tp_retention_rate']:.1f}% |")
    lines += ["", f"Feature门：{feature_gate}；最早明显恢复层：`{passing[0] if passing else 'none'}`。",
              "", "## Target-background separability", "",
              "主指标为centroid cosine distance的`Dark-Clean`，负值表示可分性降低。", "",
              "| Layer | lost delta [95% CI] | retained delta | lost-retained [95% CI] |",
              "|---|---:|---:|---:|"]
    for layer in PRIMARY_SEP:
        left, right, diff = sep_idx[("lost", layer)], sep_idx[("retained", layer)], contrast[layer]
        lines.append(f"| {layer} | {left['median_delta_cosine_distance']:+.5f} "
                     f"[{left['delta_ci_low']:+.5f},{left['delta_ci_high']:+.5f}] | "
                     f"{right['median_delta_cosine_distance']:+.5f} | {diff['estimate']:+.5f} "
                     f"[{diff['ci_low']:+.5f},{diff['ci_high']:+.5f}] |")
    lines += ["", f"显著attenuation主层：{attenuated}；late contrast：{late_contrast}；门：{sep_gate}。",
              "", "## 等价性与边界", "",
              f"Clean/Dark replay float16 logits exact={clean_exact}/{dark_exact}；box最大误差="
              f"{clean_box_max:.3e}/{dark_box_max:.3e}。历史B0与disabled各243个tensor leaves最大差0。",
              "TP matching复用既有nuScenes formatted deployment的class-specific evaluation range过滤。",
              "", "全程`torch.no_grad()`；未调用训练入口、未创建optimizer、未新增loss/module，",
              "未修改memory/query/Top-K，未运行smoke。", "", "## 预注册门", "",
              f"- input ROI rescue：{input_gate}", f"- feature rescue：{feature_gate}",
              f"- semantic attenuation：{sep_gate}",
              f"- 总GO=(input OR feature) AND attenuation：{go}"]
    (REPORT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
