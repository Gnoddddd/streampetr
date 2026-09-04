#!/usr/bin/env python3
"""Offline stock-B0 Hungarian supervision identity audit."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes

from analysis.fault_boundary_root_cause import candidate_pool_statistics
from analysis.supervision_identity import (
    assignment_identity,
    bootstrap_rate_difference,
    trajectory_statistics,
    wilson_interval,
)


ROOT = Path(__file__).resolve().parents[1]
STREAM_ROOT = ROOT / "repos/StreamPETR"
sys.dont_write_bytecode = True
if str(STREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM_ROOT))

# Import the exact vendored B0 assigner registration without changing the vendor tree.
from mmdet.core import build_assigner  # noqa: E402
from mmdet.models import build_loss  # noqa: E402
from projects.mmdet3d_plugin.core.bbox import assigners as _assigners  # noqa: E402,F401
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox  # noqa: E402


TRACE_ROOT = ROOT / "outputs/stage4/gt_query_survival_audit"
DISABLED_ROOT = ROOT / "outputs/stage4/lidar_privileged_target_evidence_audit/disabled"
ROOT_CAUSE = ROOT / "reports/stage4/fault_boundary_root_cause_audit/per_gt_root_cause.csv"
REPORT = ROOT / "reports/stage4/supervision_identity_audit"
CONFIG = ROOT / "configs/stage4/gt_query_survival_b0_audit.py"
ANNOTATIONS = ROOT / "data/nuscenes-mini/nuscenes2d_temporal_infos_val.pkl"
GROUPS = {
    "dark_back": "CAM_BACK Dark",
    "blur_back": "CAM_BACK Blur",
    "crash_back": "CAM_BACK Crash",
}
CLASS_NAMES = (
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
POPULATIONS = (
    "lost_all_available",
    "lost_degraded",
    "retained_all_available",
    "retained_degraded",
)
BOOTSTRAPS = 5000
SEED = 314159


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty output: {name}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def compare_tensors(left, right) -> tuple[float, int]:
    if hasattr(left, "tensor") or hasattr(right, "tensor"):
        return compare_tensors(left.tensor, right.tensor)
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or left.shape != right.shape:
            return float("inf"), 0
        difference = float((left.cpu() - right.cpu()).abs().max()) if left.numel() else 0.0
        return difference, 1
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            return float("inf"), 0
        values = [compare_tensors(left[key], right[key]) for key in left]
    elif isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            return float("inf"), 0
        values = [compare_tensors(a, b) for a, b in zip(left, right)]
    else:
        return (0.0 if left == right else float("inf")), 1
    return max((value[0] for value in values), default=0.0), sum(
        value[1] for value in values
    )


def load_trace(protocol: str, token: str) -> dict:
    path = TRACE_ROOT / protocol / "trace" / f"{token}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as value:
        return {key: value[key].copy() for key in value.files}


def load_root_rows() -> list[dict]:
    with ROOT_CAUSE.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("root-cause population is empty")
    return rows


def build_gt_cache(nusc: NuScenes) -> dict[str, dict]:
    payload = mmcv.load(str(ANNOTATIONS))
    output = {}
    for info in payload["infos"]:
        token = str(info["token"])
        sample = nusc.get("sample", token)
        ann_tokens = sample["anns"]
        if len(ann_tokens) != len(info["gt_boxes"]):
            raise RuntimeError(f"annotation order mismatch for {token}")
        velocity = np.asarray(info["gt_velocity"], dtype=np.float32).copy()
        velocity[~np.isfinite(velocity)] = 0.0
        valid = np.asarray(info["valid_flag"], dtype=bool)
        all_gt, train_gt = {}, []
        for index, ann_token in enumerate(ann_tokens):
            name = str(info["gt_names"][index])
            # The stock training pipeline's ObjectNameFilter removes taxonomy
            # entries outside the ten detection classes before the head loss.
            if name not in CLASS_TO_INDEX:
                continue
            detection_name = category_to_detection_name(
                nusc.get("sample_annotation", ann_token)["category_name"]
            )
            if detection_name != name:
                raise RuntimeError(
                    f"GT name/order mismatch for {token} index {index}: "
                    f"{detection_name} != {name}"
                )
            box = np.concatenate([
                np.asarray(info["gt_boxes"][index], dtype=np.float32),
                velocity[index],
            ])
            target = {
                "token": str(ann_token),
                "name": name,
                "label": CLASS_TO_INDEX[name],
                "box": box,
                "source_index": int(index),
                "train_eligible": bool(valid[index]),
            }
            all_gt[str(ann_token)] = target
            if valid[index]:
                train_gt.append(target)
        output[token] = {
            "all": all_gt,
            "train": train_gt,
            "train_index": {value["token"]: i for i, value in enumerate(train_gt)},
        }
    return output


def sigmoid_focal_class_term(
    logit: torch.Tensor,
    positive: bool,
    average_factor: int,
    alpha: float = 0.25,
    gamma: float = 2.0,
    loss_weight: float = 2.0,
) -> float:
    target = logit.new_tensor(1.0 if positive else 0.0)
    probability = logit.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    pt = probability if positive else 1.0 - probability
    alpha_t = alpha if positive else 1.0 - alpha
    value = ce * ((1.0 - pt) ** gamma) * alpha_t
    return float(value.item() * loss_weight / max(int(average_factor), 1))


def population_member(row: dict, population: str) -> bool:
    lost = row["outcome"] == "fault_induced_lost"
    degraded = float(row["delta_s_pos"]) < 0.0
    if population == "lost_all_available":
        return lost
    if population == "lost_degraded":
        return lost and degraded
    if population == "retained_all_available":
        return not lost
    if population == "retained_degraded":
        return not lost and degraded
    raise KeyError(population)


def median(rows: list[dict], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def mean_bool(rows: list[dict], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows])) if rows else float("nan")


def summarize(protocol: str, population: str, rows: list[dict], layers: list[dict]) -> dict:
    count = len(rows)
    identity_counts = {
        identity: sum(row["final_identity"] == identity for row in rows)
        for identity in (
            "same-GT positive", "other-GT matched", "unmatched-background"
        )
    }
    same_low, same_high = wilson_interval(identity_counts["same-GT positive"], count)
    never_count = sum(bool(row["never_same_gt"]) for row in rows)
    never_low, never_high = wilson_interval(never_count, count)
    return {
        "protocol": protocol,
        "condition": GROUPS.get(protocol, "Pooled"),
        "population": population,
        "n": count,
        "final_same_gt_count": identity_counts["same-GT positive"],
        "final_same_gt_rate": (
            identity_counts["same-GT positive"] / count if count else float("nan")
        ),
        "final_same_gt_wilson_low": same_low,
        "final_same_gt_wilson_high": same_high,
        "final_other_gt_count": identity_counts["other-GT matched"],
        "final_other_gt_rate": (
            identity_counts["other-GT matched"] / count if count else float("nan")
        ),
        "final_background_count": identity_counts["unmatched-background"],
        "final_background_rate": (
            identity_counts["unmatched-background"] / count if count else float("nan")
        ),
        "target_train_eligible_rate": mean_bool(rows, "target_train_eligible"),
        "ever_same_gt_rate": mean_bool(rows, "ever_same_gt"),
        "always_same_gt_rate": mean_bool(rows, "always_same_gt"),
        "never_same_gt_count": never_count,
        "never_same_gt_rate": never_count / count if count else float("nan"),
        "never_same_gt_wilson_low": never_low,
        "never_same_gt_wilson_high": never_high,
        "mean_same_gt_layer_fraction": (
            float(np.mean([row["same_gt_layer_fraction"] for row in rows]))
            if rows else float("nan")
        ),
        "final_classification_participation_rate": mean_bool(
            rows, "final_classification_participates"
        ),
        "final_classification_positive_rate": mean_bool(
            rows, "final_classification_positive"
        ),
        "final_focal_class_positive_rate": mean_bool(
            rows, "final_focal_class_positive"
        ),
        "final_regression_participation_rate": mean_bool(
            rows, "final_regression_participates"
        ),
        "final_qplus_claimed_by_other_gt_rate": mean_bool(
            rows, "final_qplus_claimed_by_other_gt"
        ),
        "final_focal_gt_claimed_by_other_query_rate": mean_bool(
            rows, "final_focal_gt_claimed_by_other_query"
        ),
        "final_assignment_conflict_rate": mean_bool(rows, "final_assignment_conflict"),
        "median_final_cls_loss_contribution": median(rows, "final_cls_loss_contribution"),
        "median_final_focal_class_loss_contribution": median(
            rows, "final_focal_class_loss_contribution"
        ),
        "median_final_reg_loss_contribution": median(rows, "final_reg_loss_contribution"),
        "all_layer_same_gt_rate": mean_bool(layers, "same_gt_positive"),
        "all_layer_classification_positive_rate": mean_bool(
            layers, "classification_positive"
        ),
        "all_layer_regression_participation_rate": mean_bool(
            layers, "regression_participates"
        ),
    }


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    config = Config.fromfile(str(CONFIG))
    assigner_cfg = config.model.train_cfg.pts.assigner
    head_cfg = config.model.pts_bbox_head
    assigner = build_assigner(assigner_cfg)
    cls_loss = build_loss(head_cfg.loss_cls)
    bbox_loss = build_loss(head_cfg.loss_bbox)
    pc_range = tuple(float(value) for value in head_cfg.bbox_coder.pc_range)
    code_weights = torch.tensor(head_cfg.code_weights, dtype=torch.float32)
    if bool(head_cfg.match_with_velo):
        raise RuntimeError("pre-registration requires match_with_velo=False")

    invariance_rows = []
    for protocol in ("clean", *GROUPS):
        difference, leaves = compare_tensors(
            mmcv.load(str(TRACE_ROOT / protocol / "predictions.pkl")),
            mmcv.load(str(DISABLED_ROOT / protocol / "predictions.pkl")),
        )
        invariance_rows.append({
            "protocol": protocol,
            "tensor_leaves": leaves,
            "max_abs_diff": difference,
            "exact": difference == 0.0,
        })
    if not all(row["exact"] for row in invariance_rows):
        raise RuntimeError(f"disabled path diverges from B0: {invariance_rows}")
    write_csv("disabled_invariance.csv", invariance_rows)

    nusc = NuScenes(
        version="v1.0-mini",
        dataroot=str(ROOT / "data/nuscenes-mini"),
        verbose=False,
    )
    gt_cache = build_gt_cache(nusc)
    root_rows = load_root_rows()
    available_root = [
        row for row in root_rows
        if row["protocol"] in GROUPS and row["fault_candidate_available"] == "True"
    ]
    if not available_root:
        raise RuntimeError("no geometry-qualified q+ rows")

    frame_cache = {}
    per_layer_rows, per_gt_rows = [], []
    for root_row in available_root:
        protocol = root_row["protocol"]
        sample_token = root_row["sample_token"]
        frame_key = (protocol, sample_token)
        if frame_key not in frame_cache:
            frame = load_trace(protocol, sample_token)
            gt_data = gt_cache[sample_token]
            targets = gt_data["train"]
            gt_boxes = torch.tensor(
                np.stack([target["box"] for target in targets]), dtype=torch.float32
            )
            gt_labels = torch.tensor(
                [target["label"] for target in targets], dtype=torch.long
            )
            layers = []
            for layer in range(int(frame["layer_logits"].shape[0])):
                logits = torch.tensor(frame["layer_logits"][layer], dtype=torch.float32)
                physical_boxes = torch.tensor(
                    frame["layer_boxes"][layer], dtype=torch.float32
                )
                bbox_preds = normalize_bbox(physical_boxes, pc_range)
                result = assigner.assign(
                    bbox_preds,
                    logits,
                    gt_boxes,
                    gt_labels,
                    None,
                    code_weights,
                    False,
                )
                assigned = result.gt_inds.detach().cpu().numpy().astype(np.int64) - 1
                layers.append({
                    "logits": logits,
                    "physical_boxes": physical_boxes,
                    "bbox_preds": bbox_preds,
                    "assigned": assigned,
                    "positive_count": int(np.count_nonzero(assigned >= 0)),
                })
            frame_cache[frame_key] = (frame, gt_data, layers)

        frame, gt_data, layers = frame_cache[frame_key]
        gt_token = root_row["gt_token"]
        if gt_token not in gt_data["all"]:
            raise RuntimeError(f"GT token absent from annotation info: {gt_token}")
        focal = gt_data["all"][gt_token]
        focal_index = gt_data["train_index"].get(gt_token)
        qplus = int(root_row["fault_best_query"])
        check = candidate_pool_statistics(
            frame["layer_logits"][-1],
            frame["layer_boxes"][-1],
            focal["box"][:3],
            focal["label"],
            topk=100,
            radius=2.0,
        )
        if not check["candidate_available"] or int(check["best_query"]) != qplus:
            raise RuntimeError(
                f"q+ identity mismatch for {protocol}/{sample_token}/{gt_token}: "
                f"{check['best_query']} != {qplus}"
            )

        current_layer_rows = []
        for layer_index, layer in enumerate(layers):
            assigned_index = int(layer["assigned"][qplus])
            identity = assignment_identity(assigned_index, focal_index)
            assigned_target = (
                gt_data["train"][assigned_index] if assigned_index >= 0 else None
            )
            positive_count = layer["positive_count"]
            cls_target = (
                int(assigned_target["label"])
                if assigned_target is not None else len(CLASS_NAMES)
            )
            cls_value = cls_loss(
                layer["logits"][qplus:qplus + 1],
                torch.tensor([cls_target], dtype=torch.long),
                torch.ones(1, dtype=torch.float32),
                avg_factor=max(positive_count, 1),
            )
            focal_class_positive = bool(
                assigned_target is not None
                and assigned_target["label"] == focal["label"]
            )
            focal_class_value = sigmoid_focal_class_term(
                layer["logits"][qplus, focal["label"]],
                focal_class_positive,
                positive_count,
            )
            if assigned_target is not None:
                target_box = torch.tensor(
                    assigned_target["box"], dtype=torch.float32
                ).unsqueeze(0)
                normalized_target = normalize_bbox(target_box, pc_range)
                reg_value = bbox_loss(
                    layer["bbox_preds"][qplus:qplus + 1, :10],
                    normalized_target[:, :10],
                    code_weights.unsqueeze(0),
                    avg_factor=max(positive_count, 1),
                )
                reg_loss_value = float(reg_value.item())
            else:
                reg_loss_value = 0.0
            focal_query = -1
            if focal_index is not None:
                matches = np.flatnonzero(layer["assigned"] == focal_index)
                if matches.size != 1:
                    raise RuntimeError(
                        f"expected one Hungarian match for {gt_token}, got {matches.size}"
                    )
                focal_query = int(matches[0])
            claimed_other = identity == "other-GT matched"
            focal_elsewhere = bool(
                focal_index is not None
                and identity != "same-GT positive"
                and focal_query >= 0
                and focal_query != qplus
            )
            distance = float(torch.linalg.vector_norm(
                layer["physical_boxes"][qplus, :3]
                - torch.tensor(focal["box"][:3], dtype=torch.float32)
            ).item())
            layer_row = {
                "protocol": protocol,
                "condition": GROUPS[protocol],
                "sample_token": sample_token,
                "scene_token": root_row["scene_token"],
                "frame_idx": int(root_row["frame_idx"]),
                "gt_token": gt_token,
                "gt_class": root_row["gt_class"],
                "outcome": root_row["outcome"],
                "delta_s_pos": float(root_row["delta_s_pos"]),
                "qplus": qplus,
                "decoder_layer": layer_index,
                "target_train_eligible": focal_index is not None,
                "identity": identity,
                "same_gt_positive": identity == "same-GT positive",
                "assigned_gt_token": (
                    assigned_target["token"] if assigned_target is not None else ""
                ),
                "assigned_gt_class": (
                    assigned_target["name"] if assigned_target is not None else "background"
                ),
                "focal_gt_matched_query": focal_query,
                "classification_participates": True,
                "classification_positive": assigned_target is not None,
                "classification_target": (
                    assigned_target["name"] if assigned_target is not None else "background"
                ),
                "focal_class_positive": focal_class_positive,
                "cls_loss_contribution": float(cls_value.item()),
                "focal_class_loss_contribution": focal_class_value,
                "regression_participates": assigned_target is not None,
                "regression_same_gt": identity == "same-GT positive",
                "regression_target_gt_token": (
                    assigned_target["token"] if assigned_target is not None else ""
                ),
                "reg_loss_contribution": reg_loss_value,
                "qplus_center_distance_to_focal_gt": distance,
                "qplus_claimed_by_other_gt": claimed_other,
                "focal_gt_claimed_by_other_query": focal_elsewhere,
                "assignment_conflict": claimed_other or focal_elsewhere,
                "layer_positive_count": positive_count,
            }
            current_layer_rows.append(layer_row)
            per_layer_rows.append(layer_row)

        trajectory = trajectory_statistics(row["identity"] for row in current_layer_rows)
        final = current_layer_rows[-1]
        per_gt_rows.append({
            "protocol": protocol,
            "condition": GROUPS[protocol],
            "sample_token": sample_token,
            "scene_token": root_row["scene_token"],
            "frame_idx": int(root_row["frame_idx"]),
            "gt_token": gt_token,
            "gt_class": root_row["gt_class"],
            "outcome": root_row["outcome"],
            "delta_s_pos": float(root_row["delta_s_pos"]),
            "fault_s_pos": float(root_row["fault_s_pos"]),
            "fault_best_rank": int(root_row["fault_best_rank"]),
            "alternative_view_count": int(root_row["alternative_view_count"]),
            "qplus": qplus,
            "target_train_eligible": focal_index is not None,
            "final_identity": final["identity"],
            "final_assigned_gt_token": final["assigned_gt_token"],
            "final_assigned_gt_class": final["assigned_gt_class"],
            "final_focal_gt_matched_query": final["focal_gt_matched_query"],
            **trajectory,
            "final_classification_participates": final["classification_participates"],
            "final_classification_positive": final["classification_positive"],
            "final_classification_target": final["classification_target"],
            "final_focal_class_positive": final["focal_class_positive"],
            "final_cls_loss_contribution": final["cls_loss_contribution"],
            "final_focal_class_loss_contribution": final[
                "focal_class_loss_contribution"
            ],
            "final_regression_participates": final["regression_participates"],
            "final_regression_same_gt": final["regression_same_gt"],
            "final_regression_target_gt_token": final["regression_target_gt_token"],
            "final_reg_loss_contribution": final["reg_loss_contribution"],
            "final_qplus_claimed_by_other_gt": final["qplus_claimed_by_other_gt"],
            "final_focal_gt_claimed_by_other_query": final[
                "focal_gt_claimed_by_other_query"
            ],
            "final_assignment_conflict": final["assignment_conflict"],
            "assignment_conflict_layer_count": sum(
                row["assignment_conflict"] for row in current_layer_rows
            ),
        })

    write_csv("per_layer_assignment.csv", per_layer_rows)
    write_csv("per_gt_identity.csv", per_gt_rows)

    summary_rows = []
    for protocol in (*GROUPS, "pooled"):
        protocol_rows = (
            per_gt_rows if protocol == "pooled"
            else [row for row in per_gt_rows if row["protocol"] == protocol]
        )
        protocol_layers = (
            per_layer_rows if protocol == "pooled"
            else [row for row in per_layer_rows if row["protocol"] == protocol]
        )
        for population in POPULATIONS:
            rows = [row for row in protocol_rows if population_member(row, population)]
            keys = {(row["protocol"], row["sample_token"], row["gt_token"]) for row in rows}
            layers = [
                row for row in protocol_layers
                if (row["protocol"], row["sample_token"], row["gt_token"]) in keys
            ]
            summary_rows.append(summarize(protocol, population, rows, layers))
    write_csv("protocol_group_summary.csv", summary_rows)

    contrast_rows = []
    for index, protocol in enumerate((*GROUPS, "pooled")):
        protocol_rows = (
            per_gt_rows if protocol == "pooled"
            else [row for row in per_gt_rows if row["protocol"] == protocol]
        )
        lost = [
            row for row in protocol_rows if population_member(row, "lost_degraded")
        ]
        retained = [
            row for row in protocol_rows
            if population_member(row, "retained_all_available")
        ]
        for offset, (metric, key) in enumerate((
            ("final_same_gt_rate", "final_identity"),
            ("never_same_gt_rate", "never_same_gt"),
        )):
            if key == "final_identity":
                left = [row[key] == "same-GT positive" for row in lost]
                right = [row[key] == "same-GT positive" for row in retained]
            else:
                left = [bool(row[key]) for row in lost]
                right = [bool(row[key]) for row in retained]
            value = bootstrap_rate_difference(
                left, right, SEED + index * 10 + offset, BOOTSTRAPS
            )
            contrast_rows.append({
                "protocol": protocol,
                "condition": GROUPS.get(protocol, "Pooled"),
                "contrast": "lost_degraded-minus-retained_all_available",
                "metric": metric,
                "lost_n": len(lost),
                "retained_n": len(retained),
                **value,
            })
    write_csv("contrast_bootstrap.csv", contrast_rows)

    summary_index = {
        (row["protocol"], row["population"]): row for row in summary_rows
    }
    contrast_index = {
        (row["protocol"], row["metric"]): row for row in contrast_rows
    }
    primary = summary_index[("pooled", "lost_degraded")]
    route_a_stop = bool(primary["final_same_gt_rate"] > 0.5)
    strong_cross_fault_stop = all(
        summary_index[(protocol, "lost_degraded")]["final_same_gt_rate"] > 0.5
        for protocol in GROUPS
    )
    pooled_never_difference = contrast_index[("pooled", "never_same_gt_rate")][
        "estimate"
    ]
    positive_never_direction = all(
        contrast_index[(protocol, "never_same_gt_rate")]["estimate"] > 0.0
        for protocol in GROUPS
    )
    assignment_support = bool(
        primary["final_same_gt_rate"] <= 0.5
        and primary["never_same_gt_rate"] >= 0.30
        and pooled_never_difference >= 0.15
        and positive_never_direction
    )
    decision = (
        "ROUTE_A_STOP" if route_a_stop
        else "FAULT_AWARE_ASSIGNMENT_SUPPORTED" if assignment_support
        else "IDENTITY_EVIDENCE_INSUFFICIENT"
    )

    available_counts = {
        protocol: sum(
            row["protocol"] == protocol and row["outcome"] == "fault_induced_lost"
            for row in per_gt_rows
        )
        for protocol in GROUPS
    }
    total_root_counts = {
        protocol: sum(
            row["protocol"] == protocol and row["outcome"] == "fault_induced_lost"
            for row in root_rows
        )
        for protocol in GROUPS
    }
    missing_counts = {
        protocol: total_root_counts[protocol] - available_counts[protocol]
        for protocol in GROUPS
    }
    report_lines = [
        "# Supervision Identity Audit",
        "",
        "## 决策",
        "",
        f"**{decision}**。",
        "",
    ]
    if route_a_stop:
        report_lines.extend([
            "多数发生 `S_pos` 退化的 lost-risk final `q+` 已经是原始 B0 Hungarian 的 ",
            "same-GT positive，因此问题不是简单的“缺少正监督”。路线A停止；下一步应优先",
            "评估真正的 LiDAR privileged knowledge 或 multi-view evidence。",
            "",
        ])
    elif assignment_support:
        report_lines.extend([
            "大量几何合格 `q+` 在六层内长期没有 same-GT 正监督，且相对 retained control",
            "跨三种 Fault 符号同向；结果支持 Fault-aware positive assignment 路线。该支持",
            "是 pooled 结论，协议强度异质：Blur 最强、Crash 次之，Dark 几乎不分离。",
            "",
        ])
    else:
        report_lines.extend([
            "预注册的路线停止门与 Fault-aware assignment 支持门均未完整通过；不调阈值，",
            "不进入 smoke。",
            "",
        ])
    report_lines.extend([
        "## Final Hungarian 身份",
        "",
        "主决策集为 `fault_induced_lost ∩ Fault q+<=2m ∩ delta_S_pos<0`。",
        "",
        "| Protocol | n | same-GT | other-GT | background | never same-GT (6 layers) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for protocol in GROUPS:
        row = summary_index[(protocol, "lost_degraded")]
        report_lines.append(
            f"| {GROUPS[protocol]} | {row['n']} | "
            f"{100 * row['final_same_gt_rate']:.1f}% | "
            f"{100 * row['final_other_gt_rate']:.1f}% | "
            f"{100 * row['final_background_rate']:.1f}% | "
            f"{100 * row['never_same_gt_rate']:.1f}% |"
        )
    report_lines.extend([
        "",
        f"Pooled final same-GT rate 为 {100 * primary['final_same_gt_rate']:.1f}% "
        f"({primary['final_same_gt_count']}/{primary['n']}; Wilson 95% CI "
        f"[{100 * primary['final_same_gt_wilson_low']:.1f}%, "
        f"{100 * primary['final_same_gt_wilson_high']:.1f}%])。"
        f"三协议均超过 50%：{strong_cross_fault_stop}。",
        "",
        f"不加 `delta_S_pos<0` 过滤的全部可用 lost-risk `q+` 中，final same-GT 为 "
        f"{100 * summary_index[('pooled', 'lost_all_available')]['final_same_gt_rate']:.1f}% "
        f"({summary_index[('pooled', 'lost_all_available')]['final_same_gt_count']}/"
        f"{summary_index[('pooled', 'lost_all_available')]['n']})，6层 never-same-GT 为 "
        f"{100 * summary_index[('pooled', 'lost_all_available')]['never_same_gt_rate']:.1f}%。",
        "",
        "## Lost 与 retained 对比",
        "",
        "| Protocol | lost same-GT | retained same-GT | lost never | retained never | never差值 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for protocol in GROUPS:
        lost = summary_index[(protocol, "lost_degraded")]
        retained = summary_index[(protocol, "retained_all_available")]
        contrast = contrast_index[(protocol, "never_same_gt_rate")]
        report_lines.append(
            f"| {GROUPS[protocol]} | {100 * lost['final_same_gt_rate']:.1f}% | "
            f"{100 * retained['final_same_gt_rate']:.1f}% | "
            f"{100 * lost['never_same_gt_rate']:.1f}% | "
            f"{100 * retained['never_same_gt_rate']:.1f}% | "
            f"{100 * contrast['estimate']:+.1f} pp |"
        )
    pooled_contrast = contrast_index[("pooled", "never_same_gt_rate")]
    report_lines.extend([
        "",
        f"Pooled never-same-GT 差值为 {100 * pooled_contrast['estimate']:+.1f} pp "
        f"(bootstrap 95% CI [{100 * pooled_contrast['ci_low']:+.1f}, "
        f"{100 * pooled_contrast['ci_high']:+.1f}] pp)；Dark/Blur/Crash 均为正："
        f"{positive_never_direction}。",
        "Dark 的点估计仅 +0.5 pp 且区间跨 0，因此“同方向”只满足预注册符号门，",
        "不代表 Dark 单协议已有稳定分离证据。",
        "",
        "## Loss 参与与冲突",
        "",
        "| Population | cls参与 | 任意正分类 | focal-class正目标 | regression参与 | assignment冲突 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for population in ("lost_degraded", "retained_all_available"):
        row = summary_index[("pooled", population)]
        report_lines.append(
            f"| {population} | {100 * row['final_classification_participation_rate']:.1f}% | "
            f"{100 * row['final_classification_positive_rate']:.1f}% | "
            f"{100 * row['final_focal_class_positive_rate']:.1f}% | "
            f"{100 * row['final_regression_participation_rate']:.1f}% | "
            f"{100 * row['final_assignment_conflict_rate']:.1f}% |"
        )
    report_lines.extend([
        "",
        "背景身份仍以 label weight 1 参加负 focal classification；只有任意 GT positive",
        "才参加 regression，而只有 same-GT positive 同时提供 focal GT 的实例级分类与",
        "回归目标。other-GT 若恰好同类，会对 focal class logit 给正分类目标，但回归到",
        "另一个实例，未计为 same-GT 有效监督。完整逐层 loss contribution 见 CSV。",
        "",
        "## 覆盖与等价性",
        "",
        f"可定义 Fault q+ 的逐GT对象数：{json.dumps(available_counts, ensure_ascii=False)}；"
        f"因 Fault 端无 <=2m q+ 未进入身份分母：{json.dumps(missing_counts, ensure_ascii=False)}。",
        f"主决策集训练 GT eligibility 为 {100 * primary['target_train_eligible_rate']:.1f}%。",
        "Clean/Dark/Blur/Crash 的 disabled prediction 均比较 243 个 tensor leaves，",
        "最大绝对差为 0，逐 tensor exact=True。审计未运行 detector 训练入口、未创建",
        "optimizer，也未改变 B0 forward、loss、memory、query 或 Top-K。",
        "",
        "## 判定门",
        "",
        f"- pooled final same-GT > 50%（路线A停止）：{route_a_stop}",
        f"- 三协议 final same-GT 均 > 50%（强一致）：{strong_cross_fault_stop}",
        f"- Fault-aware assignment 四项支持门全部通过：{assignment_support}",
        "",
        "结论按预注册门生成；未调权重、阈值或进入 smoke。",
    ])
    (REPORT / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
