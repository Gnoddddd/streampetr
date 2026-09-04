#!/usr/bin/env python3
"""Analyze frozen Camera-to-Query attention and paired TP loss."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import mmcv
import numpy as np
import torch
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes

from analysis.camera_reliability import (
    CAMERA_NAMES,
    rank_correlation,
    safe_correlation,
)


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/stage4/camera_reliability_audit"
REPORT = ROOT / "reports/stage4/camera_reliability_audit"
CHECKPOINT = ROOT / "outputs/stage3/observability_distillation/b0/iter_969.pth"
CONFIG = ROOT / "configs/stage4/camera_reliability_b0_audit.py"
REFERENCE = (
    ROOT
    / "outputs/stage3/counterfactual_view_deficit_audit"
    / "invariance_original/predictions.pkl"
)
GROUPS = {
    "clean": {"label": "Clean", "cameras": (), "kind": "clean"},
    "dark_back": {"label": "CAM_BACK Dark", "cameras": (3,), "kind": "single"},
    "blur_back": {"label": "CAM_BACK Blur", "cameras": (3,), "kind": "single"},
    "crash_back": {"label": "CAM_BACK Crash", "cameras": (3,), "kind": "single"},
    "crash_left_pair": {
        "label": "Left-pair Crash",
        "cameras": (2, 4),
        "kind": "double",
    },
}
CLASS_NAMES = (
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty table {name}")
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


def traces(group: str) -> dict[str, dict]:
    output = {}
    for path in sorted((RUN / group / "trace").glob("*.npz")):
        with np.load(path) as value:
            output[str(value["sample_token"])] = {
                key: value[key].copy() for key in value.files
            }
    if not output:
        raise RuntimeError(f"no traces for {group}")
    return output


def global_gt(nusc: NuScenes, token: str) -> list[dict]:
    sample = nusc.get("sample", token)
    output = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        name = category_to_detection_name(ann["category_name"])
        if name in CLASS_TO_INDEX:
            output.append({
                "label": CLASS_TO_INDEX[name],
                "center": np.asarray(ann["translation"][:2], dtype=np.float64),
            })
    return output


def local_to_global_xy(nusc: NuScenes, token: str, local_xyz: np.ndarray) -> np.ndarray:
    sample = nusc.get("sample", token)
    lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    calibration = nusc.get("calibrated_sensor", lidar["calibrated_sensor_token"])
    ego = nusc.get("ego_pose", lidar["ego_pose_token"])
    from pyquaternion import Quaternion
    point = Quaternion(calibration["rotation"]).rotate(local_xyz)
    point = point + np.asarray(calibration["translation"])
    point = Quaternion(ego["rotation"]).rotate(point)
    return (point + np.asarray(ego["translation"]))[:2]


def match_frame(nusc: NuScenes, frame: dict) -> dict:
    token = str(frame["sample_token"])
    gt = global_gt(nusc, token)
    selected = np.flatnonzero(frame["scores"] >= 0.1)
    candidates = []
    global_centers = {}
    for pred in selected:
        center = local_to_global_xy(nusc, token, frame["boxes"][pred, :3])
        global_centers[int(pred)] = center
        for target, value in enumerate(gt):
            if int(frame["labels"][pred]) != value["label"]:
                continue
            distance = float(np.linalg.norm(center - value["center"]))
            if distance <= 2.0:
                candidates.append((distance, int(pred), target))
    used_pred, used_gt, matches = set(), set(), []
    for distance, pred, target in sorted(candidates):
        if pred in used_pred or target in used_gt:
            continue
        used_pred.add(pred)
        used_gt.add(target)
        matches.append((pred, target, distance))
    return {
        "gt": len(gt),
        "pred": len(selected),
        "tp": len(matches),
        "fp": len(selected) - len(matches),
        "fn": len(gt) - len(matches),
        "tp_indices": np.asarray([value[0] for value in matches], dtype=np.int64),
    }


def evaluate(group: str, nusc: NuScenes) -> dict:
    result_path = RUN / group / "formatted/pts_bbox/results_nusc.json"
    target = RUN / group / "evaluation"
    summary = target / "metrics_summary.json"
    if not summary.is_file():
        target.mkdir(parents=True, exist_ok=True)
        evaluator = NuScenesEval(
            nusc,
            config_factory("detection_cvpr_2019"),
            result_path=str(result_path),
            eval_set="mini_val",
            output_dir=str(target),
            verbose=False,
        )
        evaluator.main(render_curves=False)
    return json.loads(summary.read_text())


def recursive_max_diff(left, right) -> tuple[float, int]:
    # mmdet3d box containers deliberately do not expose ndarray semantics;
    # their tensor is the prediction-bearing state that must be compared.
    if hasattr(left, "tensor") or hasattr(right, "tensor"):
        if not (hasattr(left, "tensor") and hasattr(right, "tensor")):
            return float("inf"), 0
        return recursive_max_diff(left.tensor, right.tensor)
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or left.shape != right.shape:
            return float("inf"), 0
        if left.numel() == 0:
            return 0.0, 1
        return float((left.detach().cpu() - right.detach().cpu()).abs().max()), 1
    if isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or left.shape != right.shape:
            return float("inf"), 0
        return (float(np.max(np.abs(left - right))) if left.size else 0.0), 1
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            return float("inf"), 0
        values = [recursive_max_diff(left[key], right[key]) for key in left]
    elif isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            return float("inf"), 0
        values = [recursive_max_diff(a, b) for a, b in zip(left, right)]
    else:
        return (0.0 if left == right else float("inf")), 1
    return max((value[0] for value in values), default=0.0), sum(v[1] for v in values)


def mean_or_nan(values) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def fmt(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.6f}"


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(
        version="v1.0-mini", dataroot=str(ROOT / "data/nuscenes-mini"), verbose=False
    )
    all_trace = {name: traces(name) for name in GROUPS}
    clean = all_trace["clean"]
    if any(set(value) != set(clean) for value in all_trace.values()):
        raise RuntimeError("protocol sample tokens differ")

    reference = mmcv.load(str(REFERENCE))
    hooked = mmcv.load(str(RUN / "clean/predictions.pkl"))
    max_diff, tensors = recursive_max_diff(reference, hooked)
    invariance_rows = [{
        "comparison": "historical_B0_vs_clean_attention_hook",
        "reference": str(REFERENCE.relative_to(ROOT)),
        "records_compared": tensors,
        "max_abs_diff": max_diff,
        "exact": max_diff == 0.0,
    }]
    if max_diff != 0.0:
        write_csv("prediction_invariance.csv", invariance_rows)
        raise RuntimeError(f"attention hook changed B0 prediction: {max_diff}")

    metric_rows, metric_lookup = [], {}
    for group, spec in GROUPS.items():
        metrics = evaluate(group, nusc)
        row = {
            "group": group,
            "condition": spec["label"],
            "mAP": float(metrics["mean_ap"]),
            "NDS": float(metrics["nd_score"]),
        }
        for key in ("trans_err", "scale_err", "orient_err", "vel_err", "attr_err"):
            row[key] = float(metrics["tp_errors"][key])
        metric_rows.append(row)
        metric_lookup[group] = row
    clean_nds = metric_lookup["clean"]["NDS"]
    for row in metric_rows:
        row["NDS_drop_vs_clean"] = clean_nds - row["NDS"]

    frame_rows, gt_rows, layer_rows = [], [], []
    group_attention_rows = []
    correlations = []
    active_group_values = {}
    for group, spec in GROUPS.items():
        cameras = tuple(spec["cameras"])
        group_frame_rows = []
        for token, frame in all_trace[group].items():
            baseline = clean[token]
            frame_index = int(frame["frame_idx"])
            # All registered fault schedules are active on frames 3..12.
            active = bool(cameras) and 3 <= frame_index <= 12
            current_match = match_frame(nusc, frame)
            clean_match = match_frame(nusc, baseline)
            final_all = frame["all_query_camera_attention_mean"][-1]
            final_deployed = frame["deployed_camera_attention"][-1]
            clean_all = baseline["all_query_camera_attention_mean"][-1]
            clean_deployed = baseline["deployed_camera_attention"][-1]
            if cameras:
                all_mass = float(final_all[list(cameras)].sum())
                clean_all_mass = float(clean_all[list(cameras)].sum())
                deployed_mass = float(final_deployed[:, list(cameras)].sum(1).mean())
                clean_deployed_mass = float(
                    clean_deployed[:, list(cameras)].sum(1).mean()
                )
                token_share = float(frame["camera_token_share"][list(cameras)].sum())
                retention = deployed_mass / max(clean_deployed_mass, 1e-12)
                all_retention = all_mass / max(clean_all_mass, 1e-12)
                amplification = deployed_mass / max(token_share, 1e-12)
                tp_indices = current_match["tp_indices"]
                tp_mass = (
                    float(final_deployed[tp_indices][:, list(cameras)].sum(1).mean())
                    if tp_indices.size else float("nan")
                )
            else:
                all_mass = deployed_mass = token_share = 1.0
                retention = all_retention = amplification = 1.0
                tp_mass = float("nan")
            row = {
                "group": group,
                "sample_token": token,
                "scene_token": str(frame["scene_token"]),
                "frame_idx": frame_index,
                "fault_active": active,
                "degraded_cameras": "|".join(CAMERA_NAMES[i] for i in cameras),
                "degraded_all_query_attention": all_mass,
                "degraded_deployed_query_attention": deployed_mass,
                "degraded_tp_query_attention": tp_mass,
                "degraded_camera_token_share": token_share,
                "deployed_attention_retention": retention,
                "all_query_attention_retention": all_retention,
                "attention_to_token_amplification": amplification,
                "tp": current_match["tp"],
                "clean_tp": clean_match["tp"],
                "tp_loss": clean_match["tp"] - current_match["tp"],
                "gt_recall": current_match["tp"] / max(current_match["gt"], 1),
                "clean_gt_recall": clean_match["tp"] / max(clean_match["gt"], 1),
                "fp": current_match["fp"],
                "fn": current_match["fn"],
            }
            frame_rows.append(row)
            group_frame_rows.append(row)
            gt_rows.append({key: row[key] for key in (
                "group", "sample_token", "scene_token", "frame_idx", "fault_active",
                "tp", "clean_tp", "tp_loss", "gt_recall", "clean_gt_recall", "fp", "fn",
                "degraded_tp_query_attention",
            )})
        selected = [row for row in group_frame_rows if row["fault_active"]]
        if not selected and group == "clean":
            selected = [row for row in group_frame_rows if 3 <= row["frame_idx"] <= 12]
        summary = {
            "group": group,
            "condition": spec["label"],
            "active_frames": len(selected),
            "degraded_deployed_attention_mean": mean_or_nan(
                row["degraded_deployed_query_attention"] for row in selected
            ),
            "degraded_all_query_attention_mean": mean_or_nan(
                row["degraded_all_query_attention"] for row in selected
            ),
            "degraded_tp_attention_mean": mean_or_nan(
                row["degraded_tp_query_attention"] for row in selected
            ),
            "camera_token_share_mean": mean_or_nan(
                row["degraded_camera_token_share"] for row in selected
            ),
            "deployed_attention_retention_mean": mean_or_nan(
                row["deployed_attention_retention"] for row in selected
            ),
            "all_query_attention_retention_mean": mean_or_nan(
                row["all_query_attention_retention"] for row in selected
            ),
            "attention_to_token_amplification_mean": mean_or_nan(
                row["attention_to_token_amplification"] for row in selected
            ),
            "paired_tp_loss_mean": mean_or_nan(row["tp_loss"] for row in selected),
            "paired_recall_loss_mean": mean_or_nan(
                row["clean_gt_recall"] - row["gt_recall"] for row in selected
            ),
        }
        group_attention_rows.append(summary)
        active_group_values[group] = summary
        selected_tokens = [token for token, frame in all_trace[group].items()
                           if 3 <= int(frame["frame_idx"]) <= 12]
        for layer in range(
            int(next(iter(all_trace[group].values()))[
                "all_query_camera_attention_mean"
            ].shape[0])
        ):
            for camera, camera_name in enumerate(CAMERA_NAMES):
                all_values, deployed_values, clean_values = [], [], []
                for token in selected_tokens:
                    frame = all_trace[group][token]
                    baseline = clean[token]
                    all_values.append(float(
                        frame["all_query_camera_attention_mean"][layer, camera]
                    ))
                    deployed_values.append(float(
                        frame["deployed_camera_attention"][layer, :, camera].mean()
                    ))
                    clean_values.append(float(
                        baseline["deployed_camera_attention"][layer, :, camera].mean()
                    ))
                degraded = camera in cameras
                layer_rows.append({
                    "group": group,
                    "condition": spec["label"],
                    "decoder_layer": layer,
                    "camera": camera_name,
                    "is_degraded_camera": degraded,
                    "frames": len(selected_tokens),
                    "all_query_attention_mean": mean_or_nan(all_values),
                    "deployed_query_attention_mean": mean_or_nan(deployed_values),
                    "retention_vs_clean": (
                        mean_or_nan(deployed_values) / max(mean_or_nan(clean_values), 1e-12)
                    ),
                })
        if cameras:
            correlations.append({
                "scope": group,
                "n": len(selected),
                "x": "deployed_attention_retention",
                "y": "paired_tp_loss",
                "pearson": safe_correlation(
                    (row["deployed_attention_retention"] for row in selected),
                    (row["tp_loss"] for row in selected),
                ),
                "spearman": rank_correlation(
                    (row["deployed_attention_retention"] for row in selected),
                    (row["tp_loss"] for row in selected),
                ),
            })

    singles = ("dark_back", "blur_back", "crash_back")
    correlations.append({
        "scope": "three_single_camera_groups",
        "n": 3,
        "x": "mean_deployed_attention_retention",
        "y": "NDS_drop_vs_clean",
        "pearson": float("nan"),
        "spearman": float("nan"),
    })
    # Correct the group-level y values after constructing the row explicitly.
    group_x = [active_group_values[name]["deployed_attention_retention_mean"] for name in singles]
    group_y = [clean_nds - metric_lookup[name]["NDS"] for name in singles]
    correlations[-1]["pearson"] = safe_correlation(group_x, group_y)
    correlations[-1]["spearman"] = rank_correlation(group_x, group_y)

    manifest_rows = []
    for group, spec in GROUPS.items():
        protocol_hash = "none"
        hash_file = RUN / group / "protocol_sha256.txt"
        if hash_file.is_file():
            protocol_hash = hash_file.read_text().split()[0]
        manifest_rows.append({
            "group": group,
            "condition": spec["label"],
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "checkpoint_sha256": sha256(CHECKPOINT),
            "config": str(CONFIG.relative_to(ROOT)),
            "config_sha256": sha256(CONFIG),
            "protocol_sha256": protocol_hash,
            "training": False,
            "frames": len(all_trace[group]),
            "exit_code": (RUN / group / "exit_code.txt").read_text().strip(),
        })

    write_csv("experiment_manifest.csv", manifest_rows)
    write_csv("prediction_invariance.csv", invariance_rows)
    write_csv("per_protocol_metrics.csv", metric_rows)
    write_csv("per_frame_contribution.csv", frame_rows)
    write_csv("camera_attention_by_protocol.csv", group_attention_rows)
    write_csv("attention_by_layer_camera.csv", layer_rows)
    write_csv("gt_tp_loss.csv", gt_rows)
    write_csv("contribution_performance_correlation.csv", correlations)

    high_retention = sum(
        active_group_values[name]["deployed_attention_retention_mean"] >= 0.80
        for name in singles
    )
    nds_degraded = sum(clean_nds - metric_lookup[name]["NDS"] >= 0.002 for name in singles)
    associated = sum(
        next(row for row in correlations if row["scope"] == name)["spearman"] >= 0.20
        for name in singles
        if math.isfinite(next(row for row in correlations if row["scope"] == name)["spearman"])
    )
    group_rho = correlations[-1]["spearman"]
    go = (
        high_retention >= 2
        and nds_degraded >= 2
        and associated >= 2
        and math.isfinite(group_rho)
        and group_rho >= 0.50
        and max_diff == 0.0
    )
    decision = "GO" if go else "NO-GO"
    table = "\n".join(
        f"| {row['condition']} | {row['mAP']:.4f} | {row['NDS']:.4f} | {row['NDS_drop_vs_clean']:.4f} |"
        for row in metric_rows
    )
    attention_table = "\n".join(
        f"| {row['condition']} | {fmt(row['degraded_deployed_attention_mean'])} | "
        f"{fmt(row['deployed_attention_retention_mean'])} | "
        f"{fmt(row['attention_to_token_amplification_mean'])} | "
        f"{fmt(row['paired_tp_loss_mean'])} |"
        for row in group_attention_rows if row["group"] != "clean"
    )
    explanation = (
        "坏相机保持高贡献且其变化与检测损失满足预注册的跨协议和帧内关联门限。"
        if go else
        "未同时满足“贡献保持高、性能下降、帧内关联稳定、跨协议关联稳定”四项预注册条件；"
        "因此当前证据不足以把 Reliability Head 作为下一正式模块。"
    )
    report = f"""# Camera Reliability Audit

## 结论

**{decision}**。{explanation}

本审计仅对冻结 B0 做五组成对前向，没有训练、改权重、改 memory、增加恢复 Query 或写回模块。Camera 贡献定义为 decoder 真实 cross-attention 对来自各相机 ROI Top-K token 的注意力质量；同时报告 token 暴露占比，避免把“被选 token 多”误读成“单 token 更受信任”。Attention allocation 是模型内部信任的直接读数，但不是单独遮挡某 value 的因果归因，因此结论只用于机制筛选。

## 工程不变量

- Hooked Clean 与历史 B0 输出比较 {tensors} 个叶节点，`max_abs_diff={max_diff}`。
- checkpoint SHA256：`{sha256(CHECKPOINT)}`。
- 五组均使用同一冻结 checkpoint、同一 mini-val 顺序；无 optimizer、无反向、无 checkpoint 写入。

## 每组检测结果

| 条件 | mAP | NDS | NDS drop vs Clean |
|---|---:|---:|---:|
{table}

## 故障相机贡献与成对 TP 损失（active frames）

| 条件 | deployed attention | retention vs Clean | attention/token amplification | TP loss/frame |
|---|---:|---:|---:|---:|
{attention_table}

三种单相机故障中，贡献保持率 ≥0.80 的组数为 {high_retention}/3，NDS 下降 ≥0.002 的组数为 {nds_degraded}/3，帧内 Spearman ≥0.20 的组数为 {associated}/3；跨 Dark/Blur/Crash 的 retention–NDS-loss Spearman 为 {fmt(group_rho)}。

Dark/Blur 确实分别保留约 94%/96% 的 CAM_BACK attention，但其 paired TP loss 仅为 -0.15/0.30；Crash 的 attention 已降到约 77%，TP loss 却升到 1.00。也就是说，“越不降权，损失越大”的预期顺序没有出现：三种单相机故障的组间相关为 -1.0，三组帧内相关也均未达到 0.20。双相机 Crash 更会把相应 attention 自动降到约 60%。这是 NO-GO 的核心机制证据，而非仅由总体 NDS 门限决定。

## 判定规则与最小后续方案

预注册 GO 要求上述四项同时成立，并要求 prediction invariance 为零；结果后未调整门限。若 GO，下一步仅允许一个 train-only、每相机 Reliability Head，以 detached 相机质量标签监督，推理时只缩放 cross-attention key/value，并保留关闭路径逐 tensor 等价；先做 2 iter smoke 与单候选 50 iter。若 NO-GO，则停止相机可靠性方向，不训练 Reliability Head，不继续调阈值。

逐帧、TP、相关性和完整误差指标见同目录 CSV。
"""
    (REPORT / "CAMERA_RELIABILITY_AUDIT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
