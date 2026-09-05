#!/usr/bin/env python3
"""Classify fault-induced lost GT by representation, coverage, or ranking."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import mmcv
import numpy as np
import torch
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes

from analysis.gt_query_survival import (
    flat_class_rank,
    geometry_statistics,
    projected_feature_support,
)


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/stage4/gt_query_survival_audit"
REPORT = ROOT / "reports/stage4/gt_query_survival_audit"
CHECKPOINT = ROOT / "outputs/stage3/observability_distillation/b0/iter_969.pth"
CONFIG = ROOT / "configs/stage4/gt_query_survival_b0_audit.py"
REFERENCE = ROOT / "outputs/stage3/counterfactual_view_deficit_audit/invariance_original/predictions.pkl"
GROUPS = {
    "clean": "Clean",
    "dark_back": "CAM_BACK Dark",
    "blur_back": "CAM_BACK Blur",
    "crash_back": "CAM_BACK Crash",
}
CLASS_NAMES = (
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
K = 100


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty report {name}")
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trace(group: str) -> dict[str, dict]:
    output = {}
    for path in sorted((RUN / group / "trace").glob("*.npz")):
        with np.load(path) as value:
            output[str(value["sample_token"])] = {
                key: value[key].copy() for key in value.files
            }
    if len(output) != 81:
        raise RuntimeError(f"expected 81 {group} traces, got {len(output)}")
    return output


def recursive_max_diff(left, right) -> tuple[float, int]:
    if hasattr(left, "tensor") or hasattr(right, "tensor"):
        if not (hasattr(left, "tensor") and hasattr(right, "tensor")):
            return float("inf"), 0
        return recursive_max_diff(left.tensor, right.tensor)
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or left.shape != right.shape:
            return float("inf"), 0
        return (float((left.cpu() - right.cpu()).abs().max()) if left.numel() else 0.0), 1
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            return float("inf"), 0
        values = [recursive_max_diff(left[k], right[k]) for k in left]
    elif isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            return float("inf"), 0
        values = [recursive_max_diff(a, b) for a, b in zip(left, right)]
    else:
        return (0.0 if left == right else float("inf")), 1
    return max((v[0] for v in values), default=0.0), sum(v[1] for v in values)


def local_gt(nusc: NuScenes, token: str) -> list[dict]:
    sample = nusc.get("sample", token)
    _, boxes, _ = nusc.get_sample_data(sample["data"]["LIDAR_TOP"])
    output = []
    for box in boxes:
        name = category_to_detection_name(box.name)
        if name not in CLASS_TO_INDEX:
            continue
        output.append({
            "token": box.token,
            "name": name,
            "label": CLASS_TO_INDEX[name],
            "center": np.asarray(box.center, dtype=np.float64),
            "size": np.asarray(box.wlh, dtype=np.float64),
            "yaw": float(box.orientation.yaw_pitch_roll[0]),
        })
    return output


def official_matches(nusc: NuScenes, token: str, payload: dict) -> set[str]:
    gt = []
    sample = nusc.get("sample", token)
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        name = category_to_detection_name(ann["category_name"])
        if name in CLASS_TO_INDEX:
            gt.append((ann_token, name, np.asarray(ann["translation"][:2], float)))
    predictions = [
        value for value in payload["results"].get(token, [])
        if float(value["detection_score"]) >= 0.1
    ]
    pairs = []
    for gi, (_, name, center) in enumerate(gt):
        for pi, pred in enumerate(predictions):
            if pred["detection_name"] != name:
                continue
            distance = float(np.linalg.norm(
                np.asarray(pred["translation"][:2], float) - center
            ))
            if distance <= 2.0:
                pairs.append((distance, gi, pi))
    used_gt, used_pred, matched = set(), set(), set()
    for _, gi, pi in sorted(pairs):
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi); used_pred.add(pi); matched.add(gt[gi][0])
    return matched


def layer_record(frame: dict, gt: dict, layer: int) -> dict:
    boxes = frame["layer_boxes"][layer]
    logits = frame["layer_logits"][layer].astype(np.float32)
    geometry = geometry_statistics(boxes, gt["center"], gt["size"], gt["yaw"])
    distance = np.linalg.norm(boxes[:, :3] - gt["center"][:3], axis=1)
    near = np.flatnonzero(np.isfinite(distance) & (distance <= 2.0))
    if geometry["best_query"] >= 0:
        query = geometry["best_query"]
        score = float(1 / (1 + np.exp(-np.clip(logits[query, gt["label"]], -30, 30))))
        rank = flat_class_rank(logits, query, gt["label"])
        near_ranks = [flat_class_rank(logits, int(q), gt["label"]) for q in near]
        best_rank = min(near_ranks)
    else:
        query, score, rank, best_rank = -1, float("nan"), -1, -1
    return {
        **geometry,
        "best_geometry_query": query,
        "gt_class_score": score,
        "best_geometry_query_rank": rank,
        "best_near_query_rank": best_rank,
        "geometry_survives": bool(near.size),
        "rank_survives": bool(near.size and best_rank <= K),
    }


def evaluate(group: str, nusc: NuScenes) -> dict:
    target = RUN / group / "evaluation"
    summary = target / "metrics_summary.json"
    if not summary.is_file():
        target.mkdir(parents=True, exist_ok=True)
        evaluator = NuScenesEval(
            nusc, config_factory("detection_cvpr_2019"),
            result_path=str(RUN / group / "formatted/pts_bbox/results_nusc.json"),
            eval_set="mini_val", output_dir=str(target), verbose=False,
        )
        evaluator.main(render_curves=False)
    return json.loads(summary.read_text())


def median(values) -> float:
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return float(np.median(values)) if values.size else float("nan")


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version="v1.0-mini", dataroot=str(ROOT / "data/nuscenes-mini"), verbose=False)
    traces = {group: load_trace(group) for group in GROUPS}
    clean = traces["clean"]
    if any(set(value) != set(clean) for value in traces.values()):
        raise RuntimeError("paired protocols have different samples")
    reference = mmcv.load(str(REFERENCE))
    hooked = mmcv.load(str(RUN / "clean/predictions.pkl"))
    max_diff, leaves = recursive_max_diff(reference, hooked)
    if max_diff != 0:
        raise RuntimeError(f"trace changed B0 outputs: {max_diff}")

    payloads = {
        group: json.loads((RUN / group / "formatted/pts_bbox/results_nusc.json").read_text())
        for group in GROUPS
    }
    metrics = {group: evaluate(group, nusc) for group in GROUPS}
    protocol_rows, failure_rows, layer_rows, first_rows = [], [], [], []
    outcome_counts = defaultdict(Counter)
    rank_deltas = defaultdict(list)
    all_rank_deltas = []

    for group in ("dark_back", "blur_back", "crash_back"):
        for token, fault_frame in traces[group].items():
            if not 3 <= int(fault_frame["frame_idx"]) <= 12:
                continue
            clean_frame = clean[token]
            gt_values = local_gt(nusc, token)
            clean_match = official_matches(nusc, token, payloads["clean"])
            fault_match = official_matches(nusc, token, payloads[group])
            lost = clean_match - fault_match
            for gt in gt_values:
                if gt["token"] not in lost:
                    continue
                clean_feature = projected_feature_support(
                    gt["center"], clean_frame["lidar2img"],
                    clean_frame["selected_token_index"], clean_frame["selected_token_norm"],
                    clean_frame["feature_hw"], clean_frame["image_hw"],
                )
                fault_feature = projected_feature_support(
                    gt["center"], fault_frame["lidar2img"],
                    fault_frame["selected_token_index"], fault_frame["selected_token_norm"],
                    fault_frame["feature_hw"], fault_frame["image_hw"],
                )
                ratio = fault_feature["best_feature_norm"] / max(
                    clean_feature["best_feature_norm"], 1e-12
                )
                representation_failure = (
                    clean_feature["supported_cameras"] > 0
                    and (fault_feature["supported_cameras"] == 0 or ratio < 0.20)
                )
                clean_layers, fault_layers = [], []
                first_failure = "final_output"
                for layer in range(fault_frame["layer_logits"].shape[0]):
                    clean_value = layer_record(clean_frame, gt, layer)
                    fault_value = layer_record(fault_frame, gt, layer)
                    clean_layers.append(clean_value); fault_layers.append(fault_value)
                    for condition, value in (("clean", clean_value), ("fault", fault_value)):
                        layer_rows.append({
                            "protocol": group,
                            "condition": condition,
                            "sample_token": token,
                            "scene_token": str(fault_frame["scene_token"]),
                            "frame_idx": int(fault_frame["frame_idx"]),
                            "gt_token": gt["token"],
                            "gt_class": gt["name"],
                            "decoder_layer": layer,
                            **value,
                        })
                    if first_failure == "final_output":
                        if clean_value["geometry_survives"] and not fault_value["geometry_survives"]:
                            first_failure = f"decoder_{layer}_candidate"
                        elif clean_value["rank_survives"] and not fault_value["rank_survives"]:
                            first_failure = f"decoder_{layer}_ranking"
                final_clean, final_fault = clean_layers[-1], fault_layers[-1]
                if representation_failure:
                    category = "Representation Failure"
                    first_failure = "pre_decoder_feature"
                elif not final_fault["geometry_survives"]:
                    category = "Candidate Coverage Failure"
                else:
                    category = "Ranking/Top-K Failure"
                strict_rank = bool(
                    final_fault["geometry_survives"]
                    and final_fault["best_near_query_rank"] > K
                )
                rank_delta = (
                    final_fault["best_near_query_rank"] - final_clean["best_near_query_rank"]
                    if final_fault["best_near_query_rank"] > 0
                    and final_clean["best_near_query_rank"] > 0 else float("nan")
                )
                if np.isfinite(rank_delta):
                    rank_deltas[group].append(rank_delta); all_rank_deltas.append(rank_delta)
                row = {
                    "protocol": group,
                    "sample_token": token,
                    "scene_token": str(fault_frame["scene_token"]),
                    "frame_idx": int(fault_frame["frame_idx"]),
                    "gt_token": gt["token"],
                    "gt_class": gt["name"],
                    "failure_category": category,
                    "strict_geometry_qualified_rank_out": strict_rank,
                    "first_failure_stage": first_failure,
                    "clean_feature_supported_cameras": clean_feature["supported_cameras"],
                    "fault_feature_supported_cameras": fault_feature["supported_cameras"],
                    "clean_best_feature_norm": clean_feature["best_feature_norm"],
                    "fault_best_feature_norm": fault_feature["best_feature_norm"],
                    "feature_norm_ratio": ratio,
                    "clean_final_near_queries": final_clean["near_count"],
                    "fault_final_near_queries": final_fault["near_count"],
                    "clean_final_best_center_distance": final_clean["center_distance"],
                    "fault_final_best_center_distance": final_fault["center_distance"],
                    "clean_final_best_rank": final_clean["best_near_query_rank"],
                    "fault_final_best_rank": final_fault["best_near_query_rank"],
                    "rank_delta": rank_delta,
                }
                failure_rows.append(row)
                first_rows.append({key: row[key] for key in (
                    "protocol", "sample_token", "scene_token", "frame_idx", "gt_token",
                    "gt_class", "failure_category", "first_failure_stage",
                )})
                outcome_counts[group][category] += 1
                outcome_counts[group]["strict_rank_out"] += int(strict_rank)

    total_counter = Counter()
    for group in ("dark_back", "blur_back", "crash_back"):
        counts = outcome_counts[group]
        total = sum(counts[name] for name in (
            "Representation Failure", "Candidate Coverage Failure", "Ranking/Top-K Failure"
        ))
        total_counter.update({key: value for key, value in counts.items()})
        protocol_rows.append({
            "protocol": group,
            "condition": GROUPS[group],
            "fault_induced_lost_gt": total,
            "representation_failure": counts["Representation Failure"],
            "representation_ratio": counts["Representation Failure"] / max(total, 1),
            "candidate_coverage_failure": counts["Candidate Coverage Failure"],
            "candidate_coverage_ratio": counts["Candidate Coverage Failure"] / max(total, 1),
            "ranking_topk_failure": counts["Ranking/Top-K Failure"],
            "ranking_topk_ratio": counts["Ranking/Top-K Failure"] / max(total, 1),
            "strict_geometry_qualified_rank_out": counts["strict_rank_out"],
            "strict_rank_out_ratio": counts["strict_rank_out"] / max(total, 1),
            "median_final_rank_delta": median(rank_deltas[group]),
            "mAP": metrics[group]["mean_ap"],
            "NDS": metrics[group]["nd_score"],
        })
    total = sum(total_counter[name] for name in (
        "Representation Failure", "Candidate Coverage Failure", "Ranking/Top-K Failure"
    ))
    protocol_rows.append({
        "protocol": "aggregate",
        "condition": "All single-camera faults",
        "fault_induced_lost_gt": total,
        "representation_failure": total_counter["Representation Failure"],
        "representation_ratio": total_counter["Representation Failure"] / max(total, 1),
        "candidate_coverage_failure": total_counter["Candidate Coverage Failure"],
        "candidate_coverage_ratio": total_counter["Candidate Coverage Failure"] / max(total, 1),
        "ranking_topk_failure": total_counter["Ranking/Top-K Failure"],
        "ranking_topk_ratio": total_counter["Ranking/Top-K Failure"] / max(total, 1),
        "strict_geometry_qualified_rank_out": total_counter["strict_rank_out"],
        "strict_rank_out_ratio": total_counter["strict_rank_out"] / max(total, 1),
        "median_final_rank_delta": median(all_rank_deltas),
        "mAP": "", "NDS": "",
    })
    aggregate = protocol_rows[-1]
    positive_protocols = sum(
        np.isfinite(median(rank_deltas[g])) and median(rank_deltas[g]) > 0
        for g in ("dark_back", "blur_back", "crash_back")
    )
    go = (
        aggregate["ranking_topk_ratio"] >= 0.50
        and aggregate["ranking_topk_failure"] > aggregate["representation_failure"]
        and aggregate["ranking_topk_failure"] > aggregate["candidate_coverage_failure"]
        and aggregate["strict_rank_out_ratio"] >= 0.40
        and aggregate["median_final_rank_delta"] >= 50
        and positive_protocols >= 2
        and max_diff == 0
    )
    layer_summary = []
    for group in ("dark_back", "blur_back", "crash_back"):
        for condition in ("clean", "fault"):
            for layer in range(6):
                values = [row for row in layer_rows if row["protocol"] == group
                          and row["condition"] == condition
                          and row["decoder_layer"] == layer]
                layer_summary.append({
                    "protocol": group,
                    "condition": condition,
                    "decoder_layer": layer,
                    "gt_records": len(values),
                    "geometry_survival_ratio": np.mean([
                        row["geometry_survives"] for row in values
                    ]),
                    "rank_topk_survival_ratio": np.mean([
                        row["rank_survives"] for row in values
                    ]),
                    "median_gt_near_query_count": median([
                        row["near_count"] for row in values
                    ]),
                    "median_best_center_distance": median([
                        row["center_distance"] for row in values
                    ]),
                    "median_best_geometry_cost": median([
                        row["geometry_cost"] for row in values
                    ]),
                    "median_best_near_query_rank": median([
                        row["best_near_query_rank"] for row in values
                    ]),
                })

    first_summary = []
    first_counter = defaultdict(Counter)
    for row in first_rows:
        first_counter[row["protocol"]][row["first_failure_stage"]] += 1
        first_counter["aggregate"][row["first_failure_stage"]] += 1
    for group in ("dark_back", "blur_back", "crash_back", "aggregate"):
        subtotal = sum(first_counter[group].values())
        for stage, count in sorted(first_counter[group].items()):
            first_summary.append({
                "protocol": group,
                "first_failure_stage": stage,
                "count": count,
                "ratio": count / max(subtotal, 1),
            })

    mechanism_rows = []
    for group in ("dark_back", "blur_back", "crash_back", "aggregate"):
        values = failure_rows if group == "aggregate" else [
            row for row in failure_rows if row["protocol"] == group
        ]
        mechanism_rows.append({
            "protocol": group,
            "fault_induced_lost_gt": len(values),
            "clean_feature_support_ratio": np.mean([
                row["clean_feature_supported_cameras"] > 0 for row in values
            ]),
            "fault_feature_support_ratio": np.mean([
                row["fault_feature_supported_cameras"] > 0 for row in values
            ]),
            "median_feature_norm_ratio": median([
                row["feature_norm_ratio"] for row in values
            ]),
            "clean_final_geometry_survival_ratio": np.mean([
                row["clean_final_near_queries"] > 0 for row in values
            ]),
            "fault_final_geometry_survival_ratio": np.mean([
                row["fault_final_near_queries"] > 0 for row in values
            ]),
            "median_clean_final_near_queries": median([
                row["clean_final_near_queries"] for row in values
            ]),
            "median_fault_final_near_queries": median([
                row["fault_final_near_queries"] for row in values
            ]),
            "median_clean_best_rank": median([
                row["clean_final_best_rank"] for row in values
            ]),
            "median_fault_best_rank": median([
                row["fault_final_best_rank"] for row in values
            ]),
        })

    typical = []
    for group in ("dark_back", "blur_back", "crash_back"):
        for category in (
            "Representation Failure", "Candidate Coverage Failure", "Ranking/Top-K Failure"
        ):
            values = [r for r in failure_rows if r["protocol"] == group and r["failure_category"] == category]
            values.sort(key=lambda r: (
                r["rank_delta"] if np.isfinite(r["rank_delta"]) else -1,
                -r["feature_norm_ratio"],
            ), reverse=True)
            typical.extend(values[:5])

    manifest = []
    for group, label in GROUPS.items():
        manifest.append({
            "group": group, "condition": label, "frames": len(traces[group]),
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "checkpoint_sha256": sha256(CHECKPOINT),
            "config": str(CONFIG.relative_to(ROOT)),
            "config_sha256": sha256(CONFIG),
            "training": False,
            "exit_code": (RUN / group / "exit_code.txt").read_text().strip(),
        })
    write_csv("experiment_manifest.csv", manifest)
    write_csv("prediction_invariance.csv", [{
        "comparison": "historical_B0_vs_survival_hook_clean",
        "leaves": leaves, "max_abs_diff": max_diff, "exact": max_diff == 0,
    }])
    write_csv("failure_taxonomy_by_protocol.csv", protocol_rows)
    write_csv("per_gt_failure.csv", failure_rows)
    write_csv("per_layer_survival.csv", layer_rows)
    write_csv("layer_summary.csv", layer_summary)
    write_csv("first_failure_stage.csv", first_rows)
    write_csv("first_failure_summary.csv", first_summary)
    write_csv("feature_candidate_summary.csv", mechanism_rows)
    write_csv("typical_cases.csv", typical)

    rows = "\n".join(
        f"| {r['condition']} | {r['fault_induced_lost_gt']} | {r['representation_ratio']:.1%} | "
        f"{r['candidate_coverage_ratio']:.1%} | {r['ranking_topk_ratio']:.1%} | "
        f"{r['strict_rank_out_ratio']:.1%} | {r['median_final_rank_delta']:.1f} |"
        for r in protocol_rows
    )
    decision = "GO" if go else "NO-GO"
    direction = (
        "允许下一任务只设计 Hard-positive/Top-K boundary ranking。"
        if go else
        "不允许进入 ranking loss 设计；应依据占比最高的上游失败转向鲁棒表征或候选监督。"
    )
    report = f"""# Fault-induced GT Query Survival Audit

## Decision

**{decision}**。{direction}

本审计使用冻结 B0 对 Clean/Dark/Blur/Crash 做相同 sample token 的成对前向。分析对象严格限定为 Clean 中已正确检测、Fault 中丢失的 GT；没有训练、恢复 Query、Reliability Head、memory/writeback 修改或新推理模块。Hooked Clean 与历史 B0 比较 {leaves} 个预测叶节点，`max_abs_diff={max_diff}`。

## Failure taxonomy

| Protocol | Lost GT | Representation | Candidate coverage | Ranking/Top-K | Strict geo-qualified rank>K | Median rank delta |
|---|---:|---:|---:|---:|---:|---:|
{rows}

严格的核心问题“几何仍合格但 rank 跌出 K=100”占全部 fault-induced lost GT 的 {aggregate['strict_rank_out_ratio']:.1%}。Ranking/Top-K 总类占 {aggregate['ranking_topk_ratio']:.1%}，Representation 与 Candidate Coverage 合计占 {aggregate['representation_ratio'] + aggregate['candidate_coverage_ratio']:.1%}；final geometry 双侧存在时的 rank 中位变化为 {aggregate['median_final_rank_delta']:.1f} 位，三个协议中有 {positive_protocols}/3 呈正向恶化。

聚合后，Fault 路径仍有投影邻近 ROI token 的比例为 {mechanism_rows[-1]['fault_feature_support_ratio']:.1%}，最终仍有几何合格 query 的比例为 {mechanism_rows[-1]['fault_final_geometry_survival_ratio']:.1%}；GT-near query 数中位数由 Clean 的 {mechanism_rows[-1]['median_clean_final_near_queries']:.1f} 变为 Fault 的 {mechanism_rows[-1]['median_fault_final_near_queries']:.1f}，最佳近邻 GT-class rank 中位数由 {mechanism_rows[-1]['median_clean_best_rank']:.1f} 变为 {mechanism_rows[-1]['median_fault_best_rank']:.1f}。

首次分叉定位中，decoder layer 0 ranking failure 为 {first_counter['aggregate']['decoder_0_ranking']}/{total}（{first_counter['aggregate']['decoder_0_ranking'] / max(total, 1):.1%}），layer 0 candidate failure 为 {first_counter['aggregate']['decoder_0_candidate']}/{total}（{first_counter['aggregate']['decoder_0_candidate'] / max(total, 1):.1%}）；另有 {first_counter['aggregate']['final_output']}/{total}（{first_counter['aggregate']['final_output'] / max(total, 1):.1%}）直到最终一对一输出检查才首次失败。完整层级分解见 `first_failure_summary.csv` 和 `layer_summary.csv`。

## Interpretation

`Representation Failure` 要求 Clean 投影附近存在 ROI Top-K token，而 Fault 附近 token 消失或最大特征范数低于配对 Clean 的20%；`Candidate Coverage Failure` 表示这种 feature support 仍在但最终没有2m内 query；`Ranking/Top-K Failure` 表示最终几何候选仍在但没有保住部署输出，CSV另给严格 rank>K 子集。该划分是冻结模型的阶段定位，不把 feature norm 或 attention 当作因果证明。

Go/No-Go 完全使用推理前的 `PRE_REGISTRATION.md`，结果后未修改阈值。逐GT、逐层、首次失败层和典型案例见同目录CSV。
"""
    (REPORT / "GT_QUERY_SURVIVAL_AUDIT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
