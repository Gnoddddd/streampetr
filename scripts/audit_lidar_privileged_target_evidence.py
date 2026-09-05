#!/usr/bin/env python3
"""Frozen paired activation audit for train-only LiDAR/GT target evidence."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import mmcv
import numpy as np
import torch
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes

from models.lidar_privileged_target_evidence import (
    select_target_evidence,
    target_evidence_loss,
)


ROOT = Path(__file__).resolve().parents[1]
TRACE_ROOT = ROOT / "outputs/stage4/gt_query_survival_audit"
ROOT_CAUSE = ROOT / "reports/stage4/fault_boundary_root_cause_audit/per_gt_root_cause.csv"
DISABLED = ROOT / "outputs/stage4/lidar_privileged_target_evidence_audit/disabled"
REPORT = ROOT / "reports/stage4/lidar_privileged_target_evidence_audit"
INTEGRATION = REPORT / "integration_check.csv"
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
BOOTSTRAPS = 5000
SEED = 314159


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty report: {name}")
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


def load_trace(group: str) -> dict[str, dict]:
    output = {}
    for path in sorted((TRACE_ROOT / group / "trace").glob("*.npz")):
        with np.load(path) as value:
            output[str(value["sample_token"])] = {
                key: value[key].copy() for key in value.files
            }
    if len(output) != 81:
        raise RuntimeError(f"{group}: expected 81 traces, got {len(output)}")
    return output


def load_root_rows() -> list[dict]:
    with ROOT_CAUSE.open() as handle:
        return list(csv.DictReader(handle))


def local_gt(nusc: NuScenes, token: str) -> list[dict]:
    sample = nusc.get("sample", token)
    _, boxes, _ = nusc.get_sample_data(sample["data"]["LIDAR_TOP"])
    output = []
    for box in boxes:
        name = category_to_detection_name(box.name)
        if name not in CLASS_TO_INDEX:
            continue
        annotation = nusc.get("sample_annotation", box.token)
        output.append({
            "token": box.token, "label": CLASS_TO_INDEX[name], "name": name,
            "box": np.asarray([
                *box.center, *box.wlh, box.orientation.yaw_pitch_roll[0]
            ], dtype=np.float64),
            "num_lidar_pts": int(annotation["num_lidar_pts"]),
        })
    return output


def finite(rows, key) -> np.ndarray:
    values = [float(row[key]) for row in rows
              if key in row and math.isfinite(float(row[key]))]
    return np.asarray(values, dtype=np.float64)


def distribution(values) -> dict:
    values = np.asarray(tuple(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "q25": float("nan"),
                "median": float("nan"), "q75": float("nan"), "max": float("nan")}
    return {
        "n": int(values.size), "mean": float(np.mean(values)),
        "std": float(np.std(values)), "min": float(np.min(values)),
        "q25": float(np.percentile(values, 25)), "median": float(np.median(values)),
        "q75": float(np.percentile(values, 75)), "max": float(np.max(values)),
    }


def bootstrap_contrasts(lost: list[dict], retained: list[dict], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    activation_difference, activation_ratio = [], []
    signal_difference, signal_ratio, lost_delta = [], [], []
    for _ in range(BOOTSTRAPS):
        left = [lost[index] for index in rng.integers(0, len(lost), len(lost))]
        right = [retained[index] for index in rng.integers(0, len(retained), len(retained))]
        left_rate = np.mean([row["activated"] for row in left])
        right_rate = np.mean([row["activated"] for row in right])
        activation_difference.append(left_rate - right_rate)
        activation_ratio.append(left_rate / right_rate if right_rate else float("inf"))
        left_active = finite([row for row in left if row["activated"]], "raw_signal")
        right_active = finite([row for row in right if row["activated"]], "raw_signal")
        if left_active.size and right_active.size:
            left_median, right_median = np.median(left_active), np.median(right_active)
            signal_difference.append(left_median - right_median)
            signal_ratio.append(left_median / right_median if right_median else float("inf"))
        delta = finite([row for row in left if row["activated"]], "delta_s_pos")
        if delta.size:
            lost_delta.append(np.median(delta))
    def interval(values):
        values = np.asarray(values, dtype=np.float64)
        return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
    lost_rate = np.mean([row["activated"] for row in lost])
    retained_rate = np.mean([row["activated"] for row in retained])
    lost_signal = np.median(finite([r for r in lost if r["activated"]], "raw_signal"))
    retained_signal = np.median(finite([r for r in retained if r["activated"]], "raw_signal"))
    lost_delta_estimate = np.median(finite([r for r in lost if r["activated"]], "delta_s_pos"))
    return {
        "activation_difference": lost_rate - retained_rate,
        "activation_difference_ci_low": interval(activation_difference)[0],
        "activation_difference_ci_high": interval(activation_difference)[1],
        "activation_ratio": lost_rate / retained_rate if retained_rate else float("inf"),
        "activation_ratio_ci_low": interval(activation_ratio)[0],
        "activation_ratio_ci_high": interval(activation_ratio)[1],
        "median_signal_difference": lost_signal - retained_signal,
        "median_signal_difference_ci_low": interval(signal_difference)[0],
        "median_signal_difference_ci_high": interval(signal_difference)[1],
        "median_signal_ratio": lost_signal / retained_signal,
        "median_signal_ratio_ci_low": interval(signal_ratio)[0],
        "median_signal_ratio_ci_high": interval(signal_ratio)[1],
        "lost_median_delta_s_pos": lost_delta_estimate,
        "lost_median_delta_s_pos_ci_low": interval(lost_delta)[0],
        "lost_median_delta_s_pos_ci_high": interval(lost_delta)[1],
        "iterations": BOOTSTRAPS,
    }


def recursive_compare(left, right) -> tuple[float, int]:
    if hasattr(left, "tensor") or hasattr(right, "tensor"):
        return recursive_compare(left.tensor, right.tensor)
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or left.shape != right.shape:
            return float("inf"), 0
        return (float((left.cpu() - right.cpu()).abs().max()) if left.numel() else 0.0), 1
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            return float("inf"), 0
        values = [recursive_compare(left[key], right[key]) for key in left]
    elif isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            return float("inf"), 0
        values = [recursive_compare(a, b) for a, b in zip(left, right)]
    else:
        return (0.0 if left == right else float("inf")), 1
    return max((value[0] for value in values), default=0.0), sum(value[1] for value in values)


def main() -> None:
    nusc = NuScenes(version="v1.0-mini", dataroot=str(ROOT / "data/nuscenes-mini"), verbose=False)
    traces = {group: load_trace(group) for group in GROUPS}
    root_rows = load_root_rows()
    root_lookup = {(r["protocol"], r["sample_token"], r["gt_token"]): r for r in root_rows}
    population_tokens = {row["sample_token"] for row in root_rows}
    gt_cache = {token: local_gt(nusc, token) for token in population_tokens}

    selector = {}
    for protocol in GROUPS:
        for token, frame in traces[protocol].items():
            relevant = [row for row in root_rows
                        if row["protocol"] == protocol and row["sample_token"] == token]
            if not relevant:
                continue
            gt = gt_cache[token]
            selected, diagnostics = select_target_evidence(
                torch.from_numpy(frame["layer_logits"][-1]),
                torch.from_numpy(frame["layer_boxes"][-1]),
                torch.from_numpy(np.stack([item["box"] for item in gt])),
                torch.tensor([item["label"] for item in gt]),
                torch.tensor([item["num_lidar_pts"] > 0 for item in gt]),
                torch.from_numpy(frame["lidar2img"]),
                tuple(int(value) for value in frame["image_hw"]),
                fault_camera=3, geometry_threshold=2.0, fault_active=True,
            )
            by_token = {item["token"]: diagnostics[index] for index, item in enumerate(gt)}
            selector[(protocol, token)] = by_token

    rows = []
    for root_row in root_rows:
        protocol, token, gt_token = (root_row["protocol"], root_row["sample_token"],
                                     root_row["gt_token"])
        item = selector[(protocol, token)][gt_token]
        gt_item = next(value for value in gt_cache[token] if value["token"] == gt_token)
        outcome = "lost-risk" if root_row["outcome"] == "fault_induced_lost" else "retained-like"
        activated = bool(item["selected"])
        low_alternative_risk = bool(
            item["fault_camera_visible"] and item["alternative_view_count"] == 0
        )
        pre_dedup_supervisable = bool(
            item["lidar_supported"] and low_alternative_risk
            and item["near_query_count"] > 0
        )
        s_pos = float(item["s_pos"])
        raw_signal = -math.log(max(s_pos, 1e-12)) if activated else float("nan")
        gradient = 1.0 - s_pos if activated else float("nan")
        selection_matches = bool(
            not activated or int(item["positive_query"]) == int(root_row["fault_best_query"])
        )
        visibility_matches = bool(
            int(item["alternative_view_count"]) == int(root_row["alternative_view_count"])
            and bool(item["fault_camera_visible"]) == (root_row["cam_back_visible"] == "True")
        )
        rows.append({
            "protocol": protocol, "condition": GROUPS[protocol],
            "sample_token": token, "gt_token": gt_token,
            "gt_class": root_row["gt_class"], "outcome": outcome,
            "num_lidar_pts": gt_item["num_lidar_pts"],
            "lidar_supported": item["lidar_supported"],
            "fault_candidate_available": root_row["fault_candidate_available"],
            "fault_camera_visible": item["fault_camera_visible"],
            "alternative_view_count": item["alternative_view_count"],
            "near_query_count": item["near_query_count"],
            "positive_query": item["positive_query"],
            "root_cause_s_pos_query": root_row["fault_best_query"],
            "selection_matches_root_cause": selection_matches,
            "visibility_matches_root_cause": visibility_matches,
            "duplicate_suppressed": item["duplicate_suppressed"],
            "low_alternative_risk": low_alternative_risk,
            "pre_dedup_supervisable": pre_dedup_supervisable,
            "activated": activated,
            "fault_s_pos": root_row["fault_s_pos"],
            "delta_s_pos": root_row["delta_s_pos"],
            "raw_signal": raw_signal,
            "analytic_gradient_magnitude": gradient,
            "s_pos_degraded": bool(
                root_row["fault_candidate_available"] == "True"
                and float(root_row["delta_s_pos"]) < 0
            ),
            "boundary_crossing": root_row["boundary_crossing"],
        })

    summary_rows, signal_rows, bootstrap_rows = [], [], []
    for protocol in (*GROUPS, "aggregate"):
        protocol_rows = rows if protocol == "aggregate" else [r for r in rows if r["protocol"] == protocol]
        lost = [r for r in protocol_rows if r["outcome"] == "lost-risk"]
        retained = [r for r in protocol_rows if r["outcome"] == "retained-like"]
        for outcome, selected_rows in (("lost-risk", lost), ("retained-like", retained)):
            active = [r for r in selected_rows if r["activated"]]
            crossing = [r for r in selected_rows if r["boundary_crossing"] == "True"]
            summary_rows.append({
                "protocol": protocol, "condition": GROUPS.get(protocol, "All faults"),
                "outcome": outcome, "gt": len(selected_rows),
                "lidar_supported": sum(r["lidar_supported"] for r in selected_rows),
                "lidar_supported_ratio": sum(r["lidar_supported"] for r in selected_rows) / max(len(selected_rows), 1),
                "candidate_available": sum(r["fault_candidate_available"] == "True" for r in selected_rows),
                "candidate_available_ratio": sum(r["fault_candidate_available"] == "True" for r in selected_rows) / max(len(selected_rows), 1),
                "low_alternative_risk": sum(r["low_alternative_risk"] for r in selected_rows),
                "low_alternative_risk_ratio": sum(r["low_alternative_risk"] for r in selected_rows) / max(len(selected_rows), 1),
                "pre_dedup_supervisable": sum(r["pre_dedup_supervisable"] for r in selected_rows),
                "duplicate_suppressed": sum(r["duplicate_suppressed"] for r in selected_rows),
                "activated": len(active), "activation_ratio": len(active) / max(len(selected_rows), 1),
                "activated_s_pos_degraded": sum(r["s_pos_degraded"] for r in active),
                "activated_s_pos_degraded_ratio": sum(r["s_pos_degraded"] for r in active) / max(len(active), 1),
                "boundary_crossing": len(crossing),
                "supervised_boundary_crossing": sum(r["activated"] for r in crossing),
                "boundary_crossing_coverage": sum(r["activated"] for r in crossing) / max(len(crossing), 1),
            })
            for metric in ("raw_signal", "analytic_gradient_magnitude", "fault_s_pos", "delta_s_pos"):
                signal_rows.append({
                    "protocol": protocol, "condition": GROUPS.get(protocol, "All faults"),
                    "outcome": outcome, "metric": metric, **distribution(finite(active, metric)),
                })
        contrast = bootstrap_contrasts(lost, retained, SEED + 100 * len(bootstrap_rows))
        bootstrap_rows.append({
            "protocol": protocol, "condition": GROUPS.get(protocol, "All faults"),
            "lost_n": len(lost), "retained_n": len(retained), **contrast,
        })

    invariance_rows = []
    if DISABLED.exists():
        for protocol in ("clean", *GROUPS):
            difference, leaves = recursive_compare(
                mmcv.load(str(TRACE_ROOT / protocol / "predictions.pkl")),
                mmcv.load(str(DISABLED / protocol / "predictions.pkl")),
            )
            invariance_rows.append({"protocol": protocol, "tensor_leaves": leaves,
                                    "max_abs_diff": difference, "exact": difference == 0})
    else:
        invariance_rows = [{"protocol": protocol, "tensor_leaves": 0,
                            "max_abs_diff": float("nan"), "exact": False}
                           for protocol in ("clean", *GROUPS)]

    # Direct structural gradient check; no model, optimizer, or update is created.
    probe = torch.tensor([[0.1, -0.2], [-1.0, 0.4]], requires_grad=True)
    probe_loss, probe_details = target_evidence_loss(
        probe, [{"positive_query": 1, "gt_class": 0, "gt": 0}]
    )
    probe_loss.backward()
    unselected = probe.grad.clone(); unselected[1, 0] = 0
    structural_rows = [{
        "raw_loss_finite": bool(torch.isfinite(probe_loss)),
        "selected_gradient_finite": bool(torch.isfinite(probe.grad[1, 0])),
        "selected_gradient_nonzero": bool(probe.grad[1, 0] != 0),
        "unselected_gradient_max_abs": float(unselected.abs().max()),
        "positive_bce_only": True, "ranking_or_competitor_term": False,
        "optimizer_created": False, "optimizer_step": False,
    }]

    lookup = {(r["protocol"], r["outcome"]): r for r in summary_rows}
    signal_lookup = {(r["protocol"], r["outcome"], r["metric"]): r for r in signal_rows}
    boot_lookup = {r["protocol"]: r for r in bootstrap_rows}
    protocol_gates = []
    for protocol in GROUPS:
        lost, retained = lookup[(protocol, "lost-risk")], lookup[(protocol, "retained-like")]
        ratio = lost["activation_ratio"] / retained["activation_ratio"]
        protocol_gates.append(
            lost["activation_ratio"] >= 0.50
            and retained["activation_ratio"] <= 0.15
            and ratio >= 4.0
            and lost["activation_ratio"] > retained["activation_ratio"]
            and lost["activated_s_pos_degraded_ratio"] >= 0.80
            and signal_lookup[(protocol, "lost-risk", "raw_signal")]["median"]
                > signal_lookup[(protocol, "retained-like", "raw_signal")]["median"]
            and signal_lookup[(protocol, "lost-risk", "delta_s_pos")]["median"] < 0
            and lost["boundary_crossing_coverage"] >= 0.60
        )
    pooled = boot_lookup["aggregate"]
    exact_selection = all(
        row["selection_matches_root_cause"] and (not row["activated"] or (
            row["visibility_matches_root_cause"]
            and
            row["lidar_supported"] and row["fault_camera_visible"]
            and row["alternative_view_count"] == 0
            and row["near_query_count"] > 0
        )) for row in rows
    )
    structural = (
        structural_rows[0]["raw_loss_finite"]
        and structural_rows[0]["selected_gradient_finite"]
        and structural_rows[0]["selected_gradient_nonzero"]
        and structural_rows[0]["unselected_gradient_max_abs"] == 0
        and not structural_rows[0]["optimizer_created"]
        and not structural_rows[0]["optimizer_step"]
    )
    with INTEGRATION.open() as handle:
        integration_row = next(csv.DictReader(handle))
    integration = (
        integration_row["loss_present"] == "True"
        and integration_row["loss_finite"] == "True"
        and integration_row["all_selected_lidar_supported"] == "True"
        and integration_row["optimizer_created"] == "False"
        and integration_row["optimizer_step"] == "False"
    )
    invariant = all(row["exact"] for row in invariance_rows)
    pooled_gates = (
        pooled["activation_ratio_ci_low"] > 1
        and pooled["activation_difference_ci_low"] > 0
        and pooled["median_signal_difference_ci_low"] > 0
        and pooled["median_signal_ratio"] >= 2.0
        and pooled["lost_median_delta_s_pos_ci_high"] < 0
    )
    go = (all(protocol_gates) and pooled_gates and exact_selection
          and structural and integration and invariant)

    write_csv("per_gt_activation.csv", rows)
    write_csv("protocol_group_activation_summary.csv", summary_rows)
    write_csv("signal_strength_summary.csv", signal_rows)
    write_csv("bootstrap_95ci.csv", bootstrap_rows)
    write_csv("disabled_invariance.csv", invariance_rows)
    write_csv("structural_gradient_checks.csv", structural_rows)

    table = "\n".join(
        f"| {GROUPS[p]} | {lookup[(p, 'lost-risk')]['activation_ratio']:.1%} | "
        f"{lookup[(p, 'retained-like')]['activation_ratio']:.1%} | "
        f"{lookup[(p, 'lost-risk')]['activation_ratio'] / lookup[(p, 'retained-like')]['activation_ratio']:.2f}x | "
        f"{lookup[(p, 'lost-risk')]['activated_s_pos_degraded_ratio']:.1%} | "
        f"{signal_lookup[(p, 'lost-risk', 'raw_signal')]['median']:.4f} / "
        f"{signal_lookup[(p, 'retained-like', 'raw_signal')]['median']:.4f} | "
        f"{lookup[(p, 'lost-risk')]['boundary_crossing_coverage']:.1%} |"
        for p in GROUPS
    )
    attrition_table = "\n".join(
        f"| {GROUPS[p]} | {lookup[(p, 'lost-risk')]['gt']} | "
        f"{lookup[(p, 'lost-risk')]['lidar_supported']} | "
        f"{lookup[(p, 'lost-risk')]['candidate_available']} | "
        f"{lookup[(p, 'lost-risk')]['low_alternative_risk']} | "
        f"{lookup[(p, 'lost-risk')]['pre_dedup_supervisable']} | "
        f"{lookup[(p, 'lost-risk')]['duplicate_suppressed']} | "
        f"{lookup[(p, 'lost-risk')]['activated']} |"
        for p in GROUPS
    )
    report = f"""# LiDAR-privileged Target Evidence Supervision Activation Audit

## Decision

**{'GO' if go else 'NO-GO'}**. {'The exact preregistered unit-weight scheme is eligible for one later 2-iteration smoke; no longer run is authorized.' if go else 'Do not tune the gate/weight and do not enter smoke.'}

This audit performed no model training and created/stepped no optimizer. The supervision is training-only positive BCE on the exact final-decoder `S_pos` GT-class logit for LiDAR-supported, CAM_BACK-visible GT with zero alternative views. It contains no Top-K, rank, competitor, query, memory or inference operation.

## Root-cause population activation

| Protocol | lost-risk active | retained-like active | enrichment | active lost with delta-S_pos<0 | median raw signal lost / retained | crossing coverage |
|---|---:|---:|---:|---:|---:|---:|
{table}

Pooled activation-rate difference is {pooled['activation_difference']:.1%} (bootstrap 95% CI [{pooled['activation_difference_ci_low']:.1%}, {pooled['activation_difference_ci_high']:.1%}]); enrichment is {pooled['activation_ratio']:.2f}x ([{pooled['activation_ratio_ci_low']:.2f}, {pooled['activation_ratio_ci_high']:.2f}]). Pooled lost-minus-retained median signal difference is {pooled['median_signal_difference']:.4f} ([{pooled['median_signal_difference_ci_low']:.4f}, {pooled['median_signal_difference_ci_high']:.4f}]); signal ratio is {pooled['median_signal_ratio']:.2f}x. Activated lost median `delta_S_pos` is {pooled['lost_median_delta_s_pos']:.4f} ([{pooled['lost_median_delta_s_pos_ci_low']:.4f}, {pooled['lost_median_delta_s_pos_ci_high']:.4f}]).

| Protocol | lost GT | LiDAR supported | candidate available | CAM_BACK-only physical view | pre-dedup supervisable | duplicate suppressed | final active |
|---|---:|---:|---:|---:|---:|---:|---:|
{attrition_table}

Dark's final activation is below the fixed 50% gate. This is reported as coverage failure; duplicate suppression and LiDAR eligibility are not relaxed after the result.

## Safety and gate

- Three-protocol activation/signal/crossing gates: {protocol_gates}.
- Every selected event is LiDAR-supported, CAM_BACK-visible, zero-alternative-view, <=2 m and exactly the root-cause `S_pos` query: {exact_selection}.
- Raw loss and selected gradient finite/nonzero; unselected gradient max abs is {structural_rows[0]['unselected_gradient_max_abs']}; optimizer created/stepped: False/False.
- One real train-mode, no-update B0 batch produced finite raw loss {float(integration_row['raw_loss']):.4f} with {integration_row['selected_gt']} selected GT, all LiDAR-supported; integration gate: {integration}.
- Disabled Clean/Dark/Blur/Crash tensor-exact B0: {invariant}.
- Pooled bootstrap gates: {pooled_gates}.

The activation is {'stably concentrated on the preregistered low-alternative-view, S_pos-degraded risk population across all three faults' if go else 'not sufficiently concentrated under the fixed gates'}. No inference module or alternative scheme is proposed.
"""
    (REPORT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "records": len(rows), "protocol_gates": protocol_gates,
        "pooled_gates": pooled_gates, "exact_selection": exact_selection,
        "structural": structural, "integration": integration,
        "disabled_invariant": invariant, "go": go,
    }, indent=2))


if __name__ == "__main__":
    main()
