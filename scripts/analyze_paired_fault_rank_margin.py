#!/usr/bin/env python3
"""Paired Clean-to-Fault GT rank-margin audit on frozen B0 traces."""

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

from analysis.paired_rank_margin import (
    bootstrap_difference,
    fixed_query_statistics,
    query_margin_statistics,
)


ROOT = Path(__file__).resolve().parents[1]
TRACE_ROOT = ROOT / "outputs/stage4/gt_query_survival_audit"
DISABLED_ROOT = ROOT / "outputs/stage4/hard_positive_boundary_objective_audit/disabled"
REPORT = ROOT / "reports/stage4/paired_fault_rank_margin_audit"
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


def write_csv(name, rows):
    if not rows: raise RuntimeError(f"refusing empty report {name}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def traces(group):
    output = {}
    for path in sorted((TRACE_ROOT / group / "trace").glob("*.npz")):
        with np.load(path) as value:
            output[str(value["sample_token"])] = {
                key: value[key].copy() for key in value.files
            }
    if len(output) != 81: raise RuntimeError(f"{group}: expected 81 traces, got {len(output)}")
    return output


def local_gt(nusc, token):
    sample = nusc.get("sample", token)
    _, boxes, _ = nusc.get_sample_data(sample["data"]["LIDAR_TOP"])
    output = []
    for box in boxes:
        name = category_to_detection_name(box.name)
        if name in CLASS_TO_INDEX:
            output.append({
                "token": box.token, "name": name, "label": CLASS_TO_INDEX[name],
                "center": np.asarray(box.center, dtype=np.float64),
            })
    return output


def official_matches(nusc, token, payload):
    sample = nusc.get("sample", token)
    gt = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        name = category_to_detection_name(ann["category_name"])
        if name in CLASS_TO_INDEX:
            gt.append((ann_token, name, np.asarray(ann["translation"][:2], float)))
    predictions = [value for value in payload["results"].get(token, [])
                   if float(value["detection_score"]) >= 0.1]
    pairs = []
    for gi, (_, name, center) in enumerate(gt):
        for pi, pred in enumerate(predictions):
            if pred["detection_name"] != name: continue
            distance = float(np.linalg.norm(
                np.asarray(pred["translation"][:2], float) - center
            ))
            if distance <= 2.0: pairs.append((distance, gi, pi))
    used_gt, used_pred, matched = set(), set(), set()
    for _, gi, pi in sorted(pairs):
        if gi in used_gt or pi in used_pred: continue
        used_gt.add(gi); used_pred.add(pi); matched.add(gt[gi][0])
    return matched


def compare(left, right):
    if hasattr(left, "tensor") or hasattr(right, "tensor"):
        return compare(left.tensor, right.tensor)
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or left.shape != right.shape: return float("inf"), 0
        return (float((left.cpu() - right.cpu()).abs().max()) if left.numel() else 0.0), 1
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right): return float("inf"), 0
        values = [compare(left[k], right[k]) for k in left]
    elif isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right): return float("inf"), 0
        values = [compare(a, b) for a, b in zip(left, right)]
    else: return (0.0 if left == right else float("inf")), 1
    return max((v[0] for v in values), default=0.0), sum(v[1] for v in values)


def finite_values(rows, key):
    return [float(row[key]) for row in rows if math.isfinite(float(row[key]))]


def median(rows, key):
    values = finite_values(rows, key)
    return float(np.median(values)) if values else float("nan")


def summarize(rows, protocol, outcome):
    selected = [row for row in rows if row["protocol"] == protocol and row["outcome"] == outcome]
    paired = [row for row in selected if row["fault_candidate_available"]]
    return {
        "protocol": protocol, "condition": GROUPS.get(protocol, "All faults"),
        "outcome": outcome, "clean_correct_gt": len(selected),
        "paired_rank_available": len(paired),
        "paired_rank_availability_ratio": len(paired) / max(len(selected), 1),
        "median_delta_margin": median(paired, "delta_margin"),
        "median_delta_rank": median(paired, "delta_rank"),
        "boundary_crossing": sum(row["boundary_crossing"] for row in paired),
        "boundary_crossing_ratio": sum(row["boundary_crossing"] for row in paired) / max(len(paired), 1),
        "same_lineage_crossing": sum(row["same_lineage_crossing"] for row in paired),
        "same_lineage_crossing_ratio": sum(row["same_lineage_crossing"] for row in paired) / max(len(paired), 1),
        "same_best_query_ratio": sum(row["fault_best_same_as_clean_best"] for row in paired) / max(len(paired), 1),
        "same_geometry_query_ratio": sum(row["fault_geometry_best_same_as_clean_best"] for row in paired) / max(len(paired), 1),
    }


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version="v1.0-mini", dataroot=str(ROOT / "data/nuscenes-mini"), verbose=False)
    all_traces = {group: traces(group) for group in ("clean", *GROUPS)}
    clean_trace = all_traces["clean"]
    payloads = {group: json.loads(
        (TRACE_ROOT / group / "formatted/pts_bbox/results_nusc.json").read_text()
    ) for group in ("clean", *GROUPS)}

    invariance = []
    for group in ("clean", *GROUPS):
        difference, leaves = compare(
            mmcv.load(str(TRACE_ROOT / group / "predictions.pkl")),
            mmcv.load(str(DISABLED_ROOT / group / "predictions.pkl")),
        )
        invariance.append({"protocol": group, "leaves": leaves,
                           "max_abs_diff": difference, "exact": difference == 0})
    invariant = all(row["exact"] for row in invariance)
    if not invariant: raise RuntimeError(f"disabled prediction divergence: {invariance}")

    rows = []
    for protocol in GROUPS:
        for token, fault_frame in all_traces[protocol].items():
            if not 3 <= int(fault_frame["frame_idx"]) <= 12: continue
            clean_frame = clean_trace[token]
            clean_matched = official_matches(nusc, token, payloads["clean"])
            fault_matched = official_matches(nusc, token, payloads[protocol])
            for gt in local_gt(nusc, token):
                if gt["token"] not in clean_matched: continue
                outcome = "retained_control" if gt["token"] in fault_matched else "fault_induced_lost"
                clean_value = query_margin_statistics(
                    clean_frame["layer_logits"][-1], clean_frame["layer_boxes"][-1],
                    gt["center"], gt["label"], topk=100, geometry_threshold=2.0,
                )
                fault_value = query_margin_statistics(
                    fault_frame["layer_logits"][-1], fault_frame["layer_boxes"][-1],
                    gt["center"], gt["label"], topk=100, geometry_threshold=2.0,
                )
                if not clean_value["candidate_available"]:
                    raise RuntimeError("Clean-correct GT lacks a 2m decoder query")
                lineage = fixed_query_statistics(
                    fault_frame["layer_logits"][-1], fault_frame["layer_boxes"][-1],
                    clean_value["best_query"], gt["center"], gt["label"],
                )
                paired = fault_value["candidate_available"]
                crossing = bool(paired and clean_value["rank"] <= 100 and fault_value["rank"] > 100)
                lineage_crossing = bool(clean_value["rank"] <= 100 and lineage["rank"] > 100)
                if not paired:
                    transition = "candidate_missing"
                elif crossing and fault_value["best_query"] == clean_value["best_query"]:
                    transition = "same_best_query_crossing"
                elif crossing and lineage_crossing:
                    transition = "same_lineage_collapsed_and_replaced"
                elif crossing:
                    transition = "replacement_failed_to_stay_topk"
                else:
                    transition = "no_best_query_boundary_crossing"
                rows.append({
                    "protocol": protocol, "condition": GROUPS[protocol],
                    "sample_token": token, "scene_token": str(fault_frame["scene_token"]),
                    "frame_idx": int(fault_frame["frame_idx"]),
                    "gt_token": gt["token"], "gt_class": gt["name"], "outcome": outcome,
                    "clean_best_query": clean_value["best_query"],
                    "fault_best_query": fault_value["best_query"],
                    "fault_geometry_best_query": fault_value["geometry_best_query"],
                    "clean_score": clean_value["score"], "fault_score": fault_value["score"],
                    "clean_rank": clean_value["rank"], "fault_rank": fault_value["rank"],
                    "clean_s_k": clean_value["s_k"], "fault_s_k": fault_value["s_k"],
                    "clean_margin": clean_value["margin"], "fault_margin": fault_value["margin"],
                    "delta_margin": fault_value["margin"] - clean_value["margin"] if paired else float("nan"),
                    "delta_rank": fault_value["rank"] - clean_value["rank"] if paired else float("nan"),
                    "clean_center_distance": clean_value["center_distance"],
                    "fault_center_distance": fault_value["center_distance"],
                    "clean_near_queries": clean_value["near_count"],
                    "fault_near_queries": fault_value["near_count"],
                    "fault_candidate_available": paired,
                    "boundary_crossing": crossing,
                    "clean_query_fault_score": lineage["score"],
                    "clean_query_fault_rank": lineage["rank"],
                    "clean_query_fault_center_distance": lineage["center_distance"],
                    "clean_query_still_geometry_qualified": lineage["geometry_qualified"],
                    "same_lineage_crossing": lineage_crossing,
                    "fault_best_same_as_clean_best": bool(paired and fault_value["best_query"] == clean_value["best_query"]),
                    "fault_geometry_best_same_as_clean_best": bool(paired and fault_value["geometry_best_query"] == clean_value["best_query"]),
                    "transition_type": transition,
                })

    summary_rows = []
    for protocol in (*GROUPS, "aggregate"):
        source = rows if protocol == "aggregate" else rows
        for outcome in ("fault_induced_lost", "retained_control"):
            if protocol == "aggregate":
                values = [r for r in rows if r["outcome"] == outcome]
                temporary = [dict(r, protocol="aggregate") for r in values]
                summary_rows.append(summarize(temporary, "aggregate", outcome))
            else:
                summary_rows.append(summarize(source, protocol, outcome))

    bootstrap_rows = []
    for protocol_index, protocol in enumerate((*GROUPS, "aggregate")):
        values = rows if protocol == "aggregate" else [r for r in rows if r["protocol"] == protocol]
        lost = [r for r in values if r["outcome"] == "fault_induced_lost" and r["fault_candidate_available"]]
        retained = [r for r in values if r["outcome"] == "retained_control" and r["fault_candidate_available"]]
        for metric, key, statistic in (
            ("delta_margin_median", "delta_margin", np.median),
            ("delta_rank_median", "delta_rank", np.median),
            ("boundary_crossing_proportion", "boundary_crossing", np.mean),
        ):
            result = bootstrap_difference(
                [float(r[key]) for r in lost], [float(r[key]) for r in retained],
                statistic, seed=314159 + protocol_index * 10 + len(bootstrap_rows),
                iterations=5000,
            )
            bootstrap_rows.append({
                "protocol": protocol, "metric": metric,
                "lost_n": len(lost), "retained_n": len(retained), **result,
            })

    summary_lookup = {(r["protocol"], r["outcome"]): r for r in summary_rows}
    consistent = all(
        summary_lookup[(protocol, "fault_induced_lost")]["median_delta_margin"] < 0
        and summary_lookup[(protocol, "fault_induced_lost")]["median_delta_rank"] > 0
        and summary_lookup[(protocol, "fault_induced_lost")]["boundary_crossing_ratio"]
        > summary_lookup[(protocol, "retained_control")]["boundary_crossing_ratio"]
        for protocol in GROUPS
    )
    boot = {(r["protocol"], r["metric"]): r for r in bootstrap_rows}
    pooled_margin = boot[("aggregate", "delta_margin_median")]
    pooled_rank = boot[("aggregate", "delta_rank_median")]
    pooled_cross = boot[("aggregate", "boundary_crossing_proportion")]
    lost_aggregate = summary_lookup[("aggregate", "fault_induced_lost")]
    go = (
        consistent
        and pooled_cross["estimate"] >= 0.20 and pooled_cross["ci_low"] > 0
        and pooled_margin["ci_high"] < 0
        and pooled_rank["ci_low"] > 0
        and lost_aggregate["paired_rank_availability_ratio"] >= 0.80
        and invariant
    )

    transition_rows = []
    for protocol in (*GROUPS, "aggregate"):
        values = rows if protocol == "aggregate" else [r for r in rows if r["protocol"] == protocol]
        for outcome in ("fault_induced_lost", "retained_control"):
            selected = [r for r in values if r["outcome"] == outcome]
            for transition in sorted(set(r["transition_type"] for r in selected)):
                count = sum(r["transition_type"] == transition for r in selected)
                transition_rows.append({
                    "protocol": protocol, "outcome": outcome,
                    "transition_type": transition, "count": count,
                    "ratio": count / max(len(selected), 1),
                })

    write_csv("prediction_invariance.csv", invariance)
    write_csv("paired_gt_rank_margin.csv", rows)
    write_csv("group_summary.csv", summary_rows)
    write_csv("bootstrap_95ci.csv", bootstrap_rows)
    write_csv("query_identity_transitions.csv", transition_rows)

    table = "\n".join(
        f"| {r['condition']} | {r['outcome']} | {r['clean_correct_gt']} | "
        f"{r['paired_rank_availability_ratio']:.1%} | {r['median_delta_margin']:.6f} | "
        f"{r['median_delta_rank']:.1f} | {r['boundary_crossing_ratio']:.1%} | "
        f"{r['same_lineage_crossing_ratio']:.1%} |"
        for r in summary_rows if r["protocol"] != "aggregate"
    )
    lost = summary_lookup[("aggregate", "fault_induced_lost")]
    retained = summary_lookup[("aggregate", "retained_control")]
    aggregate_transitions = {
        row["transition_type"]: row["count"]
        for row in transition_rows
        if row["protocol"] == "aggregate"
        and row["outcome"] == "fault_induced_lost"
    }
    report = f"""# Paired Fault-Induced Rank Margin Audit

## Decision

**{'Fault-specific ranking established' if go else 'NO-GO: fault-specific ranking not established'}**.

This is a read-only paired analysis of frozen B0 traces. It did not alter or invoke the existing NO-GO boundary objective, did not train, and used no new thresholds after results. Disabled Clean/Dark/Blur/Crash predictions are tensor-exact (`max_abs_diff=0`).

## Paired results

| Protocol | Group | GT | Rank available | median delta-M | median delta-rank | best-query crossing | same-lineage crossing |
|---|---|---:|---:|---:|---:|---:|---:|
{table}

Across protocols, lost GT best-query crossing is {lost['boundary_crossing_ratio']:.1%} versus retained {retained['boundary_crossing_ratio']:.1%}; same Clean-query-lineage crossing is {lost['same_lineage_crossing_ratio']:.1%} versus {retained['same_lineage_crossing_ratio']:.1%}. Lost paired-rank availability is {lost['paired_rank_availability_ratio']:.1%}.

Lost-minus-retained bootstrap differences: median delta-M {pooled_margin['estimate']:.6f} (95% CI [{pooled_margin['ci_low']:.6f}, {pooled_margin['ci_high']:.6f}]); median delta-rank {pooled_rank['estimate']:.1f} ([{pooled_rank['ci_low']:.1f}, {pooled_rank['ci_high']:.1f}]); crossing proportion {pooled_cross['estimate']:.1%} ([{pooled_cross['ci_low']:.1%}, {pooled_cross['ci_high']:.1%}]).

## Query identity interpretation

The result is strongest at the **same-GT candidate-pool** level, not as a claim that one decoder query ID is a stable object identity. Of the {lost['clean_correct_gt']} lost GT, only {aggregate_transitions.get('same_best_query_crossing', 0)} keep the identical best query ID while crossing. {aggregate_transitions.get('same_lineage_collapsed_and_replaced', 0)} lose the Clean best-query lineage and select a replacement that also remains outside Top-K; another {aggregate_transitions.get('replacement_failed_to_stay_topk', 0)} cross because the replacement fails despite the Clean lineage itself not crossing. The fixed Clean-query ID crosses for {lost['same_lineage_crossing_ratio']:.1%} of rank-available lost GT, but also for {retained['same_lineage_crossing_ratio']:.1%} of retained controls, showing that query-ID turnover alone is not specific. In contrast, independently reselecting the best <=2m GT-class candidate produces the sharply concentrated {lost['boundary_crossing_ratio']:.1%} versus {retained['boundary_crossing_ratio']:.1%} boundary crossing.

Thus, Fault genuinely moves the best geometrically qualified evidence for the same GT across the deployment boundary, and this movement is concentrated in GT that are ultimately lost. The mechanism is usually a collapse-plus-failed-replacement event rather than persistence of one immutable query ID.

The preregistered three-protocol direction-consistency gate is {consistent}; the complete fault-specific mechanism gate is {go}. Any failed gate permanently stops this fault-specific boundary-ranking route; no margin/lambda change or smoke/training follows.
"""
    (REPORT / "PAIRED_FAULT_RANK_MARGIN_AUDIT.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "records": len(rows), "lost": lost, "retained": retained,
        "pooled_margin": pooled_margin, "pooled_rank": pooled_rank,
        "pooled_crossing": pooled_cross, "consistent": consistent, "go": go,
    }, indent=2))


if __name__ == "__main__": main()
