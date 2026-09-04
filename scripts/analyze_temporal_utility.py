#!/usr/bin/env python3
"""Build the prospective cohort and run P0/P1 temporal-utility gates."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from analysis.temporal_utility import (
    CLASSES, classify_trajectory, cluster_bootstrap_delta_auroc,
    clustered_mean_ci, clustered_scalar_ci, match_trajectory_controls,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/full_nuscenes/temporal_utility_audit"
ABCD = ROOT / "reports/full_nuscenes/ctep_method_activation/per_gt_p0.csv"
F_CACHE = ROOT / "reports/full_nuscenes/bd_temporal_support_audit/per_gt_nohistory.csv"
PROTOCOLS = ("blur_back", "crash_back", "dark_back")
METRIC_KEYS = ("candidate", "qplus", "s_pos", "rank", "margin", "topk", "tp")
BOOTSTRAPS = 5000


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(
        value, indent=2, sort_keys=True,
        default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
    ) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin({"true", "1"})


def load_supplement() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    split = pd.read_csv(REPORT / "scene_split.csv")
    frames, exact, coverage = [], [], []
    for protocol in PROTOCOLS:
        directory = REPORT / "incremental/supplement" / protocol
        completed = {}
        for path in directory.glob("*.complete.json"):
            value = json.loads(path.read_text())
            if value.get("complete") and value.get("schema_version") == 1:
                completed[str(value["scene_token"])] = value
        missing = set(split.scene_token.astype(str)) - set(completed)
        for scene in sorted(completed):
            frames.append(pd.read_csv(directory / f"{scene}.csv"))
            exact.append(pd.read_csv(directory / f"{scene}.equivalence.csv"))
        coverage.append({
            "protocol": protocol, "completed_scenes": len(completed),
            "expected_scenes": 16, "rows": sum(int(x["rows"]) for x in completed.values()),
            "pre_fault_rows": sum(int(x["pre_fault_rows"]) for x in completed.values()),
            "active_rows": sum(int(x["active_rows"]) for x in completed.values()),
            "existing_F_reused": sum(int(x["existing_F_reused"]) for x in completed.values()),
            "missing_F_computed": sum(int(x["missing_F_computed"]) for x in completed.values()),
            "missing_scenes": json.dumps(sorted(missing)), "complete": not missing,
        })
    write_csv(REPORT / "supplement_coverage.csv", coverage)
    if not all(row["complete"] for row in coverage):
        raise RuntimeError("supplement incomplete; final decision prohibited")
    return pd.concat(frames, ignore_index=True), pd.concat(exact, ignore_index=True), coverage


def build_frame_table(supplement: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    abcd = pd.read_csv(ABCD)
    cache = pd.read_csv(F_CACHE, usecols=["unit_id", *[f"F_{key}" for key in METRIC_KEYS]])
    active_supp = supplement[supplement.frame_idx >= 3].copy()
    if len(active_supp) != len(abcd) or active_supp.unit_id.duplicated().any():
        raise RuntimeError("active supplement does not match all-GT A/B/C/D cache")
    physical = [
        "unit_id", "cam_back_visible", "physical_visible_view_count",
        "alternative_view_count", "max_projected_area_fraction", "supplement_role",
        *[f"F_{key}" for key in METRIC_KEYS],
    ]
    active = abcd.merge(active_supp[physical], on="unit_id", how="left", validate="one_to_one",
                        suffixes=("", "_supplement"))
    active = active.merge(cache, on="unit_id", how="left", validate="one_to_one",
                          suffixes=("_supplement", "_cache"))
    reuse_count = compute_count = 0
    for key in METRIC_KEYS:
        supplement_key, cache_key = f"F_{key}_supplement", f"F_{key}_cache"
        active[f"F_{key}"] = active[supplement_key].combine_first(active[cache_key])
    reuse_count = int((active.supplement_role == "reuse_existing_F").sum())
    compute_count = int((active.supplement_role == "missing_F_computed").sum())
    if active.F_candidate.isna().any() or reuse_count != len(cache) or reuse_count + compute_count != len(active):
        raise RuntimeError("F reuse/supplement merge incomplete")
    prelude = supplement[supplement.frame_idx <= 2].copy()
    for key in METRIC_KEYS:
        prelude[f"B_{key}"] = prelude[f"A_{key}"]
        prelude[f"D_{key}"] = prelude[f"A_{key}"]
    keep = [
        "unit_id", "protocol", "scene_token", "sample_token", "frame_idx", "gt_token",
        "instance_token", "gt_class", "distance_m", "visibility_token",
        "cam_back_visible", "physical_visible_view_count", "alternative_view_count",
        "max_projected_area_fraction", "supplement_role",
        *[f"{condition}_{key}" for condition in ("A", "B", "D", "F") for key in METRIC_KEYS],
    ]
    frames = pd.concat([prelude[keep], active[keep]], ignore_index=True, sort=False)
    for condition in ("A", "B", "D", "F"):
        for key in ("candidate", "topk", "tp"):
            frames[f"{condition}_{key}"] = bool_series(frames[f"{condition}_{key}"])
    frames["trajectory_id"] = (
        frames.protocol.astype(str) + ":" + frames.scene_token.astype(str) + ":"
        + frames.instance_token.astype(str)
    )
    finite = np.isfinite(frames[["B_s_pos", "D_s_pos", "F_s_pos"]]).all(axis=1)
    frames["U_available"] = np.where(finite, frames.B_s_pos - frames.F_s_pos, np.nan)
    frames["U_realized"] = np.where(finite, frames.D_s_pos - frames.F_s_pos, np.nan)
    frames["TU_loss"] = np.where(finite, frames.B_s_pos - frames.D_s_pos, np.nan)
    frames["TU_retention"] = np.where(
        finite & (frames.U_available > 0), frames.U_realized / frames.U_available, np.nan)
    reuse = {
        "abcd_rows_reused": len(abcd), "existing_F_rows_reused": reuse_count,
        "missing_F_rows_computed": compute_count,
        "pre_fault_rows_computed": len(prelude), "total_frame_rows": len(frames),
    }
    return frames.sort_values(["protocol", "scene_token", "instance_token", "frame_idx"]), reuse


def freeze_cohort(frames: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    anchors = frames[(frames.frame_idx == 2) & frames.A_tp].copy()
    if anchors.trajectory_id.duplicated().any():
        raise RuntimeError("duplicate frame-2 trajectory anchor")
    cohort = frames[frames.trajectory_id.isin(set(anchors.trajectory_id))].copy()
    outcome_rows = []
    for trajectory_id, rows in cohort.groupby("trajectory_id", sort=False):
        outcome, first_miss = classify_trajectory(rows)
        anchor = anchors[anchors.trajectory_id == trajectory_id].iloc[0]
        outcome_rows.append({
            "trajectory_id": trajectory_id, "protocol": anchor.protocol,
            "scene_token": anchor.scene_token, "instance_token": anchor.instance_token,
            "gt_class": anchor.gt_class, "anchor_gt_token": anchor.gt_token,
            "anchor_distance_m": anchor.distance_m,
            "anchor_visibility_token": anchor.visibility_token,
            "anchor_alternative_view_count": anchor.alternative_view_count,
            "trajectory_outcome": outcome, "first_miss_frame": first_miss,
            "observed_frames": int(len(rows)),
            "fault_episode_frames": int((rows.frame_idx >= 3).sum()),
        })
    outcomes = pd.DataFrame(outcome_rows)
    cohort = cohort.merge(
        outcomes[["trajectory_id", "trajectory_outcome", "first_miss_frame"]],
        on="trajectory_id", how="left", validate="many_to_one")
    anchor_manifest = anchors[[
        "trajectory_id", "protocol", "scene_token", "instance_token", "gt_class",
        "gt_token", "distance_m", "visibility_token", "alternative_view_count", "A_tp",
    ]].copy()
    return cohort, outcomes, anchor_manifest


def aligned_table(cohort: pd.DataFrame, outcomes: pd.DataFrame,
                  anchors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    controls = match_trajectory_controls(anchors, outcomes)
    indexed = cohort.set_index(["trajectory_id", "frame_idx"], drop=False)
    rows = []
    metrics = ("TU_loss", "TU_retention", "D_s_pos", "D_margin", "D_topk", "D_tp")
    for match_id, match in controls.reset_index(drop=True).iterrows():
        for offset in (-3, -2, -1, 0):
            frame = int(match.pseudo_miss_frame + offset)
            lost_key = (match.lost_trajectory_id, frame)
            control_key = (match.retained_trajectory_id, frame)
            available = lost_key in indexed.index and control_key in indexed.index
            row = {**match.to_dict(), "match_id": int(match_id), "offset": offset,
                   "absolute_frame": frame, "pair_available": available}
            if available:
                lost = indexed.loc[lost_key]
                control = indexed.loc[control_key]
                if isinstance(lost, pd.DataFrame) or isinstance(control, pd.DataFrame):
                    raise RuntimeError("non-unique trajectory/frame row")
                row.update({"retained_scene_token": control.scene_token,
                            "retained_instance_token": control.instance_token})
                for metric in metrics:
                    left, right = lost[metric], control[metric]
                    row[f"lost_{metric}"] = left
                    row[f"control_{metric}"] = right
                    if metric not in {"D_topk", "D_tp"}:
                        row[f"difference_{metric}"] = (
                            float(left) - float(right)
                            if np.isfinite(left) and np.isfinite(right) else np.nan)
            rows.append(row)
    aligned = pd.DataFrame(rows)
    complete = aligned.groupby("match_id").pair_available.all()
    aligned["complete_pair"] = aligned.match_id.map(complete).astype(bool)
    return aligned, controls


def p0_analysis(cohort: pd.DataFrame, aligned: pd.DataFrame,
                outcomes: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    ci_rows, decision_rows = [], []
    timeline_rows = []
    metrics = ("TU_loss", "TU_retention", "D_s_pos", "D_margin", "D_topk", "D_tp")
    complete = aligned[aligned.complete_pair & aligned.pair_available].copy()
    for protocol_index, protocol in enumerate(PROTOCOLS):
        selected = complete[complete.protocol == protocol]
        for offset_index, offset in enumerate((-3, -2, -1, 0)):
            at_time = selected[selected.offset == offset]
            for metric_index, metric in enumerate(metrics):
                for group_index, group in enumerate(("lost", "control")):
                    value = f"{group}_{metric}"
                    work = pd.DataFrame({
                        "scene_token": (at_time.lost_scene_token if group == "lost"
                                        else at_time.retained_scene_token),
                        "instance_token": (at_time.lost_instance_token if group == "lost"
                                           else at_time.retained_instance_token),
                        "value": at_time[value].astype(float),
                    })
                    statistic = clustered_mean_ci if metric in {"D_topk", "D_tp"} else clustered_scalar_ci
                    result = statistic(
                        work, "value", n_boot=BOOTSTRAPS,
                        seed=818181 + protocol_index * 1000 + offset_index * 100
                        + metric_index * 10 + group_index)
                    timeline_rows.append({
                        "protocol": protocol, "offset": offset, "group": group,
                        "metric": metric, **result,
                    })
            diff = pd.DataFrame({
                "scene_token": at_time.lost_scene_token,
                "instance_token": at_time.lost_instance_token,
                "difference_TU_loss": at_time.difference_TU_loss,
            })
            result = clustered_scalar_ci(
                diff, "difference_TU_loss", n_boot=BOOTSTRAPS,
                seed=828282 + protocol_index * 100 + offset_index)
            ci_rows.append({"protocol": protocol, "test": "lost_minus_control_TU_loss",
                            "offset": offset, **result})
        lost_minus_one = selected[selected.offset == -1].copy()
        work = pd.DataFrame({
            "scene_token": lost_minus_one.lost_scene_token,
            "instance_token": lost_minus_one.lost_instance_token,
            "lost_TU_loss": lost_minus_one.lost_TU_loss,
        })
        own = clustered_scalar_ci(work, "lost_TU_loss", n_boot=BOOTSTRAPS,
                                  seed=838383 + protocol_index)
        ci_rows.append({"protocol": protocol, "test": "future_lost_TU_loss",
                        "offset": -1, **own})
        onset_rows = []
        protocol_lost = outcomes[(outcomes.protocol == protocol)
                                 & (outcomes.trajectory_outcome == "future_lost")]
        for outcome in protocol_lost.itertuples(index=False):
            trajectory = cohort[cohort.trajectory_id == outcome.trajectory_id]
            pre = trajectory[trajectory.frame_idx < int(outcome.first_miss_frame)]
            positive = pre[np.isfinite(pre.TU_loss) & (pre.TU_loss > 0)]
            first_positive = int(positive.frame_idx.min()) if not positive.empty else None
            onset_rows.append({
                "scene_token": outcome.scene_token, "instance_token": outcome.instance_token,
                "first_miss_frame": int(outcome.first_miss_frame),
                "first_positive_TU_frame": first_positive,
                "positive_before_miss": first_positive is not None,
                "lead_frames": (int(outcome.first_miss_frame) - first_positive
                                if first_positive is not None else np.nan),
            })
        onset = pd.DataFrame(onset_rows)
        onset_ci = clustered_mean_ci(onset, "positive_before_miss", n_boot=BOOTSTRAPS,
                                     seed=848484 + protocol_index)
        ci_rows.append({"protocol": protocol, "test": "positive_TU_before_first_miss",
                        "offset": -1, **onset_ci})
        pair_ids = selected.groupby("match_id").size()
        complete_pairs = int((pair_ids == 4).sum())
        lost_scenes = int(selected.lost_scene_token.nunique())
        lookup = {(row["test"], int(row["offset"])): row for row in ci_rows
                  if row["protocol"] == protocol}
        coverage_pass = complete_pairs >= 20 and lost_scenes >= 6
        temporal_pass = bool(
            lookup[("lost_minus_control_TU_loss", -2)]["ci_low"] > 0
            and lookup[("lost_minus_control_TU_loss", -1)]["ci_low"] > 0
            and lookup[("future_lost_TU_loss", -1)]["ci_low"] > 0
            and lookup[("positive_TU_before_first_miss", -1)]["ci_low"] > .5)
        decision_rows.append({
            "protocol": protocol, "complete_pairs": complete_pairs,
            "future_lost_scenes": lost_scenes, "coverage_pass": coverage_pass,
            "temporal_order_pass": temporal_pass,
        })
    decisions = pd.DataFrame(decision_rows)
    result = {
        "status": "P0_PASS" if (decisions.coverage_pass & decisions.temporal_order_pass).all() else "P0_FAIL",
        "all_protocols_pass": bool((decisions.coverage_pass & decisions.temporal_order_pass).all()),
        "protocols": decision_rows,
    }
    return result, pd.DataFrame(timeline_rows), pd.DataFrame(ci_rows)


def transition_table(cohort: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trajectory_id, trajectory in cohort.groupby("trajectory_id", sort=False):
        trajectory = trajectory.sort_values("frame_idx")
        by_frame = {int(row.frame_idx): row for row in trajectory.itertuples(index=False)}
        for frame in sorted(by_frame):
            if frame < 2 or frame + 1 not in by_frame:
                continue
            current, following = by_frame[frame], by_frame[frame + 1]
            base = {
                "transition_id": f"{trajectory_id}:{frame}->{frame + 1}",
                "trajectory_id": trajectory_id, "protocol": current.protocol,
                "scene_token": current.scene_token, "instance_token": current.instance_token,
                "frame_idx": frame, "next_frame_idx": frame + 1,
                "gt_class": current.gt_class, "distance_m": current.distance_m,
                "visibility_token": current.visibility_token,
                "alternative_view_count": current.alternative_view_count,
                "F_s_pos": current.F_s_pos, "F_margin": current.F_margin,
                "F_topk": bool(current.F_topk), "F_tp": bool(current.F_tp),
                "TU_loss": current.TU_loss,
                "D_s_pos": current.D_s_pos, "next_D_s_pos": following.D_s_pos,
                "D_topk": bool(current.D_topk), "next_D_topk": bool(following.D_topk),
                "D_tp": bool(current.D_tp), "next_D_tp": bool(following.D_tp),
                "trajectory_outcome": current.trajectory_outcome,
            }
            base["next_s_pos_drop"] = (
                float(current.D_s_pos - following.D_s_pos)
                if np.isfinite(current.D_s_pos) and np.isfinite(following.D_s_pos) else np.nan)
            rows.append(base)
    return pd.DataFrame(rows)


def design_matrices(train: pd.DataFrame, test: pd.DataFrame, add_tu: bool):
    continuous = ["F_s_pos", "F_margin", "distance_m", "alternative_view_count"]
    if add_tu:
        continuous.append("TU_loss")
    train_parts, test_parts = [], []
    for column in continuous:
        source = train[column].to_numpy(float)
        median = float(np.nanmedian(source)) if np.isfinite(source).any() else 0.0
        source = np.where(np.isfinite(source), source, median)
        target = test[column].to_numpy(float)
        target = np.where(np.isfinite(target), target, median)
        mean, scale = float(source.mean()), float(source.std())
        scale = scale if scale > 1e-12 else 1.0
        train_parts.append(((source - mean) / scale)[:, None])
        test_parts.append(((target - mean) / scale)[:, None])
    train_parts.append(train[["F_topk", "F_tp"]].astype(float).to_numpy())
    test_parts.append(test[["F_topk", "F_tp"]].astype(float).to_numpy())
    for value in CLASSES:
        train_parts.append((train.gt_class == value).to_numpy(float)[:, None])
        test_parts.append((test.gt_class == value).to_numpy(float)[:, None])
    for value in (1, 2, 3, 4):
        train_parts.append((train.visibility_token.astype(int) == value).to_numpy(float)[:, None])
        test_parts.append((test.visibility_token.astype(int) == value).to_numpy(float)[:, None])
    return np.concatenate(train_parts, axis=1), np.concatenate(test_parts, axis=1)


def p1_analysis(transitions: pd.DataFrame, folds: dict[str, int]):
    endpoints = {
        "s_pos_collapse": transitions[np.isfinite(transitions.next_s_pos_drop)].assign(
            outcome=lambda x: (x.next_s_pos_drop >= .10).astype(int)),
        "topk_crossing": transitions[transitions.D_topk].assign(
            outcome=lambda x: (~x.next_D_topk).astype(int)),
        "tp_to_fn_miss": transitions[transitions.D_tp].assign(
            outcome=lambda x: (~x.next_D_tp).astype(int)),
    }
    prediction_rows, summary_rows = [], []
    for protocol_index, protocol in enumerate(PROTOCOLS):
        for endpoint_index, (endpoint, all_rows) in enumerate(endpoints.items()):
            data = all_rows[(all_rows.protocol == protocol) & np.isfinite(all_rows.TU_loss)].copy()
            data["fold"] = data.scene_token.map(folds).astype(int)
            data["baseline_score"] = np.nan
            data["augmented_score"] = np.nan
            folds_valid = True
            for fold in range(4):
                train, test = data[data.fold != fold], data[data.fold == fold]
                if train.outcome.nunique() < 2 or test.outcome.nunique() < 2:
                    folds_valid = False
                    continue
                for add_tu, score_name in ((False, "baseline_score"), (True, "augmented_score")):
                    x_train, x_test = design_matrices(train, test, add_tu)
                    model = LogisticRegression(
                        penalty="l2", C=1.0, class_weight="balanced", solver="liblinear",
                        random_state=929292, max_iter=1000)
                    model.fit(x_train, train.outcome.to_numpy(int))
                    data.loc[test.index, score_name] = model.predict_proba(x_test)[:, 1]
            data["endpoint"] = endpoint
            prediction_rows.extend(data.to_dict("records"))
            complete_predictions = data.baseline_score.notna().all() and data.augmented_score.notna().all()
            coverage_eligible = bool(
                complete_predictions and folds_valid and data.outcome.sum() >= 20
                and (1 - data.outcome).sum() >= 20)
            if complete_predictions:
                result = cluster_bootstrap_delta_auroc(
                    data, n_boot=BOOTSTRAPS,
                    seed=939393 + protocol_index * 100 + endpoint_index)
            else:
                result = {
                    "baseline_auroc": np.nan, "augmented_auroc": np.nan,
                    "delta_auroc": np.nan, "delta_ci_low": np.nan,
                    "delta_ci_high": np.nan, "n": len(data),
                    "positives": int(data.outcome.sum()),
                    "negatives": int((1 - data.outcome).sum()),
                    "n_scenes": int(data.scene_token.nunique()),
                    "n_trajectories": int(data.trajectory_id.nunique()),
                    "finite_bootstraps": 0,
                }
            summary_rows.append({
                "protocol": protocol, "endpoint": endpoint,
                "all_test_folds_two_class": folds_valid,
                "complete_oof_predictions": complete_predictions,
                "coverage_eligible": coverage_eligible, **result,
                "endpoint_pass": bool(coverage_eligible and result["delta_ci_low"] > 0
                                      and result["delta_auroc"] > 0),
            })
    summary = pd.DataFrame(summary_rows)
    passed = bool(len(summary) == 9 and summary.coverage_eligible.all()
                  and summary.endpoint_pass.all() and (summary.delta_auroc > 0).all())
    return ({"status": "P1_PASS" if passed else "P1_FAIL", "all_endpoints_pass": passed,
             "endpoint_results": summary_rows},
            pd.DataFrame(prediction_rows), summary)


def main() -> None:
    supplement, exact, coverage = load_supplement()
    validation = json.loads((REPORT / "source_validation.json").read_text())
    if sha256(REPORT / "scene_split.csv") != validation["scene_split_sha256"]:
        raise RuntimeError("scene split changed")
    if len(exact) != 48 or not exact.output_bitwise_equal.astype(bool).all() \
            or not exact.memory_bitwise_equal.astype(bool).all() \
            or exact.output_max_abs_diff.astype(float).max() != 0 \
            or exact.memory_max_abs_diff.astype(float).max() != 0:
        raise RuntimeError("supplement disabled exactness failed")
    frames, reuse = build_frame_table(supplement)
    cohort, outcomes, anchors = freeze_cohort(frames)
    frames.to_csv(REPORT / "per_gt_frame_all.csv", index=False)
    cohort.to_csv(REPORT / "per_gt_frame_cohort.csv", index=False)
    outcomes.to_csv(REPORT / "trajectory_outcomes.csv", index=False)
    anchors.to_csv(REPORT / "cohort_manifest.csv", index=False)
    aligned, controls = aligned_table(cohort, outcomes, anchors)
    aligned.to_csv(REPORT / "first_miss_aligned.csv", index=False)
    controls.to_csv(REPORT / "trajectory_control_matches.csv", index=False)
    p0, timeline, p0_ci = p0_analysis(cohort, aligned, outcomes)
    timeline.to_csv(REPORT / "p0_aligned_timeline_ci.csv", index=False)
    p0_ci.to_csv(REPORT / "p0_temporal_order_ci.csv", index=False)
    atomic_json(REPORT / "p0_decision.json", p0)

    transitions = transition_table(cohort)
    transitions.to_csv(REPORT / "per_transition.csv", index=False)
    split = pd.read_csv(REPORT / "scene_split.csv")
    folds = split.set_index("scene_token").fold.astype(int).to_dict()
    p1, predictions, p1_summary = p1_analysis(transitions, folds)
    predictions.to_csv(REPORT / "per_transition_prediction.csv", index=False)
    p1_summary.to_csv(REPORT / "p1_prediction_summary.csv", index=False)
    atomic_json(REPORT / "p1_decision.json", p1)

    both = bool(p0["all_protocols_pass"] and p1["all_endpoints_pass"])
    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    progress["stages"]["supplement"] = {"status": "COMPLETE", "coverage": coverage, **reuse}
    progress["stages"]["P0"] = p0["status"]
    progress["stages"]["P1"] = p1["status"]
    if both:
        progress["status"] = "P0_P1_PASS_P2_PENDING"
        progress["stages"]["P2"] = "UNLOCKED_PENDING"
        (REPORT / "PARTIAL_STATUS.md").write_text(
            "# PARTIAL STATUS\n\n`P0_P1_PASS_P2_PENDING`\n\n"
            "P0 and P1 passed. P2 mediation is required before a final decision.\n")
    else:
        progress["status"] = "FINAL_NO_GO_TEMPORAL_UTILITY_MECHANISM"
        progress["stages"]["P2"] = "LOCKED_NOT_RUN_P0_OR_P1_FAILED"
        atomic_json(REPORT / "p2_status.json", {
            "status": "LOCKED_NOT_RUN", "P0": p0["status"], "P1": p1["status"]})
        atomic_json(REPORT / "final_decision.json", {
            "decision": "NO_GO_TEMPORAL_UTILITY_MECHANISM",
            "P0": p0["status"], "P1": p1["status"], "P2": "LOCKED_NOT_RUN",
            "reason": "Temporal precedence and independent next-frame prediction did not both pass every preregistered protocol/endpoint.",
            "training": "NOT_RUN_PROHIBITED",
        })
    atomic_json(REPORT / "progress_manifest.json", progress)
    print(json.dumps({"reuse": reuse, "trajectory_counts": outcomes.trajectory_outcome.value_counts().to_dict(),
                      "P0": p0["status"], "P1": p1["status"], "P2_required": both}, indent=2))


if __name__ == "__main__":
    main()
