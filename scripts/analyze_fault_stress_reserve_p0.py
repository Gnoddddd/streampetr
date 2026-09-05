#!/usr/bin/env python3
"""Join prospective outcomes and evaluate the seven preregistered P0 gates."""

from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.fault_boundary_root_cause import auroc
from analysis.fault_stress_reserve import (
    RISK_LABELS, interval, risk_bin, spearman, two_stage_bootstrap, weighted_spearman,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/full_nuscenes/fault_stress_reserve_audit"
SOURCE = ROOT / "reports/full_nuscenes/temporal_utility_audit"
PROTOCOLS = ("blur_back", "crash_back", "dark_back")
BOOTSTRAPS = 5000


def atomic_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
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


def atomic_json(path: Path, value: object):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    default=lambda item: item.item() if isinstance(item, np.generic) else str(item)) + "\n")
    os.replace(temporary, path)


def bool_series(values):
    return values if values.dtype == bool else values.astype(str).str.lower().isin({"true", "1"})


def load_forward(validation):
    expected = set(pd.read_csv(REPORT / "frozen_cohort.csv").scene_token.astype(str))
    data, exact, completed = [], [], set()
    for marker in sorted((REPORT / "incremental/P0").glob("*.complete.json")):
        value = json.loads(marker.read_text())
        if value.get("complete") and value.get("schema_version") == 1 \
                and value.get("frozen_cohort_sha256") == validation["frozen_cohort_sha256"]:
            scene = str(value["scene_token"])
            completed.add(scene)
            data.append(pd.read_csv(REPORT / "incremental/P0" / f"{scene}.csv"))
            exact.append(pd.read_csv(REPORT / "incremental/P0" / f"{scene}.equivalence.csv"))
    coverage = {"completed_scenes": len(completed), "expected_scenes": len(expected),
                "missing_scenes": sorted(expected - completed)}
    if completed != expected:
        atomic_json(REPORT / "p0_coverage.json", {**coverage, "status": "PARTIAL_INSUFFICIENT_COVERAGE"})
        raise RuntimeError("P0 forward coverage incomplete; final decision prohibited")
    per_gt, equivalence = pd.concat(data, ignore_index=True), pd.concat(exact, ignore_index=True)
    if len(per_gt) != 1323 or per_gt.trajectory_id.duplicated().any():
        raise RuntimeError("per-GT coverage/identity mismatch")
    if not (bool_series(equivalence.output_bitwise_equal).all()
            and bool_series(equivalence.memory_bitwise_equal).all()
            and float(equivalence.output_max_abs_diff.max()) == 0.
            and float(equivalence.memory_max_abs_diff.max()) == 0.):
        raise RuntimeError("disabled/state-isolation exactness failed")
    atomic_json(REPORT / "p0_coverage.json", {**coverage, "rows": len(per_gt),
                "rows_per_protocol": per_gt.groupby("protocol").size().to_dict(), "status": "COMPLETE"})
    atomic_json(REPORT / "disabled_state_isolation_summary.json", {
        "checks": len(equivalence), "scenes": int(equivalence.scene_token.nunique()),
        "output_bitwise_equal_all": True, "memory_bitwise_equal_all": True,
        "output_max_abs_diff": 0.0, "memory_max_abs_diff": 0.0,
    })
    return per_gt, equivalence


def future_metrics(frames):
    for key in ("A_candidate", "D_candidate", "A_topk", "D_topk"):
        frames[key] = bool_series(frames[key])
    rows = []
    active = frames[frames.frame_idx >= 3]
    for trajectory_id, group in active.groupby("trajectory_id", sort=False):
        clean_available = group.A_candidate & np.isfinite(group.A_s_pos)
        fault_s = np.where(group.D_candidate & np.isfinite(group.D_s_pos), group.D_s_pos, 0.0)
        collapse = np.where(clean_available, np.maximum(group.A_s_pos - fault_s, 0.0), np.nan)
        rows.append({
            "trajectory_id": trajectory_id,
            "future_max_s_pos_collapse": float(np.nanmax(collapse)) if np.isfinite(collapse).any() else np.nan,
            "future_any_topk_crossing": bool(np.any(group.A_topk & ~group.D_topk)),
            "future_metric_frames": int(np.isfinite(collapse).sum()),
        })
    return pd.DataFrame(rows)


def bootstrap(rows, columns, statistic, seed):
    samples = two_stage_bootstrap(rows, columns, statistic, n_boot=BOOTSTRAPS, seed=seed)
    low, high, finite = interval(samples)
    return low, high, finite


def main():
    validation = json.loads((REPORT / "source_validation.json").read_text())
    if validation.get("status") != "VALIDATED_BEFORE_FORWARD":
        raise RuntimeError("source validation missing")
    per_gt, equivalence = load_forward(validation)
    outcomes = pd.read_csv(SOURCE / "trajectory_outcomes.csv")
    future = future_metrics(pd.read_csv(SOURCE / "per_gt_frame_cohort.csv"))
    per_gt = per_gt.merge(outcomes[["trajectory_id", "trajectory_outcome", "first_miss_frame"]],
                          on="trajectory_id", how="left", validate="one_to_one")
    per_gt = per_gt.merge(future, on="trajectory_id", how="left", validate="one_to_one")
    if per_gt.trajectory_outcome.isna().any():
        raise RuntimeError("future outcome join incomplete")
    per_gt["first_miss_episode_age"] = per_gt.first_miss_frame - 3
    per_gt["risk_bin_index"] = risk_bin(per_gt.M_p)
    per_gt["risk_bin"] = per_gt.risk_bin_index.map(
        {index: label for index, label in enumerate(RISK_LABELS)}).fillna("nonfinite")
    per_gt.to_csv(REPORT / "per_gt_stress_reserve.csv", index=False)
    per_scene = per_gt.groupby(["protocol", "scene_token", "trajectory_outcome"], as_index=False).agg(
        trajectories=("trajectory_id", "nunique"), median_R_K=("R_K", "median"),
        median_D_cam=("D_cam", "median"), median_J_p=("J_p", "median"),
        median_M_p=("M_p", "median"), future_crossing_rate=("future_any_topk_crossing", "mean"),
        median_future_collapse=("future_max_s_pos_collapse", "median"))
    per_scene.to_csv(REPORT / "per_scene_stress_reserve.csv", index=False)

    ci_rows, decision_rows, risk_rows = [], [], []
    all_protocol_pass = True
    for protocol_index, protocol in enumerate(PROTOCOLS):
        p = per_gt[per_gt.protocol == protocol].copy()
        primary = p[p.trajectory_outcome.isin(["future_lost", "always_retained"])].copy()
        primary["lost"] = (primary.trajectory_outcome == "future_lost").astype(int)
        lost, retained = primary[primary.lost == 1], primary[primary.lost == 0]
        coverage_pass = (len(lost) >= 20 and lost.scene_token.nunique() >= 6
                         and len(retained) >= 50 and retained.scene_token.nunique() >= 8)

        def median_diff(values):
            m, y = values[:, 0].astype(float), values[:, 1].astype(int)
            return float(np.median(m[y == 1]) - np.median(m[y == 0])) if (y == 1).any() and (y == 0).any() else np.nan
        observed_diff = median_diff(primary[["M_p", "lost"]].to_numpy())
        low, high, finite = bootstrap(primary, ["M_p", "lost"], median_diff, 949494 + protocol_index * 100)
        diff_pass = bool(low > 0)
        ci_rows.append({"protocol": protocol, "gate": "lost_minus_retained_M_median",
                        "estimate": observed_diff, "ci_low": low, "ci_high": high,
                        "finite_bootstraps": finite, "pass": diff_pass})

        bin_summary = primary.groupby("risk_bin_index").agg(
            trajectories=("trajectory_id", "size"), scenes=("scene_token", "nunique"),
            future_miss_risk=("lost", "mean"), future_lost=("lost", "sum")).reset_index()
        estimable = bin_summary[(bin_summary.trajectories >= 10) & (bin_summary.scenes >= 3)].risk_bin_index.astype(int).tolist()
        for index in range(len(RISK_LABELS)):
            found = bin_summary[bin_summary.risk_bin_index == index]
            values = found.iloc[0].to_dict() if len(found) else {}
            risk_rows.append({"protocol": protocol, "risk_bin_index": index,
                              "risk_bin": RISK_LABELS[index],
                              "trajectories": int(values.get("trajectories", 0)),
                              "scenes": int(values.get("scenes", 0)),
                              "future_lost": int(values.get("future_lost", 0)),
                              "future_miss_risk": values.get("future_miss_risk", np.nan),
                              "estimable": index in estimable})

        def bin_correlation(values):
            m, y = values[:, 0].astype(float), values[:, 1].astype(int)
            bins = risk_bin(m)
            risks, weights, indexes = [], [], []
            for index in estimable:
                selected = bins == index
                if not selected.any():
                    return np.nan
                indexes.append(index); risks.append(float(y[selected].mean())); weights.append(int(selected.sum()))
            return weighted_spearman(indexes, risks, weights)
        risks = bin_summary.set_index("risk_bin_index").reindex(estimable).future_miss_risk.to_numpy(float)
        observed_bin_corr = bin_correlation(primary[["M_p", "lost"]].to_numpy())
        adjacent_max_drop = float(np.max(np.maximum(risks[:-1] - risks[1:], 0))) if len(risks) > 1 else np.nan
        nondecreasing = bool(len(risks) >= 2 and np.all(np.diff(risks) >= 0))
        low, high, finite = bootstrap(primary, ["M_p", "lost"], bin_correlation, 949495 + protocol_index * 100)
        bin_pass = bool(nondecreasing and adjacent_max_drop <= .05 and low > 0)
        ci_rows.append({"protocol": protocol, "gate": "fixed_bin_monotonic_risk",
                        "estimate": observed_bin_corr, "ci_low": low, "ci_high": high,
                        "finite_bootstraps": finite, "estimable_bins": len(estimable),
                        "adjacent_max_drop": adjacent_max_drop, "pass": bin_pass})

        def correlation(values):
            return spearman(values[:, 0].astype(float), values[:, 1].astype(float))
        observed_collapse = correlation(primary[["M_p", "future_max_s_pos_collapse"]].to_numpy())
        low, high, finite = bootstrap(primary, ["M_p", "future_max_s_pos_collapse"], correlation,
                                      949496 + protocol_index * 100)
        collapse_pass = bool(low > 0)
        ci_rows.append({"protocol": protocol, "gate": "M_vs_future_max_s_pos_collapse_spearman",
                        "estimate": observed_collapse, "ci_low": low, "ci_high": high,
                        "finite_bootstraps": finite, "pass": collapse_pass})

        def crossing_auroc(values):
            return auroc(values[:, 0].astype(float), values[:, 1].astype(int))
        observed_crossing = crossing_auroc(primary[["M_p", "future_any_topk_crossing"]].to_numpy())
        low, high, finite = bootstrap(primary, ["M_p", "future_any_topk_crossing"], crossing_auroc,
                                      949497 + protocol_index * 100)
        crossing_pass = bool(low > .5)
        ci_rows.append({"protocol": protocol, "gate": "M_predicts_future_topk_crossing_AUROC",
                        "estimate": observed_crossing, "ci_low": low, "ci_high": high,
                        "finite_bootstraps": finite, "pass": crossing_pass})

        observed_age = correlation(lost[["M_p", "first_miss_episode_age"]].to_numpy())
        low, high, finite = bootstrap(lost, ["M_p", "first_miss_episode_age"], correlation,
                                      949498 + protocol_index * 100)
        age_pass = bool(high < 0)
        ci_rows.append({"protocol": protocol, "gate": "M_vs_first_miss_age_spearman",
                        "estimate": observed_age, "ci_low": low, "ci_high": high,
                        "finite_bootstraps": finite, "pass": age_pass})

        primary["protected"] = (primary.M_p <= -.2).astype(int)
        def protection_diff(values):
            protected, y = values[:, 0].astype(int), values[:, 1].astype(int)
            return (float(y[protected == 1].mean() - y[protected == 0].mean())
                    if (protected == 1).any() and (protected == 0).any() else np.nan)
        observed_protection = protection_diff(primary[["protected", "lost"]].to_numpy())
        low, high, finite = bootstrap(primary, ["protected", "lost"], protection_diff,
                                      949499 + protocol_index * 100)
        protection_pass = bool(high < 0)
        ci_rows.append({"protocol": protocol, "gate": "protected_M_le_minus_0p2_risk_difference",
                        "estimate": observed_protection, "ci_low": low, "ci_high": high,
                        "finite_bootstraps": finite, "pass": protection_pass})

        gates = {"coverage": coverage_pass, "lost_M": diff_pass, "risk_bins": bin_pass,
                 "future_collapse": collapse_pass, "topk_crossing": crossing_pass,
                 "earlier_miss": age_pass, "protection": protection_pass}
        protocol_pass = all(gates.values())
        all_protocol_pass &= protocol_pass
        decision_rows.append({"protocol": protocol, "future_lost": len(lost),
                              "future_lost_scenes": lost.scene_token.nunique(),
                              "always_retained": len(retained),
                              "always_retained_scenes": retained.scene_token.nunique(),
                              **{f"gate_{key}": value for key, value in gates.items()},
                              "P0_protocol_pass": protocol_pass})

    atomic_csv(REPORT / "p0_cluster_bootstrap_ci.csv", ci_rows)
    atomic_csv(REPORT / "risk_bins.csv", risk_rows)
    atomic_csv(REPORT / "p0_gate_summary.csv", decision_rows)
    decision = ("P0_PASS_P1_ELIGIBLE" if all_protocol_pass
                else "NO_GO_STRESS_RESERVE_SUSCEPTIBILITY")
    failed = [{"protocol": row["protocol"], "gates": [key[len("gate_"):]
               for key, value in row.items() if key.startswith("gate_") and not value]}
              for row in decision_rows if not row["P0_protocol_pass"]]
    atomic_json(REPORT / "decision.json", {"decision": decision, "P0_all_protocol_pass": all_protocol_pass,
                "failed_protocols": failed, "P1_status": ("ELIGIBLE" if all_protocol_pass else "LOCKED_P0_FAILED"),
                "P2_status": ("LOCKED_PENDING_P1" if all_protocol_pass else "LOCKED_P0_FAILED"),
                "bootstraps": BOOTSTRAPS, "bootstrap_unit": "scene_then_instance_trajectory"})
    if not all_protocol_pass:
        atomic_json(REPORT / "p1_status.json", {"status": "NOT_RUN_LOCKED_P0_FAILED"})
        atomic_json(REPORT / "p2_status.json", {"status": "NOT_RUN_LOCKED_P0_FAILED"})
        atomic_csv(REPORT / "camera_transfer_results.csv", [{
            "status": "NOT_RUN_LOCKED_P0_FAILED", "camera": "CAM_FRONT",
            "protocol": "all", "reason": "preregistered P0 main gate failed",
        }], ["status", "camera", "protocol", "reason"])
        atomic_csv(REPORT / "severity_ladder_results.csv", [{
            "status": "NOT_RUN_LOCKED_P0_FAILED", "camera": "CAM_BACK",
            "protocol": "blur_back,dark_back", "severity": "0.3,0.6,0.9",
            "reason": "preregistered P0 main gate failed",
        }], ["status", "camera", "protocol", "severity", "reason"])
    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    progress["status"] = "FINAL_" + decision
    progress["stages"]["P0_analysis"] = "COMPLETE"
    progress["stages"]["P1"] = ("ELIGIBLE" if all_protocol_pass else "LOCKED_P0_FAILED")
    progress["stages"]["P2"] = ("LOCKED_PENDING_P1" if all_protocol_pass else "LOCKED_P0_FAILED")
    atomic_json(REPORT / "progress_manifest.json", progress)
    if not all_protocol_pass:
        (REPORT / "PARTIAL_STATUS.md").unlink(missing_ok=True)
    print(json.dumps({"decision": decision, "failed": failed}, indent=2))


if __name__ == "__main__":
    main()
