#!/usr/bin/env python3
"""Scene/trajectory cluster analysis and preregistered P1 causal gates."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/full_nuscenes/temporal_representation_localization_audit"
PROTOCOLS = ("blur_back", "crash_back", "dark_back")
PAIRS = ("AC", "BD")
BOOTSTRAPS = 5000
SEED = 515151


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def atomic_json(value: object, path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def bootstrap(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan, math.nan
    estimate = float(values.mean())
    if len(values) == 1:
        return estimate, estimate, estimate
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    samples = values[indexes].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return estimate, float(low), float(high)


def scene_values(frame: pd.DataFrame, metric: str, rate: bool) -> pd.Series:
    chosen = frame[["scene_token", "instance_token", metric]].copy()
    chosen[metric] = pd.to_numeric(chosen[metric], errors="coerce")
    chosen = chosen[np.isfinite(chosen[metric])]
    if chosen.empty:
        return pd.Series(dtype=float)
    trajectories = chosen.groupby(
        ["scene_token", "instance_token"], observed=True, sort=False
    )[metric].agg("mean" if rate else "median")
    return trajectories.groupby(level="scene_token", sort=False).mean()


def interval(frame, metric, rate, seed):
    scenes = scene_values(frame, metric, rate)
    estimate, low, high = bootstrap(scenes.to_numpy(), seed)
    return {
        "estimate": estimate, "ci_low": low, "ci_high": high,
        "intervention_n": len(frame), "scene_n": len(scenes),
        "bootstrap_n": BOOTSTRAPS, "seed": seed,
    }


def contrast(left, right, metric, rate, seed):
    a = scene_values(left, metric, rate)
    b = scene_values(right, metric, rate)
    shared = a.index.intersection(b.index)
    estimate, low, high = bootstrap((a.loc[shared] - b.loc[shared]).to_numpy(), seed)
    return {
        "estimate": estimate, "ci_low": low, "ci_high": high,
        "intervention_n": len(left) + len(right), "scene_n": len(shared),
        "bootstrap_n": BOOTSTRAPS, "seed": seed,
    }


def load_incremental():
    patches, equivalence, metas = [], [], []
    for protocol in PROTOCOLS:
        directory = REPORT / "incremental/P1" / protocol
        if not directory.exists():
            continue
        metas.extend(json.loads(path.read_text()) for path in sorted(directory.glob("*.complete.json")))
        for path in sorted(directory.glob("*.csv")):
            if path.name.endswith(".equivalence.csv"):
                equivalence.append(pd.read_csv(path))
            else:
                patches.append(pd.read_csv(path))
    return (
        pd.concat(patches, ignore_index=True) if patches else pd.DataFrame(),
        pd.concat(equivalence, ignore_index=True) if equivalence else pd.DataFrame(),
        metas,
    )


def main() -> None:
    p0_decision = json.loads((REPORT / "p0_decision.json").read_text())
    if p0_decision.get("verdict") != "P0_GO_P1_REQUIRED":
        raise RuntimeError(f"P1 analysis locked by {p0_decision.get('verdict')}")
    taps = tuple(p0_decision["p0_passing_taps"])
    p0 = pd.read_csv(REPORT / "per_gt_drift.csv")
    p0 = p0[(p0.tap_id.isin(taps)) & (p0.matched_pair_count > 0)]
    patches, equivalence, metas = load_incremental()
    coverage_rows = []
    for protocol in PROTOCOLS:
        expected_scenes = int(p0[p0.protocol == protocol].scene_token.nunique())
        expected_target = len(p0[p0.protocol == protocol])
        expected_non_gt = len(p0[(p0.protocol == protocol) & (p0.population == "lost")])
        protocol_metas = [meta for meta in metas if meta.get("protocol") == protocol and meta.get("complete")]
        observed_scenes = len({meta["scene_token"] for meta in protocol_metas})
        protocol_rows = patches[patches.protocol == protocol] if not patches.empty else patches
        observed_target = int((protocol_rows.control_type == "gt_target_patch").sum()) if len(protocol_rows) else 0
        observed_non_gt = int((protocol_rows.control_type == "non_gt_patch").sum()) if len(protocol_rows) else 0
        coverage_rows.append({
            "protocol": protocol,
            "expected_scenes": expected_scenes,
            "observed_scenes": observed_scenes,
            "expected_gt_target_rows": expected_target,
            "observed_gt_target_rows": observed_target,
            "expected_non_gt_rows": expected_non_gt,
            "observed_non_gt_rows": observed_non_gt,
            "complete": (
                observed_scenes == expected_scenes
                and observed_target == expected_target
                and observed_non_gt == expected_non_gt
            ),
        })
    coverage = pd.DataFrame(coverage_rows)
    complete = bool(coverage.complete.all())
    atomic_csv(coverage, REPORT / "p1_coverage.csv")
    atomic_csv(patches, REPORT / "per_gt_causal_patch.csv")
    atomic_csv(equivalence, REPORT / "p1_disabled_equivalence.csv")

    summary_rows, ci_rows, scene_rows = [], [], []
    seed_index = 3000
    continuous = ("delta_s_pos", "history_gap_closure")
    rates = ("topk_recovery", "tp_recovery", "topk_damage", "tp_damage")
    if not patches.empty:
        for protocol in (*PROTOCOLS, "pooled"):
            protocol_frame = patches if protocol == "pooled" else patches[patches.protocol == protocol]
            for tap in taps:
                for pair in PAIRS:
                    chosen = protocol_frame[(protocol_frame.tap_id == tap) & (protocol_frame.pair == pair)]
                    groups = {
                        "lost_target": chosen[(chosen.population == "lost") & (chosen.control_type == "gt_target_patch")],
                        "lost_non_gt": chosen[(chosen.population == "lost") & (chosen.control_type == "non_gt_patch")],
                        "retained_target": chosen[(chosen.population == "retained") & (chosen.control_type == "gt_target_patch")],
                    }
                    for group_name, group in groups.items():
                        summary_rows.append({
                            "protocol": protocol, "tap_id": tap, "pair": pair,
                            "group": group_name, "intervention_n": len(group),
                            "finite_delta_n": int(np.isfinite(pd.to_numeric(group.delta_s_pos, errors="coerce")).sum()),
                            "scene_n": group.scene_token.nunique(),
                            "median_delta_s_pos": float(group.delta_s_pos.median()) if len(group) else math.nan,
                            "median_gap_closure": float(group.history_gap_closure.median()) if len(group) else math.nan,
                            "topk_recovery_rate": float(group.topk_recovery.mean()) if len(group) else math.nan,
                            "tp_recovery_rate": float(group.tp_recovery.mean()) if len(group) else math.nan,
                            "topk_damage_rate": float(group.topk_damage.mean()) if len(group) else math.nan,
                            "tp_damage_rate": float(group.tp_damage.mean()) if len(group) else math.nan,
                        })
                        for metric in (*continuous, *rates):
                            rate = metric in rates
                            result = interval(group, metric, rate, SEED + seed_index)
                            seed_index += 1
                            ci_rows.append({
                                "category": "group_scene_mean_of_trajectory_rates" if rate else "group_scene_mean_of_trajectory_medians",
                                "protocol": protocol, "tap_id": tap, "pair": pair,
                                "comparison": group_name, "metric": metric,
                                "cluster": "scene_bootstrap_on_trajectory_aggregates", **result,
                            })
                    for comparison, left_name, right_name in (
                        ("lost_target_minus_lost_non_gt", "lost_target", "lost_non_gt"),
                        ("lost_target_minus_retained_target", "lost_target", "retained_target"),
                    ):
                        for metric in continuous:
                            result = contrast(
                                groups[left_name], groups[right_name], metric, False,
                                SEED + seed_index,
                            )
                            seed_index += 1
                            ci_rows.append({
                                "category": "paired_scene_control_contrast",
                                "protocol": protocol, "tap_id": tap, "pair": pair,
                                "comparison": comparison, "metric": metric,
                                "cluster": "scene_bootstrap_on_trajectory_aggregates", **result,
                            })
        for (protocol, scene, tap, pair, population_name, control), group in patches.groupby(
            ["protocol", "scene_token", "tap_id", "pair", "population", "control_type"], sort=True
        ):
            scene_rows.append({
                "protocol": protocol, "scene_token": scene, "tap_id": tap, "pair": pair,
                "population": population_name, "control_type": control,
                "intervention_n": len(group),
                "trajectory_mean_median_delta_s_pos": float(group.groupby("instance_token").delta_s_pos.median().mean()),
                "trajectory_mean_topk_recovery_rate": float(group.groupby("instance_token").topk_recovery.mean().mean()),
                "trajectory_mean_tp_recovery_rate": float(group.groupby("instance_token").tp_recovery.mean().mean()),
            })
    summary = pd.DataFrame(summary_rows)
    cis = pd.DataFrame(ci_rows)
    atomic_csv(summary, REPORT / "p1_summary.csv")
    atomic_csv(cis, REPORT / "p1_cluster_ci.csv")
    atomic_csv(pd.DataFrame(scene_rows), REPORT / "p1_per_scene.csv")

    replay_exact = bool(
        not equivalence.empty
        and equivalence.no_patch_output_bitwise_equal.astype(bool).all()
        and (equivalence.no_patch_output_max_abs_diff == 0).all()
    )
    tap_gates, passing_taps = {}, []
    if complete:
        ci_index = {
            (row.protocol, row.tap_id, row.pair, row.comparison, row.metric): row
            for row in cis.itertuples(index=False)
        }
        for tap in taps:
            tap_pass = True
            tap_gates[tap] = {}
            for protocol in PROTOCOLS:
                tap_gates[tap][protocol] = {}
                for pair in PAIRS:
                    lost_delta = ci_index[(protocol, tap, pair, "lost_target", "delta_s_pos")]
                    lost_closure = ci_index[(protocol, tap, pair, "lost_target", "history_gap_closure")]
                    non_gt = ci_index[(protocol, tap, pair, "lost_target_minus_lost_non_gt", "delta_s_pos")]
                    retained = ci_index[(protocol, tap, pair, "lost_target_minus_retained_target", "delta_s_pos")]
                    gates = {
                        "lost_target_delta_s_pos_ci_low_gt_0": bool(lost_delta.ci_low > 0),
                        "lost_target_gap_closure_ci_low_gt_0": bool(lost_closure.ci_low > 0),
                        "lost_target_minus_non_gt_delta_ci_low_gt_0": bool(non_gt.ci_low > 0),
                        "lost_target_minus_retained_delta_ci_low_gt_0": bool(retained.ci_low > 0),
                    }
                    tap_gates[tap][protocol][pair] = gates
                    tap_pass &= all(gates.values())
            if tap_pass:
                passing_taps.append(tap)
    if not complete:
        verdict = "PARTIAL_INSUFFICIENT_COVERAGE"
    elif passing_taps and replay_exact:
        verdict = "P1_GO_P2_REQUIRED"
    else:
        verdict = "NO_GO_DIRECT_TEMPORAL_REPRESENTATION_PRESERVATION"
    decision = {
        "verdict": verdict,
        "complete_48_protocol_scenes": complete,
        "no_patch_downstream_output_exact_B0": replay_exact,
        "tap_gates": tap_gates,
        "p1_passing_taps": passing_taps,
        "P2": "UNLOCKED" if complete and passing_taps and replay_exact else "LOCKED",
        "training": "PROHIBITED_IN_THIS_AUDIT",
    }
    atomic_json(decision, REPORT / "p1_decision.json")
    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    if not complete:
        progress["status"] = "PARTIAL_INSUFFICIENT_COVERAGE"
    elif passing_taps and replay_exact:
        progress["status"] = "P1_COMPLETE_P2_UNLOCKED"
        progress["stages"]["P2"] = {"status": "UNLOCKED", "taps": passing_taps}
    else:
        progress["status"] = "FINAL_DIRECT_REPRESENTATION_NO_GO"
    progress["p1_verdict"] = verdict
    atomic_json(progress, REPORT / "progress_manifest.json")

    if not complete:
        lines = [
            "# PARTIAL STATUS", "", "`PARTIAL_INSUFFICIENT_COVERAGE`", "",
            "P1 is incomplete; no causal tap or final decision is made.", "",
            "| protocol | scenes | target rows | non-GT rows |", "|---|---:|---:|---:|",
        ]
        for row in coverage.itertuples(index=False):
            lines.append(
                f"| {row.protocol} | {row.observed_scenes}/{row.expected_scenes} | "
                f"{row.observed_gt_target_rows}/{row.expected_gt_target_rows} | "
                f"{row.observed_non_gt_rows}/{row.expected_non_gt_rows} |"
            )
        lines += ["", "Resume:", "", "```bash"]
        lines += [f"python scripts/run_temporal_representation_p1.py --protocol {p}" for p in PROTOCOLS]
        lines += ["python scripts/analyze_temporal_representation_p1.py", "```", ""]
        (REPORT / "PARTIAL_STATUS.md").write_text("\n".join(lines))
    else:
        table = []
        for protocol in PROTOCOLS:
            for tap in taps:
                for pair in PAIRS:
                    target = cis[
                        (cis.protocol == protocol) & (cis.tap_id == tap) & (cis.pair == pair)
                        & (cis.comparison == "lost_target") & (cis.metric == "delta_s_pos")
                    ].iloc[0]
                    nongt = cis[
                        (cis.protocol == protocol) & (cis.tap_id == tap) & (cis.pair == pair)
                        & (cis.comparison == "lost_target_minus_lost_non_gt")
                        & (cis.metric == "delta_s_pos")
                    ].iloc[0]
                    retained = cis[
                        (cis.protocol == protocol) & (cis.tap_id == tap) & (cis.pair == pair)
                        & (cis.comparison == "lost_target_minus_retained_target")
                        & (cis.metric == "delta_s_pos")
                    ].iloc[0]
                    table.append(
                        f"| {protocol} | {tap} | {pair} | {target.estimate:.5f} "
                        f"[{target.ci_low:.5f}, {target.ci_high:.5f}] | "
                        f"{nongt.estimate:.5f} [{nongt.ci_low:.5f}, {nongt.ci_high:.5f}] | "
                        f"{retained.estimate:.5f} [{retained.ci_low:.5f}, {retained.ci_high:.5f}] |"
                    )
        report = [
            "# Temporal Target Representation Localization Audit: P1", "",
            f"**`{verdict}`**", "", "- Coverage: 48/48 protocol-scenes.",
            f"- Empty-patch downstream output exact B0: `{replay_exact}`.",
            f"- P1 passing taps: `{', '.join(passing_taps) if passing_taps else 'none'}`.",
            "", "| protocol | tap | pair | lost target delta S_pos [95% CI] | target-nonGT [95% CI] | target-retained [95% CI] |",
            "|---|---|---|---:|---:|---:|", *table, "",
        ]
        if passing_taps and replay_exact:
            report += ["Only the listed causal taps are unlocked for P2 gradient compatibility.", ""]
            status = "P1_COMPLETE_P2_UNLOCKED"
        else:
            report += [
                "No P0 tap has stable lost-target rescue stronger than both controls across all "
                "protocols and history pairs. P2 is not run and direct temporal representation "
                "preservation is closed.", "",
            ]
            status = "FINAL_DIRECT_REPRESENTATION_NO_GO"
        (REPORT / "REPORT_P1.md").write_text("\n".join(report))
        (REPORT / "PARTIAL_STATUS.md").write_text(
            f"# STATUS\n\n`{status}`\n\nP1 decision: `{verdict}`. See `REPORT_P1.md`.\n"
        )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

