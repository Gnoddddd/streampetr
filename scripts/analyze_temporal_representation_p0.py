#!/usr/bin/env python3
"""Scene/trajectory cluster analysis and preregistered P0 tap gates."""

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
TAPS = (
    "temporal_alignment_query_state",
    "decoder_layer5_temporal_self_attn_output",
    "final_decoder_pre_cls_query",
)
PAIRS = ("AC", "BD")
METRICS = ("cosine_distance", "normalized_l2")
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
    values = np.asarray(values, dtype=float)
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


def scene_values(frame: pd.DataFrame, metric: str) -> pd.Series:
    chosen = frame[["scene_token", "instance_token", metric]].copy()
    chosen[metric] = pd.to_numeric(chosen[metric], errors="coerce")
    chosen = chosen[np.isfinite(chosen[metric])]
    if chosen.empty:
        return pd.Series(dtype=float)
    trajectories = chosen.groupby(
        ["scene_token", "instance_token"], observed=True, sort=False
    )[metric].median()
    return trajectories.groupby(level="scene_token", sort=False).mean()


def interval(frame: pd.DataFrame, metric: str, seed: int) -> dict:
    scenes = scene_values(frame, metric)
    estimate, low, high = bootstrap(scenes.to_numpy(), seed)
    return {
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "gt_n": len(frame),
        "scene_n": len(scenes),
        "bootstrap_n": BOOTSTRAPS,
        "seed": seed,
    }


def contrast(frame: pd.DataFrame, metric: str, seed: int) -> dict:
    lost = scene_values(frame[frame.population == "lost"], metric)
    retained = scene_values(frame[frame.population == "retained"], metric)
    shared = lost.index.intersection(retained.index)
    estimate, low, high = bootstrap((lost.loc[shared] - retained.loc[shared]).to_numpy(), seed)
    return {
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "gt_n": len(frame),
        "scene_n": len(shared),
        "bootstrap_n": BOOTSTRAPS,
        "seed": seed,
    }


def load_incremental():
    drift, equivalence, metas = [], [], []
    for protocol in PROTOCOLS:
        directory = REPORT / "incremental/P0" / protocol
        if not directory.exists():
            continue
        metas.extend(json.loads(path.read_text()) for path in sorted(directory.glob("*.complete.json")))
        for path in sorted(directory.glob("*.csv")):
            if path.name.endswith(".equivalence.csv"):
                equivalence.append(pd.read_csv(path))
            else:
                drift.append(pd.read_csv(path))
    return (
        pd.concat(drift, ignore_index=True) if drift else pd.DataFrame(),
        pd.concat(equivalence, ignore_index=True) if equivalence else pd.DataFrame(),
        metas,
    )


def main() -> None:
    validation = json.loads((REPORT / "source_validation.json").read_text())
    population = pd.read_csv(REPORT / "population.csv", usecols=["protocol", "scene_token"])
    drift, equivalence, metas = load_incremental()
    coverage_rows = []
    for protocol in PROTOCOLS:
        expected_scenes = int(population[population.protocol == protocol].scene_token.nunique())
        expected_population = int((population.protocol == protocol).sum())
        protocol_metas = [
            meta for meta in metas if meta.get("protocol") == protocol and meta.get("complete")
        ]
        observed_scenes = len({meta["scene_token"] for meta in protocol_metas})
        observed_rows = len(drift[drift.protocol == protocol]) if not drift.empty else 0
        coverage_rows.append({
            "protocol": protocol,
            "expected_scenes": expected_scenes,
            "observed_scenes": observed_scenes,
            "expected_population_gt": expected_population,
            "expected_drift_rows": expected_population * len(TAPS) * len(PAIRS),
            "observed_drift_rows": observed_rows,
            "complete": (
                observed_scenes == expected_scenes
                and observed_rows == expected_population * len(TAPS) * len(PAIRS)
            ),
        })
    coverage = pd.DataFrame(coverage_rows)
    complete = bool(coverage.complete.all())
    atomic_csv(coverage, REPORT / "p0_coverage.csv")
    atomic_csv(drift, REPORT / "per_gt_drift.csv")
    atomic_csv(equivalence, REPORT / "p0_disabled_equivalence.csv")

    summary_rows = []
    ci_rows = []
    seed_index = 1000
    if not drift.empty:
        eligible = drift[(drift.matched_pair_count > 0)].copy()
        for protocol in (*PROTOCOLS, "pooled"):
            protocol_frame = eligible if protocol == "pooled" else eligible[eligible.protocol == protocol]
            for tap in TAPS:
                for pair in PAIRS:
                    tap_pair = protocol_frame[
                        (protocol_frame.tap_id == tap) & (protocol_frame.pair == pair)
                    ]
                    for population_name in ("lost", "retained"):
                        chosen = tap_pair[tap_pair.population == population_name]
                        summary_rows.append({
                            "protocol": protocol,
                            "tap_id": tap,
                            "pair": pair,
                            "population": population_name,
                            "eligible_gt_n": len(chosen),
                            "scene_n": chosen.scene_token.nunique(),
                            "trajectory_n": chosen.instance_token.nunique(),
                            "median_cosine_distance": float(chosen.cosine_distance.median()) if len(chosen) else math.nan,
                            "median_normalized_l2": float(chosen.normalized_l2.median()) if len(chosen) else math.nan,
                            "median_candidate_matches": float(chosen.matched_pair_count.median()) if len(chosen) else math.nan,
                        })
                        for metric in METRICS:
                            result = interval(chosen, metric, SEED + seed_index)
                            seed_index += 1
                            ci_rows.append({
                                "category": "population_scene_mean_of_trajectory_medians",
                                "protocol": protocol,
                                "tap_id": tap,
                                "pair": pair,
                                "population": population_name,
                                "metric": metric,
                                "cluster": "scene_bootstrap_on_trajectory_aggregates",
                                **result,
                            })
                    for metric in METRICS:
                        result = contrast(tap_pair, metric, SEED + seed_index)
                        seed_index += 1
                        ci_rows.append({
                            "category": "paired_scene_population_contrast",
                            "protocol": protocol,
                            "tap_id": tap,
                            "pair": pair,
                            "population": "lost_minus_retained",
                            "metric": metric,
                            "cluster": "scene_bootstrap_on_trajectory_aggregates",
                            **result,
                        })
    summary = pd.DataFrame(summary_rows)
    cis = pd.DataFrame(ci_rows)
    atomic_csv(summary, REPORT / "p0_summary.csv")
    atomic_csv(cis, REPORT / "p0_cluster_ci.csv")

    passive_exact = bool(
        not equivalence.empty
        and equivalence.passive_capture_output_bitwise_equal.astype(bool).all()
        and (equivalence.passive_capture_output_max_abs_diff == 0).all()
        and equivalence.passive_capture_memory_bitwise_equal.astype(bool).all()
        and (equivalence.passive_capture_memory_max_abs_diff == 0).all()
    )
    tap_gates = {}
    passing_taps = []
    if complete:
        summary_index = {
            (row.protocol, row.tap_id, row.pair, row.population): row
            for row in summary.itertuples(index=False)
        }
        ci_index = {
            (row.protocol, row.tap_id, row.pair, row.population, row.metric): row
            for row in cis.itertuples(index=False)
        }
        for tap in TAPS:
            tap_gates[tap] = {}
            tap_pass = True
            for protocol in PROTOCOLS:
                tap_gates[tap][protocol] = {}
                for pair in PAIRS:
                    lost = summary_index[(protocol, tap, pair, "lost")]
                    retained = summary_index[(protocol, tap, pair, "retained")]
                    contrast_ci = ci_index[
                        (protocol, tap, pair, "lost_minus_retained", "cosine_distance")
                    ]
                    gates = {
                        "lost_gt_n_ge_20": bool(lost.eligible_gt_n >= 20),
                        "lost_scene_n_ge_6": bool(lost.scene_n >= 6),
                        "retained_gt_n_ge_50": bool(retained.eligible_gt_n >= 50),
                        "retained_scene_n_ge_8": bool(retained.scene_n >= 8),
                        "lost_minus_retained_cosine_ci_low_gt_0": bool(contrast_ci.ci_low > 0),
                    }
                    tap_gates[tap][protocol][pair] = gates
                    tap_pass &= all(gates.values())
            if tap_pass:
                passing_taps.append(tap)
    if not complete:
        verdict = "PARTIAL_INSUFFICIENT_COVERAGE"
    elif passing_taps:
        verdict = "P0_GO_P1_REQUIRED"
    else:
        verdict = "NO_GO_DIRECT_TEMPORAL_REPRESENTATION_PRESERVATION"
    decision = {
        "verdict": verdict,
        "complete_48_protocol_scenes": complete,
        "frozen_population_rows": validation["population_rows"],
        "passive_capture_output_memory_exact_B0": passive_exact,
        "tap_gates": tap_gates,
        "p0_passing_taps": passing_taps,
        "P1": "UNLOCKED" if complete and passing_taps and passive_exact else "LOCKED",
        "P2": "LOCKED",
        "training": "PROHIBITED_IN_THIS_AUDIT",
    }
    atomic_json(decision, REPORT / "p0_decision.json")
    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    if not complete:
        progress["status"] = "PARTIAL_INSUFFICIENT_COVERAGE"
    elif passing_taps and passive_exact:
        progress["status"] = "P0_COMPLETE_P1_UNLOCKED"
        progress["stages"]["P1"] = {"status": "UNLOCKED", "taps": passing_taps}
    else:
        progress["status"] = "FINAL_DIRECT_REPRESENTATION_NO_GO"
    progress["p0_verdict"] = verdict
    atomic_json(progress, REPORT / "progress_manifest.json")

    if not complete:
        lines = [
            "# PARTIAL STATUS", "", "`PARTIAL_INSUFFICIENT_COVERAGE`", "",
            "P0 is incomplete; no tap or final Go/No-Go decision is made.", "",
            "| protocol | scenes | drift rows |", "|---|---:|---:|",
        ]
        for row in coverage.itertuples(index=False):
            lines.append(
                f"| {row.protocol} | {row.observed_scenes}/{row.expected_scenes} | "
                f"{row.observed_drift_rows}/{row.expected_drift_rows} |"
            )
        lines += ["", "Resume:", "", "```bash"]
        lines += [f"python scripts/run_temporal_representation_p0.py --protocol {p}" for p in PROTOCOLS]
        lines += ["python scripts/analyze_temporal_representation_p0.py", "```", ""]
        (REPORT / "PARTIAL_STATUS.md").write_text("\n".join(lines))
    else:
        table = []
        for protocol in PROTOCOLS:
            for tap in TAPS:
                for pair in PAIRS:
                    row = cis[
                        (cis.protocol == protocol)
                        & (cis.tap_id == tap)
                        & (cis.pair == pair)
                        & (cis.population == "lost_minus_retained")
                        & (cis.metric == "cosine_distance")
                    ].iloc[0]
                    table.append(
                        f"| {protocol} | {tap} | {pair} | {row.estimate:.6f} "
                        f"[{row.ci_low:.6f}, {row.ci_high:.6f}] |"
                    )
        report = [
            "# Temporal Target Representation Localization Audit: P0", "",
            f"**`{verdict}`**", "", "- Frozen population: 13,803 GT-protocol events.",
            "- Coverage: 48/48 protocol-scenes.",
            f"- Passive tap capture output/memory exact B0: `{passive_exact}`.",
            f"- P0 passing taps: `{', '.join(passing_taps) if passing_taps else 'none'}`.",
            "", "| protocol | tap | pair | lost-retained cosine distance [95% CI] |",
            "|---|---|---|---:|", *table, "",
        ]
        if passing_taps and passive_exact:
            report += [
                "Only the listed P0 taps are unlocked for GT-local causal patching. "
                "P2 and every training action remain locked.", "",
            ]
            status = "P0_COMPLETE_P1_UNLOCKED"
        else:
            report += [
                "No preregistered tap has stable lost-specific drift across every protocol and "
                "both history pairs. P1/P2 are not run and direct temporal representation "
                "preservation is closed.", "",
            ]
            status = "FINAL_DIRECT_REPRESENTATION_NO_GO"
        (REPORT / "REPORT_P0.md").write_text("\n".join(report))
        (REPORT / "PARTIAL_STATUS.md").write_text(
            f"# STATUS\n\n`{status}`\n\nP0 decision: `{verdict}`. See `REPORT_P0.md`.\n"
        )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

