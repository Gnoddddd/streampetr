#!/usr/bin/env python3
"""Finalize preregistered representation-side CTEP routing gates."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/full_nuscenes/ctep_method_activation"
REPORT = ROOT / "reports/full_nuscenes/ctep_gradient_routing_audit"
PROTOCOLS = ("blur_back", "crash_back", "dark_back")
LEVELS = (
    "selected_query_representation",
    "final_decoder_layer_5",
    "final_decoder_temporal_self_attention",
    "all_decoder_temporal_self_attention",
    "temporal_alignment_modules",
)
BOOTSTRAPS = 5000
SEED = 424242


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


def ci(frame: pd.DataFrame, metric: str, rate: bool, seed: int) -> dict:
    scenes = scene_values(frame, metric, rate)
    estimate, low, high = bootstrap(scenes.to_numpy(), seed)
    return {
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "term_n": len(frame),
        "scene_n": len(scenes),
        "bootstrap_n": BOOTSTRAPS,
        "seed": seed,
    }


def load_incremental() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    gradients, heads, equivalence, metas = [], [], [], []
    for protocol in PROTOCOLS:
        directory = REPORT / "incremental/gradient_routing" / protocol
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.complete.json")):
            metas.append(json.loads(path.read_text()))
        for path in sorted(directory.glob("*.csv")):
            if path.name.endswith(".head_zero.csv"):
                heads.append(pd.read_csv(path))
            elif path.name.endswith(".equivalence.csv"):
                equivalence.append(pd.read_csv(path))
            else:
                gradients.append(pd.read_csv(path))
    empty = pd.DataFrame()
    return (
        pd.concat(gradients, ignore_index=True) if gradients else empty.copy(),
        pd.concat(heads, ignore_index=True) if heads else empty.copy(),
        pd.concat(equivalence, ignore_index=True) if equivalence else empty.copy(),
        metas,
    )


def source_reuse_table() -> pd.DataFrame:
    p0 = pd.read_csv(SOURCE / "p0_cluster_bootstrap_ci.csv")
    activation = pd.read_csv(SOURCE / "p1_activation_cluster_ci.csv")
    rows = []
    for protocol in PROTOCOLS:
        for metric in ("A_minus_C_s_pos", "B_minus_D_s_pos"):
            selected = p0[
                (p0.protocol == protocol)
                & (p0.population == "lost")
                & (p0.metric == metric)
                & (p0.category == "scene_mean_of_trajectory_medians")
            ].iloc[0]
            rows.append({
                "protocol": protocol,
                "source_gate": f"lost_{metric}",
                "estimate": selected.estimate,
                "ci_low": selected.ci_low,
                "ci_high": selected.ci_high,
                "passed": selected.ci_low > 0,
                "source_file": "ctep_method_activation/p0_cluster_bootstrap_ci.csv",
            })
        for metric in ("ctep_active", "L_CTEP"):
            selected = activation[
                (activation.protocol == protocol)
                & (activation.population == "history_sensitive_lost_minus_retained")
                & (activation.metric == metric)
                & (activation.category == "paired_scene_population_contrast")
            ].iloc[0]
            rows.append({
                "protocol": protocol,
                "source_gate": f"history_sensitive_minus_retained_{metric}",
                "estimate": selected.estimate,
                "ci_low": selected.ci_low,
                "ci_high": selected.ci_high,
                "passed": selected.ci_low > 0,
                "source_file": "ctep_method_activation/p1_activation_cluster_ci.csv",
            })
    return pd.DataFrame(rows)


def main() -> None:
    validation = json.loads((REPORT / "source_validation.json").read_text())
    units = pd.read_csv(REPORT / "gradient_units.csv")
    gradients, heads, equivalence, metas = load_incremental()
    source_reuse = source_reuse_table()
    atomic_csv(source_reuse, REPORT / "source_mechanism_enrichment_reuse.csv")

    expected_terms = {
        protocol: int(
            units.loc[units.protocol == protocol, "active_terms"]
            .map(lambda value: len(json.loads(value)))
            .sum()
        )
        for protocol in PROTOCOLS
    }
    expected_scenes = {
        protocol: int(units.loc[units.protocol == protocol, "scene_token"].nunique())
        for protocol in PROTOCOLS
    }
    coverage_rows = []
    for protocol in PROTOCOLS:
        protocol_meta = [meta for meta in metas if meta.get("protocol") == protocol and meta.get("complete")]
        observed_scenes = len({meta["scene_token"] for meta in protocol_meta})
        observed_terms = len(heads[heads.protocol == protocol]) if not heads.empty else 0
        observed_gradient_rows = len(gradients[gradients.protocol == protocol]) if not gradients.empty else 0
        coverage_rows.append({
            "protocol": protocol,
            "expected_scenes": expected_scenes[protocol],
            "observed_scenes": observed_scenes,
            "expected_terms": expected_terms[protocol],
            "observed_terms": observed_terms,
            "expected_gradient_rows": expected_terms[protocol] * len(LEVELS),
            "observed_gradient_rows": observed_gradient_rows,
            "complete": (
                observed_scenes == expected_scenes[protocol]
                and observed_terms == expected_terms[protocol]
                and observed_gradient_rows == expected_terms[protocol] * len(LEVELS)
            ),
        })
    coverage = pd.DataFrame(coverage_rows)
    atomic_csv(coverage, REPORT / "coverage.csv")
    complete = bool(coverage.complete.all())

    summary_rows = []
    ci_rows = []
    scene_rows = []
    seed_index = 1000
    if not gradients.empty:
        for protocol in (*PROTOCOLS, "pooled"):
            protocol_frame = gradients if protocol == "pooled" else gradients[gradients.protocol == protocol]
            for level in LEVELS:
                chosen = protocol_frame[protocol_frame.gradient_level == level]
                summary_rows.append({
                    "protocol": protocol,
                    "gradient_level": level,
                    "term_n": len(chosen),
                    "scene_n": chosen.scene_token.nunique(),
                    "nonzero_cosine_coverage": float(chosen.nonzero_cosine.mean()) if len(chosen) else math.nan,
                    "mean_term_cosine": float(chosen.gradient_cosine.mean()) if len(chosen) else math.nan,
                    "conflict_rate": float(chosen.gradient_conflict.mean()) if len(chosen) else math.nan,
                })
                for metric, rate in (("gradient_cosine", False), ("gradient_conflict", True)):
                    result = ci(chosen, metric, rate, SEED + seed_index)
                    seed_index += 1
                    ci_rows.append({
                        "category": (
                            "scene_mean_of_trajectory_rates" if rate
                            else "scene_mean_of_trajectory_medians"
                        ),
                        "protocol": protocol,
                        "gradient_level": level,
                        "metric": metric,
                        "cluster": "scene_bootstrap_on_trajectory_aggregates",
                        **result,
                    })
        for (protocol, scene, level), chosen in gradients.groupby(
            ["protocol", "scene_token", "gradient_level"], sort=True
        ):
            scene_rows.append({
                "protocol": protocol,
                "scene_token": scene,
                "gradient_level": level,
                "term_n": len(chosen),
                "trajectory_mean_median_cosine": float(
                    chosen.groupby("instance_token").gradient_cosine.median().mean()
                ),
                "trajectory_mean_conflict_rate": float(
                    chosen.groupby("instance_token").gradient_conflict.mean().mean()
                ),
                "nonzero_cosine_coverage": float(chosen.nonzero_cosine.mean()),
            })
    summary = pd.DataFrame(summary_rows)
    cis = pd.DataFrame(ci_rows)
    atomic_csv(gradients, REPORT / "per_term_gradient_routing.csv")
    atomic_csv(heads, REPORT / "classification_head_aux_zero.csv")
    atomic_csv(equivalence, REPORT / "disabled_equivalence.csv")
    atomic_csv(summary, REPORT / "gradient_summary.csv")
    atomic_csv(cis, REPORT / "gradient_cluster_ci.csv")
    atomic_csv(pd.DataFrame(scene_rows), REPORT / "gradient_per_scene.csv")

    head_zero = bool(
        not heads.empty
        and heads.classification_head_aux_all_none.astype(bool).all()
        and (heads.classification_head_aux_max_abs == 0).all()
        and (heads.classification_head_aux_grad_norm == 0).all()
    )
    mapping_exact = bool(
        not heads.empty
        and heads.frozen_classifier_logits_bitwise_equal.astype(bool).all()
        and not equivalence.empty
        and equivalence.frozen_classifier_logits_bitwise_equal.astype(bool).all()
        and (equivalence.frozen_classifier_logits_max_abs_diff == 0).all()
    )
    evidence_increase = bool(
        not heads.empty and heads.ctep_descent_increases_target_s_pos.astype(bool).all()
    )
    disabled_exact = bool(
        not equivalence.empty
        and equivalence.loss_tensor_identity.astype(bool).all()
        and equivalence.output_tensor_identity.astype(bool).all()
        and (equivalence.output_max_abs_diff == 0).all()
        and (equivalence.gradient_max_abs_diff == 0).all()
        and (equivalence.detection_loss == equivalence.disabled_loss).all()
    )
    source_gates = bool(
        validation.get("p0_train_mechanism_reproduced")
        and source_reuse.passed.astype(bool).all()
        and all(
            all(validation["history_sensitive_enrichment_each_protocol"][protocol].values())
            for protocol in PROTOCOLS
        )
    )

    gradient_gates = {}
    if complete:
        summary_index = {
            (row.protocol, row.gradient_level): row for row in summary.itertuples(index=False)
        }
        ci_index = {
            (row.protocol, row.gradient_level, row.metric): row for row in cis.itertuples(index=False)
        }
        for protocol in PROTOCOLS:
            gradient_gates[protocol] = {}
            for level in LEVELS:
                item = summary_index[(protocol, level)]
                cosine = ci_index[(protocol, level, "gradient_cosine")]
                conflict = ci_index[(protocol, level, "gradient_conflict")]
                gradient_gates[protocol][level] = {
                    "nonzero_cosine_coverage_ge_0_8": bool(item.nonzero_cosine_coverage >= 0.8),
                    "cosine_ci_low_ge_0": bool(cosine.ci_low >= 0),
                    "conflict_ci_high_lt_0_5": bool(conflict.ci_high < 0.5),
                }
    upstream_compatible = bool(
        complete
        and gradient_gates
        and all(
            all(all(criteria.values()) for criteria in levels.values())
            for levels in gradient_gates.values()
        )
    )
    routing_go = bool(
        complete and source_gates and mapping_exact and head_zero and evidence_increase
        and disabled_exact and upstream_compatible
    )
    if not complete:
        verdict = "PARTIAL_INSUFFICIENT_COVERAGE"
    elif routing_go:
        verdict = "GO_CTEP_GRADIENT_ROUTING"
    else:
        verdict = "NO_GO_SCORE_LEVEL_CTEP"
    decision = {
        "verdict": verdict,
        "complete_87_terms_12_protocol_scenes": complete,
        "source_train_mechanism_and_enrichment_preserved": source_gates,
        "frozen_classifier_logits_bitwise_exact": mapping_exact,
        "classification_head_aux_gradient_strict_zero": head_zero,
        "ctep_descent_raises_C_D_target_evidence_all_terms": evidence_increase,
        "disabled_output_loss_gradient_exact_B0": disabled_exact,
        "upstream_gradient_compatibility": gradient_gates,
        "all_key_upstream_groups_compatible": upstream_compatible,
        "routing_go": routing_go,
        "conditional_stages": {
            "single_batch_overfit": "UNLOCKED" if routing_go else "LOCKED",
            "two_iter_smoke": "LOCKED",
            "short_training": "LOCKED",
            "full_training": "LOCKED",
        },
    }
    atomic_json(decision, REPORT / "routing_decision.json")

    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    progress["status"] = (
        "FINAL_ROUTING_GO" if routing_go
        else "FINAL_SCORE_LEVEL_NO_GO" if complete
        else "PARTIAL_INSUFFICIENT_COVERAGE"
    )
    progress["routing_verdict"] = verdict
    progress["conditional_stages"] = decision["conditional_stages"]
    atomic_json(progress, REPORT / "progress_manifest.json")

    if not complete:
        status = [
            "# PARTIAL STATUS", "", "`PARTIAL_INSUFFICIENT_COVERAGE`", "",
            "No final Go/No-Go is made from partial data.", "", "| protocol | scenes | terms |", "|---|---:|---:|",
        ]
        for row in coverage.itertuples(index=False):
            status.append(
                f"| {row.protocol} | {row.observed_scenes}/{row.expected_scenes} | "
                f"{row.observed_terms}/{row.expected_terms} |"
            )
        status += ["", "Resume:", "", "```bash"]
        status += [f"python scripts/run_ctep_gradient_routing_audit.py --protocol {protocol}" for protocol in PROTOCOLS]
        status += ["python scripts/analyze_ctep_gradient_routing_audit.py", "```", ""]
        (REPORT / "PARTIAL_STATUS.md").write_text("\n".join(status))
    else:
        mechanism_lines = []
        for protocol in PROTOCOLS:
            selected = source_reuse[source_reuse.protocol == protocol].set_index("source_gate")
            ac = selected.loc["lost_A_minus_C_s_pos"]
            bd = selected.loc["lost_B_minus_D_s_pos"]
            active = selected.loc["history_sensitive_minus_retained_ctep_active"]
            magnitude = selected.loc["history_sensitive_minus_retained_L_CTEP"]
            mechanism_lines.append(
                f"| {protocol} | {ac.estimate:.4f} [{ac.ci_low:.4f}, {ac.ci_high:.4f}] | "
                f"{bd.estimate:.4f} [{bd.ci_low:.4f}, {bd.ci_high:.4f}] | "
                f"{active.estimate:.4f} [{active.ci_low:.4f}, {active.ci_high:.4f}] | "
                f"{magnitude.estimate:.4f} [{magnitude.ci_low:.4f}, {magnitude.ci_high:.4f}] |"
            )
        gradient_lines = []
        for protocol in PROTOCOLS:
            for level in LEVELS:
                cosine = cis[
                    (cis.protocol == protocol) & (cis.gradient_level == level)
                    & (cis.metric == "gradient_cosine")
                ].iloc[0]
                conflict = cis[
                    (cis.protocol == protocol) & (cis.gradient_level == level)
                    & (cis.metric == "gradient_conflict")
                ].iloc[0]
                coverage_value = summary[
                    (summary.protocol == protocol) & (summary.gradient_level == level)
                ].iloc[0].nonzero_cosine_coverage
                gradient_lines.append(
                    f"| {protocol} | {level} | {coverage_value:.3f} | "
                    f"{cosine.estimate:.4f} [{cosine.ci_low:.4f}, {cosine.ci_high:.4f}] | "
                    f"{conflict.estimate:.4f} [{conflict.ci_low:.4f}, {conflict.ci_high:.4f}] |"
                )
        lines = [
            "# Representation-side CTEP Gradient Routing Audit", "", "## Decision", "",
            f"**`{verdict}`**", "", "## Hard invariants", "",
            f"- Exact source mechanism and enrichment gates preserved: `{source_gates}`.",
            "- Coverage: 87/87 active terms in 12/12 protocol-scenes.",
            f"- Fixed classifier logits bitwise equal to canonical logits: `{mapping_exact}`.",
            f"- Classification-head CTEP gradient strictly zero: `{head_zero}`.",
            f"- CTEP descent raises selected C/D evidence for all terms: `{evidence_increase}`.",
            f"- Disabled output/loss/gradient exact B0: `{disabled_exact}`.",
            "", "## Reused train mechanism and enrichment", "",
            "| protocol | lost A-C [95% CI] | lost B-D [95% CI] | HS-retained activation [95% CI] | HS-retained loss [95% CI] |",
            "|---|---:|---:|---:|---:|", *mechanism_lines,
            "", "## Real-graph gradient routing", "",
            "| protocol | actual graph view | nonzero coverage | cosine [95% CI] | conflict rate [95% CI] |",
            "|---|---|---:|---:|---:|", *gradient_lines, "",
        ]
        if routing_go:
            lines += [
                "All preregistered routing gates pass.  The fixed single-batch overfit is unlocked; "
                "the two-iteration smoke and later training remain locked pending that result.", "",
            ]
        else:
            failed = []
            for protocol, levels in gradient_gates.items():
                for level, criteria in levels.items():
                    if not all(criteria.values()):
                        failed.append(f"`{protocol}/{level}`")
            lines += [
                "Stopping rule triggered: after removing all classification-head auxiliary gradient, "
                "the preregistered upstream compatibility gate still fails for " + ", ".join(failed) + ".",
                "No single-batch overfit, two-iteration smoke, or training was run.  Score-level CTEP "
                "repair is closed; the next method candidate must use a genuine representation-level "
                "temporal objective.", "",
            ]
        lines += [
            "## Artifacts", "",
            "- `source_validation.json`, `source_mechanism_enrichment_reuse.csv`, `coverage.csv`",
            "- `parameter_group_manifest.csv`, `classification_head_aux_zero.csv`",
            "- `per_term_gradient_routing.csv`, `gradient_per_scene.csv`, `gradient_summary.csv`, `gradient_cluster_ci.csv`",
            "- `disabled_equivalence.csv`, `routing_decision.json`, `progress_manifest.json`", "",
        ]
        (REPORT / "REPORT.md").write_text("\n".join(lines))
        (REPORT / "PARTIAL_STATUS.md").write_text(
            f"# STATUS\n\n`{progress['status']}`\n\nDecision: `{verdict}`. "
            "See `REPORT.md` and `routing_decision.json`.\n"
        )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

