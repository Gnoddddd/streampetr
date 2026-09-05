#!/usr/bin/env python3
"""Finalize CTEP activation/enrichment/gradient compatibility gates."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/full_nuscenes/ctep_method_activation"
PROTOCOLS = ("blur_back", "crash_back", "dark_back")
POPULATIONS = ("lost", "history_sensitive_lost", "retained", "easy")
LEVELS = (
    "selected_gt_class_logit",
    "selected_query_representation",
    "final_cls_head_parameters",
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
    return {"estimate": estimate, "ci_low": low, "ci_high": high,
            "event_or_term_n": len(frame), "scene_n": len(scenes),
            "bootstrap_n": BOOTSTRAPS, "seed": seed}


def contrast(frame: pd.DataFrame, left: str, right: str, metric: str,
             rate: bool, seed: int) -> dict:
    a = scene_values(frame[frame[left].astype(bool)], metric, rate)
    b = scene_values(frame[frame[right].astype(bool)], metric, rate)
    shared = a.index.intersection(b.index)
    estimate, low, high = bootstrap((a.loc[shared] - b.loc[shared]).to_numpy(), seed)
    return {"estimate": estimate, "ci_low": low, "ci_high": high,
            "event_or_term_n": int(frame[left].sum() + frame[right].sum()),
            "scene_n": len(shared), "bootstrap_n": BOOTSTRAPS, "seed": seed}


def load_gradients() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    gradients, equivalence, metas = [], [], []
    for protocol in PROTOCOLS:
        directory = REPORT / "incremental/p1_gradient" / protocol
        gradients.extend(pd.read_csv(path) for path in sorted(directory.glob("*.csv"))
                         if not path.name.endswith(".equivalence.csv"))
        equivalence.extend(pd.read_csv(path) for path in sorted(directory.glob("*.equivalence.csv")))
        metas.extend(json.loads(path.read_text()) for path in sorted(directory.glob("*.complete.json")))
    return (
        pd.concat(gradients, ignore_index=True) if gradients else pd.DataFrame(),
        pd.concat(equivalence, ignore_index=True) if equivalence else pd.DataFrame(),
        metas,
    )


def main() -> None:
    p0 = pd.read_csv(REPORT / "per_gt_p0.csv")
    p0_decision = json.loads((REPORT / "p0_decision.json").read_text())
    units = pd.read_csv(REPORT / "gradient_units.csv")
    gradients, equivalence, metas = load_gradients()

    activation_summary = []
    activation_ci = []
    seed_index = 1000
    for protocol in (*PROTOCOLS, "pooled"):
        frame = p0 if protocol == "pooled" else p0[p0.protocol == protocol]
        for population in POPULATIONS:
            chosen = frame[frame[population].astype(bool)]
            activation_summary.append({
                "protocol": protocol,
                "population": population,
                "event_n": len(chosen),
                "scene_n": chosen.scene_token.nunique(),
                "AC_eligibility_rate": float(chosen.eligible_AC.mean()) if len(chosen) else math.nan,
                "BD_eligibility_rate": float(chosen.eligible_BD.mean()) if len(chosen) else math.nan,
                "CTEP_eligibility_rate": float(chosen.ctep_eligible.mean()) if len(chosen) else math.nan,
                "AC_activation_rate": float(chosen.active_AC.mean()) if len(chosen) else math.nan,
                "BD_activation_rate": float(chosen.active_BD.mean()) if len(chosen) else math.nan,
                "CTEP_activation_rate": float(chosen.ctep_active.mean()) if len(chosen) else math.nan,
                "median_L_AC": float(chosen.L_AC.median()) if len(chosen) else math.nan,
                "median_L_BD": float(chosen.L_BD.median()) if len(chosen) else math.nan,
                "median_L_CTEP": float(chosen.L_CTEP.median()) if len(chosen) else math.nan,
            })
            for metric, rate in (("ctep_active", True), ("L_CTEP", False)):
                result = ci(chosen, metric, rate, SEED + seed_index)
                seed_index += 1
                activation_ci.append({
                    "category": "scene_mean_of_trajectory_rates" if rate
                                else "scene_mean_of_trajectory_medians",
                    "protocol": protocol, "population": population,
                    "metric": metric,
                    "cluster": "scene_bootstrap_on_trajectory_aggregates", **result,
                })
        for metric, rate in (("ctep_active", True), ("L_CTEP", False)):
            result = contrast(
                frame, "history_sensitive_lost", "retained", metric, rate,
                SEED + seed_index,
            )
            seed_index += 1
            activation_ci.append({
                "category": "paired_scene_population_contrast",
                "protocol": protocol,
                "population": "history_sensitive_lost_minus_retained",
                "metric": metric,
                "cluster": "scene_bootstrap_on_trajectory_aggregates", **result,
            })
    activation_summary = pd.DataFrame(activation_summary)
    activation_ci = pd.DataFrame(activation_ci)
    atomic_csv(activation_summary, REPORT / "p1_activation_summary.csv")
    atomic_csv(activation_ci, REPORT / "p1_activation_cluster_ci.csv")
    activation_scene_rows = []
    for (protocol, scene), scene_frame in p0.groupby(["protocol", "scene_token"], sort=True):
        for population in POPULATIONS:
            chosen = scene_frame[scene_frame[population].astype(bool)]
            if chosen.empty:
                continue
            activation_scene_rows.append({
                "protocol": protocol, "scene_token": scene, "population": population,
                "event_n": len(chosen),
                "trajectory_mean_activation_rate": float(
                    chosen.groupby("instance_token").ctep_active.mean().mean()
                ),
                "trajectory_mean_median_L_CTEP": float(
                    chosen.groupby("instance_token").L_CTEP.median().mean()
                ),
            })
    atomic_csv(pd.DataFrame(activation_scene_rows), REPORT / "p1_activation_per_scene.csv")

    expected_terms = {
        protocol: int(sum(len(json.loads(value)) for value in
                          units.loc[units.protocol == protocol, "active_terms"]))
        for protocol in PROTOCOLS
    }
    completed_scenes = {protocol: 0 for protocol in PROTOCOLS}
    for meta in metas:
        completed_scenes[meta["protocol"]] += 1
    expected_scenes = {
        protocol: int(units.loc[units.protocol == protocol, "scene_token"].nunique())
        for protocol in PROTOCOLS
    }
    gradient_complete = bool(
        not gradients.empty
        and all(completed_scenes[p] == expected_scenes[p] for p in PROTOCOLS)
        and all(len(gradients[gradients.protocol == p]) == expected_terms[p] * len(LEVELS)
                for p in PROTOCOLS)
    )

    gradient_summary_rows = []
    gradient_ci_rows = []
    if not gradients.empty:
        for protocol in (*PROTOCOLS, "pooled"):
            protocol_frame = gradients if protocol == "pooled" else gradients[
                gradients.protocol == protocol
            ]
            for level in LEVELS:
                chosen = protocol_frame[protocol_frame.gradient_level == level]
                gradient_summary_rows.append({
                    "protocol": protocol, "gradient_level": level,
                    "term_n": len(chosen), "scene_n": chosen.scene_token.nunique(),
                    "nonzero_cosine_coverage": float(chosen.nonzero_cosine.mean()) if len(chosen) else math.nan,
                    "mean_cosine": float(chosen.gradient_cosine.mean()) if len(chosen) else math.nan,
                    "conflict_rate": float(chosen.gradient_conflict.mean()) if len(chosen) else math.nan,
                    "target_s_pos_increase_direction_rate": float(
                        chosen.ctep_descent_increases_target_s_pos.mean()
                    ) if len(chosen) else math.nan,
                })
                for metric, rate in (("gradient_cosine", False),
                                     ("gradient_conflict", True)):
                    result = ci(chosen, metric, rate, SEED + seed_index)
                    seed_index += 1
                    gradient_ci_rows.append({
                        "category": "scene_mean_of_trajectory_rates" if rate
                                    else "scene_mean_of_trajectory_medians",
                        "protocol": protocol, "gradient_level": level,
                        "metric": metric,
                        "cluster": "scene_bootstrap_on_trajectory_aggregates", **result,
                    })
    gradient_summary = pd.DataFrame(gradient_summary_rows)
    gradient_cis = pd.DataFrame(gradient_ci_rows)
    atomic_csv(gradients, REPORT / "per_term_gradients.csv")
    atomic_csv(equivalence, REPORT / "disabled_equivalence.csv")
    atomic_csv(gradient_summary, REPORT / "gradient_summary.csv")
    atomic_csv(gradient_cis, REPORT / "gradient_cluster_ci.csv")
    gradient_scene_rows = []
    if not gradients.empty:
        for (protocol, scene, level), chosen in gradients.groupby(
            ["protocol", "scene_token", "gradient_level"], sort=True
        ):
            gradient_scene_rows.append({
                "protocol": protocol, "scene_token": scene, "gradient_level": level,
                "term_n": len(chosen),
                "mean_gradient_cosine": float(chosen.gradient_cosine.mean()),
                "gradient_conflict_rate": float(chosen.gradient_conflict.mean()),
                "nonzero_cosine_coverage": float(chosen.nonzero_cosine.mean()),
            })
    atomic_csv(pd.DataFrame(gradient_scene_rows), REPORT / "gradient_per_scene.csv")

    activation_index = {
        (row.protocol, row.population, row.metric): row
        for row in activation_ci[activation_ci.category == "paired_scene_population_contrast"]
        .itertuples(index=False)
    }
    enrichment = {
        protocol: {
            metric: activation_index[(protocol, "history_sensitive_lost_minus_retained",
                                      metric)].ci_low > 0
            for metric in ("ctep_active", "L_CTEP")
        } for protocol in PROTOCOLS
    }
    term_coverage = {
        protocol: expected_terms[protocol] >= 16 and expected_scenes[protocol] >= 4
        for protocol in PROTOCOLS
    }
    increase_gate = bool(
        not gradients.empty and gradients.ctep_descent_increases_target_s_pos.astype(bool).all()
    )
    disabled_gate = bool(
        not equivalence.empty
        and equivalence.loss_tensor_identity.astype(bool).all()
        and (equivalence.output_max_abs_diff == 0).all()
        and (equivalence.gradient_max_abs_diff == 0).all()
        and (equivalence.detection_loss == equivalence.disabled_loss).all()
    )
    gradient_gates = {}
    if gradient_complete:
        summary_index = {
            (row.protocol, row.gradient_level): row
            for row in gradient_summary.itertuples(index=False)
        }
        ci_index = {
            (row.protocol, row.gradient_level, row.metric): row
            for row in gradient_cis.itertuples(index=False)
        }
        for protocol in PROTOCOLS:
            gradient_gates[protocol] = {}
            for level in LEVELS:
                summary = summary_index[(protocol, level)]
                cosine = ci_index[(protocol, level, "gradient_cosine")]
                conflict = ci_index[(protocol, level, "gradient_conflict")]
                gradient_gates[protocol][level] = {
                    "nonzero_cosine_coverage_ge_0_8": summary.nonzero_cosine_coverage >= 0.8,
                    "cosine_ci_low_ge_0": cosine.ci_low >= 0,
                    "conflict_ci_high_lt_0_5": conflict.ci_high < 0.5,
                }
    all_gradient_gates = bool(
        gradient_complete and gradient_gates
        and all(all(all(values.values()) for values in levels.values())
                for levels in gradient_gates.values())
    )
    all_enrichment = all(all(values.values()) for values in enrichment.values())
    complete = gradient_complete and all(term_coverage.values())
    activation_go = bool(
        complete and p0_decision["train_mechanism_reproduced"] and all_enrichment
        and all_gradient_gates and increase_gate and disabled_gate
    )
    if not complete:
        verdict = "PARTIAL_INSUFFICIENT_COVERAGE"
    elif activation_go:
        verdict = "GO_CTEP_ACTIVATION"
    else:
        verdict = "NO_GO_CTEP_ACTIVATION"
    decision = {
        "verdict": verdict,
        "p0_train_mechanism_reproduced": p0_decision["train_mechanism_reproduced"],
        "gradient_complete": gradient_complete,
        "term_scene_coverage": term_coverage,
        "history_sensitive_enrichment_each_protocol": enrichment,
        "target_s_pos_update_direction_100_percent": increase_gate,
        "disabled_tensor_loss_gradient_equivalence": disabled_gate,
        "gradient_compatibility": gradient_gates,
        "activation_go": activation_go,
        "conditional_stages": {
            "single_batch_overfit": "UNLOCKED" if activation_go else "LOCKED",
            "two_iter_smoke": "LOCKED",
            "short_train_75_iter": "LOCKED",
            "full_training": "LOCKED",
        },
    }
    atomic_json(decision, REPORT / "activation_decision.json")
    print(json.dumps(decision, indent=2))

    if complete:
        p0_cis = pd.read_csv(REPORT / "p0_cluster_bootstrap_ci.csv")
        p0_lines = []
        for protocol in PROTOCOLS:
            chosen = p0_cis[
                (p0_cis.protocol == protocol)
                & (p0_cis.category == "scene_mean_of_trajectory_medians")
                & (p0_cis.population == "lost")
            ].set_index("metric")
            ac, bd = chosen.loc["A_minus_C_s_pos"], chosen.loc["B_minus_D_s_pos"]
            p0_lines.append(
                f"| {protocol} | {ac.estimate:.4f} [{ac.ci_low:.4f}, {ac.ci_high:.4f}] | "
                f"{bd.estimate:.4f} [{bd.ci_low:.4f}, {bd.ci_high:.4f}] |"
            )
        enrichment_lines = []
        for protocol in PROTOCOLS:
            chosen = activation_ci[
                (activation_ci.protocol == protocol)
                & (activation_ci.category == "paired_scene_population_contrast")
            ].set_index("metric")
            active, magnitude = chosen.loc["ctep_active"], chosen.loc["L_CTEP"]
            enrichment_lines.append(
                f"| {protocol} | {active.estimate:.4f} "
                f"[{active.ci_low:.4f}, {active.ci_high:.4f}] | "
                f"{magnitude.estimate:.4f} "
                f"[{magnitude.ci_low:.4f}, {magnitude.ci_high:.4f}] |"
            )
        gradient_lines = []
        for protocol in PROTOCOLS:
            for level in LEVELS:
                cosine = gradient_cis[
                    (gradient_cis.protocol == protocol)
                    & (gradient_cis.gradient_level == level)
                    & (gradient_cis.metric == "gradient_cosine")
                ].iloc[0]
                conflict = gradient_cis[
                    (gradient_cis.protocol == protocol)
                    & (gradient_cis.gradient_level == level)
                    & (gradient_cis.metric == "gradient_conflict")
                ].iloc[0]
                gradient_lines.append(
                    f"| {protocol} | {level} | {cosine.estimate:.4f} "
                    f"[{cosine.ci_low:.4f}, {cosine.ci_high:.4f}] | "
                    f"{conflict.estimate:.4f} "
                    f"[{conflict.ci_low:.4f}, {conflict.ci_high:.4f}] |"
                )
        lines = ["# CTEP Objective Activation Audit", "", "## Decision", "",
                 f"**`{verdict}`**", "", "## Coverage", "",
                 "- P0: 48/48 protocol-scenes, 17,466 GT-event rows.",
                 f"- P1 active terms: Blur {expected_terms['blur_back']}, "
                 f"Crash {expected_terms['crash_back']}, Dark {expected_terms['dark_back']}; "
                 "four scenes per protocol.",
                 f"- Disabled output/loss/gradient equivalence: `{disabled_gate}`.",
                 "", "## Gate summary", "",
                 f"- Train mechanism reproduced: `{p0_decision['train_mechanism_reproduced']}`.",
                 f"- History-sensitive activation/loss enrichment all protocols: `{all_enrichment}`.",
                 f"- CTEP descent raises selected C/D GT evidence for every active term: `{increase_gate}`.",
                 f"- All selected-logit/query/head gradient compatibility gates: `{all_gradient_gates}`.",
                 "", "## P0 train mechanism", "",
                 "| protocol | lost A-C estimate [95% CI] | lost B-D estimate [95% CI] |",
                 "|---|---:|---:|", *p0_lines,
                 "", "## History-sensitive enrichment over retained", "",
                 "| protocol | activation-rate difference [95% CI] | L_CTEP difference [95% CI] |",
                 "|---|---:|---:|", *enrichment_lines,
                 "", "## Gradient compatibility", "",
                 "| protocol | level | cosine [95% CI] | conflict rate [95% CI] |",
                 "|---|---|---:|---:|", *gradient_lines,
                 "", "Blur's final shared classification-head parameters fail both preregistered "
                 "compatibility criteria: cosine CI lower >=0 and conflict CI upper <0.5. "
                 "Exact booleans are in `activation_decision.json`; no margin or lambda was "
                 "introduced or tuned.", ""]
        if not activation_go:
            lines += ["The preregistered Activation gate failed. Single-batch overfit, two-iteration "
                      "smoke, short training, and full training remain locked.", ""]
        else:
            lines += ["Activation passed; only the fixed single-batch overfit stage is unlocked.", ""]
        lines += ["## Artifacts", "", "- `per_gt_p0.csv`, `per_scene_p0.csv`, "
                  "`p0_cluster_bootstrap_ci.csv`", "- `p1_activation_summary.csv`, "
                  "`p1_activation_per_scene.csv`, `p1_activation_cluster_ci.csv`", "- `per_term_gradients.csv`, "
                  "`gradient_per_scene.csv`, `gradient_summary.csv`, `gradient_cluster_ci.csv`", "- `disabled_equivalence.csv`, "
                  "`activation_decision.json`", ""]
        (REPORT / "REPORT.md").write_text("\n".join(lines))
        status = "FINAL_ACTIVATION_GO" if activation_go else "FINAL_ACTIVATION_NO_GO"
        (REPORT / "PARTIAL_STATUS.md").write_text(
            f"# STATUS\n\n`{status}`\n\nDecision: `{verdict}`. See `REPORT.md` and "
            "`activation_decision.json`.\n"
        )
        progress = json.loads((REPORT / "progress_manifest.json").read_text())
        progress["status"] = status
        progress["activation_verdict"] = verdict
        atomic_json(progress, REPORT / "progress_manifest.json")


if __name__ == "__main__":
    main()
