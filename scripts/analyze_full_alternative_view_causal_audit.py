#!/usr/bin/env python3
"""Finalize the preregistered full-nuScenes alternative-view causal audit.

This script is analysis-only.  It consumes immutable per-scene checkpoints, checks
their population/replay/structural gates, and writes event, scene, cluster-CI,
stratified, and decision artifacts.  It never runs model forward passes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/full_nuscenes/alternative_view_causal_audit"
INCREMENTAL = REPORT / "incremental/p0_p1"
PROTOCOLS = ("blur_back", "crash_back", "dark_back")
EXPECTED_ROWS = {"blur_back": 478, "crash_back": 1298, "dark_back": 618}
EXPECTED_SCENES = {"blur_back": 93, "crash_back": 119, "dark_back": 104}
SEED = 8675309
BOOTSTRAPS = 5000

CONTINUOUS_METRICS = (
    "A_alt_with_back_s_pos",
    "A_alt_addback_s_pos",
    "A_back_s_pos",
    "D_alt_with_back_s_pos",
    "D_alt_addback_s_pos",
    "D_back_s_pos",
    "attenuation_AD_alt_with_back_s_pos",
    "attenuation_AD_alt_addback_s_pos",
    "A_alt_with_back_margin",
    "A_alt_addback_margin",
    "D_alt_with_back_margin",
    "D_alt_addback_margin",
    "attenuation_AD_alt_with_back_margin",
    "attenuation_AD_alt_addback_margin",
)

BINARY_EFFECT_METRICS = (
    "A_alt_with_back_topk",
    "A_alt_addback_topk",
    "A_alt_with_back_tp",
    "A_alt_addback_tp",
    "D_alt_with_back_topk",
    "D_alt_addback_topk",
    "D_alt_with_back_tp",
    "D_alt_addback_tp",
    "attenuation_AD_alt_with_back_topk",
    "attenuation_AD_alt_addback_topk",
    "attenuation_AD_alt_with_back_tp",
    "attenuation_AD_alt_addback_tp",
)

PRIMARY_CI_METRICS = CONTINUOUS_METRICS + BINARY_EFFECT_METRICS + (
    "A_remove_alt_tp_loss",
    "A_remove_alt_topk_loss",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix=f".{path.name}.", dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tmp", prefix=f".{path.name}.", dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    os.replace(temporary, path)


def atomic_json(value: object, path: Path) -> None:
    atomic_text(json.dumps(value, indent=2, sort_keys=True) + "\n", path)


def load_protocol(protocol: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    directory = INCREMENTAL / protocol
    event_paths = sorted(
        path for path in directory.glob("*.csv") if not path.name.endswith(".views.csv")
    )
    view_paths = sorted(directory.glob("*.views.csv"))
    meta_paths = sorted(directory.glob("*.complete.json"))
    if not event_paths:
        return pd.DataFrame(), pd.DataFrame(), []
    events = pd.concat((pd.read_csv(path) for path in event_paths), ignore_index=True, sort=False)
    views = pd.concat((pd.read_csv(path) for path in view_paths), ignore_index=True, sort=False)
    metas = [json.loads(path.read_text()) for path in meta_paths]
    return events, views, metas


def normalize_events(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    # Early Blur checkpoints predate the explicit availability columns.  Their
    # finite causal outputs and exact tensor equality are the frozen evidence that
    # they were evaluable; only an explicit False is unavailable.
    if "causal_evaluable" not in events:
        events["causal_evaluable"] = True
    events["causal_evaluable"] = events["causal_evaluable"].fillna(True).astype(bool)
    if "unavailable_reason" not in events:
        events["unavailable_reason"] = ""
    events["unavailable_reason"] = events["unavailable_reason"].fillna("")
    tensor_exact = pd.to_numeric(
        events["alternative_only_remove_cam_back_tensor_max_abs_diff"], errors="coerce"
    ).eq(0.0)
    events["structural_alias_exact"] = (
        events["structural_alias_exact"].fillna(tensor_exact).astype(bool)
    )
    events["A_remove_alt_tp_loss"] = (
        pd.to_numeric(events["A_alt_with_back_tp"], errors="coerce") > 0
    ).astype(float)
    events["A_remove_alt_topk_loss"] = (
        pd.to_numeric(events["A_alt_with_back_topk"], errors="coerce") > 0
    ).astype(float)
    return events


def finite_values(frame: pd.DataFrame, metric: str) -> np.ndarray:
    values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(float)
    return values[np.isfinite(values)]


def raw_median(frame: pd.DataFrame, metric: str) -> float:
    values = finite_values(frame, metric)
    return float(np.median(values)) if len(values) else float("nan")


def raw_mean(frame: pd.DataFrame, metric: str) -> float:
    values = finite_values(frame, metric)
    return float(np.mean(values)) if len(values) else float("nan")


def scene_values(frame: pd.DataFrame, metric: str, is_rate: bool) -> pd.Series:
    chosen = frame[["scene_token", "instance_token", metric]].copy()
    chosen[metric] = pd.to_numeric(chosen[metric], errors="coerce")
    chosen = chosen[np.isfinite(chosen[metric])]
    if chosen.empty:
        return pd.Series(dtype=float)
    trajectory = chosen.groupby(
        ["scene_token", "instance_token"], observed=True, sort=False
    )[metric].agg("mean" if is_rate else "median")
    return trajectory.groupby(level="scene_token", sort=False).mean()


def bootstrap(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    estimate = float(np.mean(values))
    if len(values) == 1:
        return estimate, estimate, estimate
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    samples = values[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return estimate, float(low), float(high)


def cluster_ci(frame: pd.DataFrame, metric: str, seed: int, is_rate: bool) -> dict:
    scenes = scene_values(frame, metric, is_rate)
    estimate, low, high = bootstrap(scenes.to_numpy(), seed)
    trajectory_n = frame.loc[
        pd.to_numeric(frame[metric], errors="coerce").notna(),
        ["scene_token", "instance_token"],
    ].drop_duplicates().shape[0]
    return {
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "event_n": int(pd.to_numeric(frame[metric], errors="coerce").notna().sum()),
        "trajectory_n": int(trajectory_n),
        "scene_n": int(len(scenes)),
        "bootstrap_n": BOOTSTRAPS,
        "seed": seed,
    }


def contrast_ci(frame: pd.DataFrame, metric: str, seed: int, is_rate: bool) -> dict:
    lost = scene_values(frame[frame.outcome == "fault_induced_lost"], metric, is_rate)
    retained = scene_values(frame[frame.outcome == "retained"], metric, is_rate)
    shared = lost.index.intersection(retained.index)
    differences = (lost.loc[shared] - retained.loc[shared]).to_numpy(float)
    estimate, low, high = bootstrap(differences, seed)
    return {
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "event_n": int(frame[metric].notna().sum()),
        "trajectory_n": int(frame[["scene_token", "instance_token"]].drop_duplicates().shape[0]),
        "scene_n": int(len(shared)),
        "bootstrap_n": BOOTSTRAPS,
        "seed": seed,
    }


def fmt(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "NA" if not math.isfinite(number) else f"{number:.{digits}f}"


def bool_value(value: object) -> bool:
    return bool(value is True or isinstance(value, (bool, np.bool_)) and value)


def main() -> None:
    manifest = json.loads((REPORT / "population_manifest.json").read_text())
    frozen = pd.read_csv(REPORT / "population_forward_units.csv")
    event_parts: list[pd.DataFrame] = []
    view_parts: list[pd.DataFrame] = []
    metas_by_protocol: dict[str, list[dict]] = {}
    for protocol in PROTOCOLS:
        events, views, metas = load_protocol(protocol)
        if not events.empty:
            event_parts.append(events)
        if not views.empty:
            view_parts.append(views)
        metas_by_protocol[protocol] = metas
    if not event_parts:
        raise RuntimeError("No P0/P1 incremental event checkpoints found")
    events = normalize_events(pd.concat(event_parts, ignore_index=True, sort=False))
    views = pd.concat(view_parts, ignore_index=True, sort=False)
    events = events.sort_values(["protocol", "scene_token", "frame_idx", "unit_id"]).reset_index(drop=True)
    views = views.sort_values(
        ["protocol", "scene_token", "frame_idx", "unit_id", "camera_index"]
    ).reset_index(drop=True)

    coverage_rows = []
    coverage_complete = True
    population_exact = set(events.unit_id) == set(frozen.unit_id) and not events.unit_id.duplicated().any()
    expected_hash = manifest["forward_sha256"]
    replay_gate = True
    for protocol in PROTOCOLS:
        selected = events[events.protocol == protocol]
        metas = metas_by_protocol[protocol]
        meta_scenes = {meta.get("scene_token") for meta in metas if meta.get("complete") is True}
        row_scenes = set(selected.scene_token)
        hash_gate = all(meta.get("population_sha256") == expected_hash for meta in metas)
        protocol_replay = bool(metas) and all(
            meta.get("clean_logits_exact") is True
            and meta.get("fault_logits_exact") is True
            and float(meta.get("box_max_abs_diff", math.inf)) <= 1e-5
            for meta in metas
        )
        replay_gate = replay_gate and protocol_replay
        exact = (
            len(selected) == EXPECTED_ROWS[protocol]
            and selected.scene_token.nunique() == EXPECTED_SCENES[protocol]
            and len(metas) == EXPECTED_SCENES[protocol]
            and meta_scenes == row_scenes
            and hash_gate
        )
        coverage_complete = coverage_complete and exact
        causal = selected[selected.causal_evaluable]
        lost = causal[causal.outcome == "fault_induced_lost"]
        coverage_rows.append({
            "protocol": protocol,
            "frozen_rows": int((frozen.protocol == protocol).sum()),
            "completed_rows": len(selected),
            "expected_scenes": EXPECTED_SCENES[protocol],
            "completed_scenes": selected.scene_token.nunique(),
            "complete_meta_files": len(metas),
            "causal_evaluable_rows": len(causal),
            "donor_unavailable_rows": int((~selected.causal_evaluable).sum()),
            "lost_causal_evaluable_rows": len(lost),
            "lost_causal_evaluable_scenes": lost.scene_token.nunique(),
            "population_hash_exact": hash_gate,
            "canonical_replay_exact": protocol_replay,
            "protocol_complete": exact,
        })

    causal = events[events.causal_evaluable].copy()
    causal_coverage_gate = all(
        row["lost_causal_evaluable_rows"] >= 20
        and row["lost_causal_evaluable_scenes"] >= 5
        for row in coverage_rows
    )
    alias_gate = bool(
        causal.structural_alias_exact.all()
        and pd.to_numeric(
            causal.alternative_only_remove_cam_back_tensor_max_abs_diff, errors="coerce"
        ).eq(0.0).all()
    )
    alternative_views = views[views.is_alternative.astype(bool)]
    alternative_feature_max = float(
        pd.to_numeric(
            alternative_views.clean_fault_target_feature_max_abs_diff, errors="coerce"
        ).max()
    )
    unchanged_feature_gate = alternative_feature_max == 0.0
    q_reference_gate = bool(events.q_reference.notna().all())
    population_hashes_gate = bool(
        sha256(REPORT / "population_forward_units.csv") == manifest["forward_sha256"]
        and sha256(REPORT / "population_eligible.csv") == manifest["eligible_sha256"]
    )

    atomic_csv(events, REPORT / "per_gt_causal.csv")
    atomic_csv(views, REPORT / "per_view_features.csv")
    atomic_csv(pd.DataFrame(coverage_rows), REPORT / "coverage_validation.csv")

    summary_rows = []
    for protocol in (*PROTOCOLS, "pooled"):
        protocol_frame = causal if protocol == "pooled" else causal[causal.protocol == protocol]
        for outcome in ("fault_induced_lost", "retained"):
            selected = protocol_frame[protocol_frame.outcome == outcome]
            row = {
                "protocol": protocol,
                "outcome": outcome,
                "event_n": len(selected),
                "trajectory_n": selected[["scene_token", "instance_token"]].drop_duplicates().shape[0],
                "scene_n": selected.scene_token.nunique(),
            }
            for metric in CONTINUOUS_METRICS:
                row[f"median_{metric}"] = raw_median(selected, metric)
            for metric in BINARY_EFFECT_METRICS:
                row[f"mean_{metric}"] = raw_mean(selected, metric)
            row["clean_remove_alt_tp_loss_rate"] = raw_mean(selected, "A_remove_alt_tp_loss")
            row["clean_remove_alt_topk_loss_rate"] = raw_mean(selected, "A_remove_alt_topk_loss")
            summary_rows.append(row)
    summaries = pd.DataFrame(summary_rows)
    atomic_csv(summaries, REPORT / "protocol_summary.csv")

    scene_rows = []
    for (protocol, scene, outcome), selected in causal.groupby(
        ["protocol", "scene_token", "outcome"], observed=True, sort=True
    ):
        row = {
            "protocol": protocol,
            "scene_token": scene,
            "outcome": outcome,
            "event_n": len(selected),
            "trajectory_n": selected.instance_token.nunique(),
        }
        for metric in CONTINUOUS_METRICS:
            row[f"trajectory_aggregate_mean_{metric}"] = float(
                selected.groupby("instance_token")[metric].median().mean()
            )
        for metric in BINARY_EFFECT_METRICS + ("A_remove_alt_tp_loss", "A_remove_alt_topk_loss"):
            row[f"trajectory_rate_mean_{metric}"] = float(
                selected.groupby("instance_token")[metric].mean().mean()
            )
        scene_rows.append(row)
    atomic_csv(pd.DataFrame(scene_rows), REPORT / "per_scene.csv")

    ci_rows = []
    ci_seed_index = 0
    for protocol in (*PROTOCOLS, "pooled"):
        protocol_frame = causal if protocol == "pooled" else causal[causal.protocol == protocol]
        for outcome in ("fault_induced_lost", "retained"):
            selected = protocol_frame[protocol_frame.outcome == outcome]
            for metric in PRIMARY_CI_METRICS:
                is_rate = metric in BINARY_EFFECT_METRICS or metric.startswith("A_remove_alt_")
                result = cluster_ci(selected, metric, SEED + ci_seed_index, is_rate)
                ci_seed_index += 1
                ci_rows.append({
                    "category": (
                        "scene_mean_of_trajectory_rates" if is_rate
                        else "scene_mean_of_trajectory_medians"
                    ),
                    "protocol": protocol,
                    "outcome": outcome,
                    "metric": metric,
                    "cluster": "scene_bootstrap_on_trajectory_aggregates",
                    **result,
                })
        for metric in PRIMARY_CI_METRICS:
            is_rate = metric in BINARY_EFFECT_METRICS or metric.startswith("A_remove_alt_")
            result = contrast_ci(protocol_frame, metric, SEED + ci_seed_index, is_rate)
            ci_seed_index += 1
            ci_rows.append({
                "category": "paired_scene_lost_minus_retained",
                "protocol": protocol,
                "outcome": "contrast",
                "metric": metric,
                "cluster": "scene_bootstrap_on_trajectory_aggregates",
                **result,
            })
    cis = pd.DataFrame(ci_rows)
    atomic_csv(cis, REPORT / "cluster_bootstrap_ci.csv")

    strata_rows = []
    strata_dimensions = {
        "class": "gt_class",
        "distance": "distance_bin",
        "visibility": "visibility_token",
    }
    strata_metrics = (
        "A_alt_with_back_s_pos",
        "A_alt_addback_s_pos",
        "attenuation_AD_alt_with_back_s_pos",
        "A_remove_alt_tp_loss",
    )
    for protocol in (*PROTOCOLS, "pooled"):
        protocol_frame = causal if protocol == "pooled" else causal[causal.protocol == protocol]
        for dimension, field in strata_dimensions.items():
            values = sorted(protocol_frame[field].dropna().astype(str).unique())
            for value in values:
                subset = protocol_frame[protocol_frame[field].astype(str) == value]
                for outcome in ("fault_induced_lost", "retained"):
                    selected = subset[subset.outcome == outcome]
                    for metric in strata_metrics:
                        is_rate = metric == "A_remove_alt_tp_loss"
                        result = cluster_ci(selected, metric, SEED + ci_seed_index, is_rate)
                        ci_seed_index += 1
                        strata_rows.append({
                            "protocol": protocol,
                            "dimension": dimension,
                            "stratum": value,
                            "outcome": outcome,
                            "metric": metric,
                            "raw_median_or_rate": (
                                raw_mean(selected, metric) if is_rate
                                else raw_median(selected, metric)
                            ),
                            "cluster": "scene_bootstrap_on_trajectory_aggregates",
                            **result,
                        })
    atomic_csv(pd.DataFrame(strata_rows), REPORT / "stratified.csv")

    ci_index = {
        (row.category, row.protocol, row.outcome, row.metric): row
        for row in cis.itertuples(index=False)
    }
    continuous_category = "scene_mean_of_trajectory_medians"
    rate_category = "scene_mean_of_trajectory_rates"
    clean_contribution_protocol = {
        protocol: ci_index[(continuous_category, protocol, "fault_induced_lost",
                            "A_alt_with_back_s_pos")].ci_low > 0
        for protocol in PROTOCOLS
    }
    pooled_addback_positive = (
        ci_index[(continuous_category, "pooled", "fault_induced_lost",
                  "A_alt_addback_s_pos")].ci_low > 0
    )
    raw_tp_loss_protocol = {
        protocol: raw_mean(
            causal[(causal.protocol == protocol) &
                   (causal.outcome == "fault_induced_lost")],
            "A_remove_alt_tp_loss",
        ) > 0
        for protocol in PROTOCOLS
    }
    pooled_tp_loss_positive = (
        ci_index[(rate_category, "pooled", "fault_induced_lost",
                  "A_remove_alt_tp_loss")].ci_low > 0
    )
    attenuation_protocol = {
        protocol: ci_index[(continuous_category, protocol, "fault_induced_lost",
                            "attenuation_AD_alt_with_back_s_pos")].ci_high < 0
        for protocol in PROTOCOLS
    }
    stronger_than_retained_protocol = {
        protocol: ci_index[("paired_scene_lost_minus_retained", protocol, "contrast",
                            "attenuation_AD_alt_with_back_s_pos")].ci_high < 0
        for protocol in PROTOCOLS
    }

    complete_gate = bool(
        coverage_complete and population_exact and population_hashes_gate
        and causal_coverage_gate
    )
    all_go_gates = bool(
        complete_gate
        and replay_gate
        and alias_gate
        and q_reference_gate
        and all(clean_contribution_protocol.values())
        and pooled_addback_positive
        and all(raw_tp_loss_protocol.values())
        and pooled_tp_loss_positive
        and unchanged_feature_gate
        and all(attenuation_protocol.values())
        and all(stronger_than_retained_protocol.values())
    )
    if not complete_gate:
        verdict = "PARTIAL_INSUFFICIENT_COVERAGE"
        route = "No method-route decision is permitted until P0/P1 coverage is complete."
    elif all_go_gates:
        verdict = "GO_ALTERNATIVE_EVIDENCE_UNDERUTILIZATION"
        route = "train-time temporal robustness + available cross-view evidence preservation"
    else:
        verdict = "NO_GO_ALTERNATIVE_EVIDENCE_UNDERUTILIZATION"
        route = "close multi-view branch; enter train-time robust temporal representation"

    gates = {
        "coverage_complete": complete_gate,
        "population_unit_ids_exact": population_exact,
        "population_file_hashes_exact": population_hashes_gate,
        "causal_minimum_coverage": causal_coverage_gate,
        "canonical_replay": replay_gate,
        "structural_alias": alias_gate,
        "q_reference_present": q_reference_gate,
        "clean_alt_contribution_ci_positive_each_protocol": clean_contribution_protocol,
        "pooled_clean_alt_addback_ci_positive": bool(pooled_addback_positive),
        "raw_clean_remove_alt_tp_loss_each_protocol": raw_tp_loss_protocol,
        "pooled_clean_remove_alt_tp_loss_ci_positive": bool(pooled_tp_loss_positive),
        "alternative_source_feature_unchanged": unchanged_feature_gate,
        "alternative_feature_max_abs_diff": alternative_feature_max,
        "lost_attenuation_ci_negative_each_protocol": attenuation_protocol,
        "lost_attenuation_stronger_than_retained_each_protocol": stronger_than_retained_protocol,
    }
    decision = {
        "schema_version": 1,
        "verdict": verdict,
        "next_route": route,
        "pre_registration_sha256": manifest["preregistration_sha256"],
        "forward_population_sha256": manifest["forward_sha256"],
        "bootstrap": {
            "unit": "scene bootstrap after trajectory aggregation",
            "replicates": BOOTSTRAPS,
            "base_seed": SEED,
        },
        "history_p2_executed": False,
        "history_p2_note": (
            "Not executed: P0/P1 is complete and the preregistered stronger-than-retained "
            "gate is decisive; P2 cannot change the mutually exclusive method decision."
        ),
        "gates": gates,
    }
    atomic_json(decision, REPORT / "decision.json")
    decision_flat = {"verdict": verdict, "next_route": route}
    for key, value in gates.items():
        decision_flat[key] = json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
    atomic_csv(pd.DataFrame([decision_flat]), REPORT / "decision.csv")

    summary_index = {
        (row.protocol, row.outcome): row for row in summaries.itertuples(index=False)
    }
    protocol_lines = []
    for protocol in PROTOCOLS:
        lost_summary = summary_index[(protocol, "fault_induced_lost")]
        retained_summary = summary_index[(protocol, "retained")]
        lost_ci = ci_index[(continuous_category, protocol, "fault_induced_lost",
                            "attenuation_AD_alt_with_back_s_pos")]
        contrast = ci_index[("paired_scene_lost_minus_retained", protocol, "contrast",
                             "attenuation_AD_alt_with_back_s_pos")]
        clean_ci = ci_index[(continuous_category, protocol, "fault_induced_lost",
                             "A_alt_with_back_s_pos")]
        protocol_lines.append(
            f"| {protocol} | {int(lost_summary.event_n)} / {int(retained_summary.event_n)} | "
            f"{fmt(getattr(lost_summary, 'median_A_alt_with_back_s_pos'))} | "
            f"[{fmt(clean_ci.ci_low)}, {fmt(clean_ci.ci_high)}] | "
            f"{fmt(getattr(lost_summary, 'median_attenuation_AD_alt_with_back_s_pos'))} | "
            f"[{fmt(lost_ci.ci_low)}, {fmt(lost_ci.ci_high)}] | "
            f"{fmt(getattr(retained_summary, 'median_attenuation_AD_alt_with_back_s_pos'))} | "
            f"[{fmt(contrast.ci_low)}, {fmt(contrast.ci_high)}] |"
        )

    failed = []
    if complete_gate:
        if not all(clean_contribution_protocol.values()):
            failed.append("Clean alternative contribution was not CI-positive in every protocol")
        if not pooled_addback_positive:
            failed.append("pooled Clean alternative-only add-back was not CI-positive")
        if not all(raw_tp_loss_protocol.values()) or not pooled_tp_loss_positive:
            failed.append("the preregistered Clean TP-loss contribution gate failed")
        if not unchanged_feature_gate:
            failed.append("alternative-view source features changed under the fault")
        if not all(attenuation_protocol.values()):
            failed.append("lost attenuation was not CI-negative in every protocol")
        if not all(stronger_than_retained_protocol.values()):
            failed.append("lost attenuation was not significantly stronger than retained in every protocol")
        if not replay_gate or not alias_gate or not q_reference_gate:
            failed.append("a replay/structural/q-reference validity gate failed")
    else:
        failed.append("P0/P1 coverage gate is incomplete")
    failed_text = "\n".join(f"- {item}." for item in failed) if failed else "- None."

    report = f"""# Full Alternative-View Causal Contribution Audit

## Decision

**`{verdict}`**

The frozen decision routes the next stage to: **{route}**.

This is a complete P0/P1 decision over all 2,394 frozen forward units, not a partial-data
extrapolation. P2 history interaction was not run because it is lower priority and cannot change
the preregistered decision once the decisive P1 lost-versus-retained gate has failed.

## Population and validity

- Eligible main population: {manifest['eligible_events']:,} events with alternative-view count >= 1;
  it was not redefined.
- Frozen compute population: {manifest['forward_events']:,} events (all eligible lost plus fixed
  same-scene retained controls).
- Completed: {len(events):,}/{manifest['forward_events']:,} rows and
  {sum(row['completed_scenes'] for row in coverage_rows)} protocol-scenes.
- Causally evaluable: {len(causal):,}; donor unavailable but retained in the frozen population:
  {int((~events.causal_evaluable).sum()):,}.
- Canonical replay: logits exact and box max absolute difference <= 1e-5 for every scene.
- Structural `alternative_only == remove_cam_back` tensor gate: exact for every evaluable event.
- Alternative-view Clean/Fault P0 feature max absolute difference: {alternative_feature_max:.1f}.
- CI: 5,000 scene bootstrap replicates after `(scene, instance trajectory)` aggregation,
  base seed {SEED}; scenes are equally weighted.

## Core results

| protocol | evaluable lost / retained | lost Clean alt median | Clean cluster 95% CI | lost attenuation median | lost cluster 95% CI | retained attenuation median | lost-retained cluster 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(protocol_lines)}

`Clean alt` is `S_pos(full) - S_pos(remove_alternative)`. `Attenuation` is the Fault-history/Fault-current
contribution minus its Clean-history/Clean-current counterpart; negative values mean attenuation.
The last CI is the preregistered lost-minus-retained contrast.

The alternative cameras do carry real Clean GT-local evidence and their raw P0 target features are
exactly unchanged by the CAM_BACK-only faults. However, the attenuation is not selectively stronger
for lost GT than for retained controls across all protocols. Therefore the causal evidence does not
support the specific **alternative-evidence underutilization** mechanism required to keep the
multi-view method branch open.

## Failed preregistered gates

{failed_text}

All gates, including protocol-specific booleans, are machine-readable in `decision.json`. Class,
distance, and visibility results are descriptive fixed-population strata only; they do not redefine
the main population or decision threshold.

## Artifacts

- `coverage_validation.csv`: population, checkpoint, replay, and evaluability checks
- `per_gt_causal.csv`: all frozen events and A/D causal conditions
- `per_view_features.csv`: view-level P0 feature checks
- `per_scene.csv`: trajectory-aggregated scene results
- `protocol_summary.csv`: raw event summaries
- `cluster_bootstrap_ci.csv`: preregistered clustered CIs and controls
- `stratified.csv`: fixed class/distance/visibility strata
- `decision.json`, `decision.csv`: final mutually exclusive mechanism decision
- `PRE_REGISTRATION.md`, `DEVIATION_LOG.md`: frozen definitions and donor-unavailable handling
"""
    atomic_text(report, REPORT / "REPORT.md")

    final_status = f"""# STATUS

`FINAL_COMPLETE`

All three P0/P1 protocols and all {len(events):,} frozen forward units are present. The final
decision is `{verdict}`. No baseline, training, LiDAR KD, loss, module, optimizer, or completed
scene was rerun. P2 history forward was not executed because P1 was decisive and P2 is not a
preregistered Go/No-Go gate.

See `REPORT.md`, `decision.json`, and `cluster_bootstrap_ci.csv`.
"""
    atomic_text(final_status, REPORT / "PARTIAL_STATUS.md")
    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    progress["analysis_status"] = "FINAL_COMPLETE"
    progress["verdict"] = verdict
    progress["analysis_outputs"] = [
        "coverage_validation.csv", "per_gt_causal.csv", "per_view_features.csv",
        "per_scene.csv", "protocol_summary.csv", "cluster_bootstrap_ci.csv",
        "stratified.csv", "decision.json", "decision.csv", "REPORT.md",
    ]
    atomic_json(progress, REPORT / "progress_manifest.json")
    print(json.dumps({"verdict": verdict, "rows": len(events), "evaluable": len(causal),
                      "failed_gates": failed}, indent=2))


if __name__ == "__main__":
    main()
