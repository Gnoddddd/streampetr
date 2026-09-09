#!/usr/bin/env python3
"""Analyze a completed frozen CARE-3D P2-A2-R1-C0 cohort."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.care3d_p2a2_r1_confirmation import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FROZEN_ARTIFACT_SHA256,
    INSTANCE_CLUSTER_COLUMNS,
    SCENE_CLUSTER_COLUMNS,
    SOURCE_HEAD,
    confirmatory_point_metrics,
    default_discipline,
    evaluate_confirmatory_gate,
    load_frozen_arbiter,
    paired_confirmatory_bootstrap,
    require_clean_cohort,
    sha256_file,
    validate_full_population_keys,
)
from analysis.care3d_p2a_association import PROTOCOLS
from scripts.export_care3d_p2a2_r1_confirmation import (
    FROZEN_ARTIFACT,
    INCREMENTAL,
    REPORT,
    SCHEMA_VERSION,
    atomic_csv,
    atomic_json,
    marker_valid,
)


REPAIR_COLUMNS = (
    "wrong_repaired_count",
    "correct_broken_to_wrong_count",
    "correct_broken_to_unmatched_count",
    "wrong_to_unmatched_count",
    "wrong_remaining_wrong_count",
)


def load_completed_population(
    *,
    report: Path = REPORT,
    incremental: Path = INCREMENTAL,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict[str, bool]]:
    """Load only audited C0 outputs; raw probe and detector inputs stay closed."""
    manifest_path = report / "confirmatory_manifest.csv"
    audit_path = report / "heldout_lineage_audit.json"
    progress_path = report / "progress_manifest.json"
    if not all(path.is_file() for path in (manifest_path, audit_path, progress_path)):
        raise RuntimeError("confirmatory cohort/extraction metadata is incomplete")
    manifest = pd.read_csv(manifest_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    require_clean_cohort(audit)
    if set(manifest.columns) != {"scene_token", "split"}:
        raise RuntimeError("confirmatory manifest contains non-provenance columns")
    if set(manifest.split.astype(str)) != {"official_val"}:
        raise RuntimeError("analysis cohort is not official_val only")
    if manifest.scene_token.astype(str).duplicated().any():
        raise RuntimeError("analysis cohort contains duplicate scenes")
    if progress.get("status") != "CONFIRMATORY_EXTRACTION_COMPLETE_ANALYSIS_ELIGIBLE":
        raise RuntimeError("full confirmatory extraction is not complete")
    if progress.get("completed_scenes") != len(manifest):
        raise RuntimeError("confirmatory progress does not cover the full cohort")
    if progress.get("frozen_artifact_sha256") != FROZEN_ARTIFACT_SHA256:
        raise RuntimeError("confirmatory progress artifact identity changed")
    if progress.get("probe_val_read") is not False or progress.get("probe_test_read") is not False:
        raise RuntimeError("confirmatory progress indicates locked split access")

    frames = []
    discipline_contract = default_discipline()
    marker_discipline_rows = []
    for scene in manifest.scene_token.astype(str):
        marker_path = incremental / f"{scene}.complete.json"
        rows_path = incremental / f"{scene}.rows.csv"
        if not marker_path.is_file() or not rows_path.is_file():
            raise RuntimeError(f"confirmatory scene is incomplete: {scene}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        source_sha = str(marker.get("source_rows_sha256"))
        if not marker_valid(marker_path, scene, source_sha):
            raise RuntimeError(f"invalid confirmatory scene marker: {scene}")
        marker_discipline_rows.append({
            name: marker.get(name) for name in discipline_contract
        })
        if marker.get("rows_sha256") != sha256_file(rows_path):
            raise RuntimeError(f"confirmatory scene rows checksum changed: {scene}")
        frame = pd.read_csv(rows_path)
        if len(frame) != int(marker.get("rows", -1)):
            raise RuntimeError(f"confirmatory scene row count changed: {scene}")
        if len(frame):
            if set(frame.scene_token.astype(str)) != {scene}:
                raise RuntimeError(f"confirmatory scene identity changed: {scene}")
            frames.append(frame)
    if not frames:
        raise RuntimeError("confirmatory full population contains no eligible rows")
    if len(marker_discipline_rows) != len(manifest) or any(
        row != discipline_contract for row in marker_discipline_rows
    ):
        raise RuntimeError("confirmatory discipline was not verified from all markers")
    population = pd.concat(frames, ignore_index=True)
    validate_full_population_keys(population)
    if set(population.protocol.astype(str)) != set(PROTOCOLS):
        raise RuntimeError("confirmatory population lacks a fixed protocol")
    p2a0 = population[
        ["p2a0_exact", "p2a0_wrong", "p2a0_unmatched"]
    ].astype(int).sum(axis=1)
    arbiter = population[
        ["arbiter_exact", "arbiter_wrong", "arbiter_unmatched"]
    ].astype(int).sum(axis=1)
    if not (p2a0 == 1).all() or not (arbiter == 1).all():
        raise RuntimeError("confirmatory exact/wrong/unmatched accounting changed")
    agreement = population.p2a0_selected_query == population.lineage_child_query
    if population.loc[agreement, "arbiter_selected_query"].ne(
        population.loc[agreement, "p2a0_selected_query"]
    ).any():
        raise RuntimeError("agreement row was modified")
    agreement_probabilities = population.loc[
        agreement, ["p_lineage", "p_both_wrong"]
    ]
    if not agreement_probabilities.isna().all().all():
        raise RuntimeError("agreement row received head probabilities")
    verified_discipline = dict(marker_discipline_rows[0])
    return manifest, population, audit, verified_discipline


def analyze(
    *,
    report: Path = REPORT,
    incremental: Path = INCREMENTAL,
    artifact_path: Path = FROZEN_ARTIFACT,
) -> dict:
    frozen = load_frozen_arbiter(artifact_path)
    artifact_before = sha256_file(artifact_path)
    manifest, population, audit, verified_discipline = load_completed_population(
        report=report, incremental=incremental
    )
    metric_rows = []
    repair_rows = []
    scene_rows = []
    instance_rows = []
    scene_population = manifest.scene_token.astype(str).tolist()
    for protocol in PROTOCOLS:
        group = population.loc[
            population.protocol.astype(str) == protocol
        ].reset_index(drop=True)
        metrics = confirmatory_point_metrics(group)
        metric_rows.append({"protocol": protocol, **metrics})
        repair_rows.append({
            "protocol": protocol,
            **{name: int(metrics[name]) for name in REPAIR_COLUMNS},
        })
        scene, scene_redraws = paired_confirmatory_bootstrap(
            group,
            cluster_columns=SCENE_CLUSTER_COLUMNS,
            cluster_population=scene_population,
        )
        scene.insert(0, "protocol", protocol)
        scene["redraw_count"] = scene_redraws
        scene_rows.append(scene)
        instance, instance_redraws = paired_confirmatory_bootstrap(
            group, cluster_columns=INSTANCE_CLUSTER_COLUMNS
        )
        instance.insert(0, "protocol", protocol)
        instance["redraw_count"] = instance_redraws
        instance_rows.append(instance)

    metrics_frame = pd.DataFrame(metric_rows)
    repair_frame = pd.DataFrame(repair_rows)
    scene_frame = pd.concat(scene_rows, ignore_index=True)
    instance_frame = pd.concat(instance_rows, ignore_index=True)
    gate, status = evaluate_confirmatory_gate(
        metrics_frame, scene_frame, instance_frame, verified_discipline
    )
    decision = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "source_head": SOURCE_HEAD,
        "frozen_artifact_sha256": frozen.artifact_sha256,
        "tau_preference": frozen.runtime.tau_preference,
        "tau_defer": frozen.runtime.tau_defer,
        "protocols": list(PROTOCOLS),
        "all_three_protocols_required": True,
        "crash_mandatory": True,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "scene_cluster_columns": list(SCENE_CLUSTER_COLUMNS),
        "instance_cluster_columns": list(INSTANCE_CLUSTER_COLUMNS),
        "confirmatory_scenes": len(manifest),
        "full_population_rows": len(population),
        "heldout_lineage_status": audit["status"],
        "discipline_verified_from_all_markers": True,
        **verified_discipline,
    }
    if sha256_file(artifact_path) != artifact_before:
        raise RuntimeError("frozen artifact was modified during confirmatory analysis")
    atomic_csv(report / "protocol_metrics.csv", metrics_frame)
    atomic_csv(report / "repair_breakdown.csv", repair_frame)
    atomic_csv(report / "scene_cluster_bootstrap_summary.csv", scene_frame)
    atomic_csv(report / "instance_cluster_bootstrap_summary.csv", instance_frame)
    atomic_csv(report / "gate_summary.csv", gate)
    atomic_json(report / "decision.json", decision)
    return decision


def main() -> None:
    decision = analyze()
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
