#!/usr/bin/env python3
"""Run the preregistered CARE-3D P2-A2-R1 nested development analysis."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from analysis.care3d_p2a2_r1_arbiter import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    INNER_SPLITS,
    OUTER_SPLITS,
    assign_scene_folds,
    evaluate_final_gate,
    nested_outer_oof,
    paired_cluster_bootstrap,
    protocol_point_metrics,
    reconstruct_full_population,
    repair_breakdown,
)
from analysis.care3d_p2a2_r1_features import MODEL_FEATURE_COLUMNS
from analysis.care3d_p2a2_r1_features import finite_model_matrix
from analysis.care3d_p2a_association import PROTOCOLS


ROOT = Path(__file__).resolve().parents[1]
R0_REPORT = ROOT / "reports/care3d/p2a2_memory_lineage_r0"
R1_F0_REPORT = ROOT / "reports/care3d/p2a2_r1_relative_evidence/schema_2"
REPORT = ROOT / "reports/care3d/p2a2_r1_identity_arbiter"
SOURCE_HEAD = "459dcb73a4b721d7322f89ef760dbb01418a0ff0"
SCHEMA_VERSION = 1
EXPECTED_SCENES = 419
EXPECTED_ELIGIBLE_OBJECTS = 91995
EXPECTED_PROTOCOL_ROWS = 275985
EXPECTED_DISAGREEMENT_ROWS = 34342


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_feature_columns_sha256() -> str:
    serialized = json.dumps(
        list(MODEL_FEATURE_COLUMNS), separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing source metadata: {path}")
    return json.loads(path.read_text())


def _heldout_locked(value: dict, name: str) -> None:
    if value.get("probe_val_read") is not False or value.get("probe_test_read") is not False:
        raise RuntimeError(f"{name} indicates held-out access")


def _load_scene_rows(
    report: Path,
    manifest: pd.DataFrame,
    *,
    schema_version: int,
    count_field: str,
) -> tuple[pd.DataFrame, int]:
    frames = []
    eligible_objects = 0
    for scene in manifest.scene_token.astype(str):
        prefix = report / "incremental/probe_train" / scene
        marker = _read_json(prefix.with_suffix(".complete.json"))
        rows_path = prefix.with_suffix(".rows.csv")
        if not rows_path.is_file():
            raise RuntimeError(f"missing source scene rows: {scene}")
        if not all((
            marker.get("complete") is True,
            marker.get("schema_version") == schema_version,
            marker.get("split") == "probe_train",
            str(marker.get("scene_token")) == scene,
            marker.get("probe_val_read") is False,
            marker.get("probe_test_read") is False,
        )):
            raise RuntimeError(f"invalid source scene marker: {scene}")
        frame = pd.read_csv(rows_path)
        if len(frame) != int(marker.get(count_field, -1)):
            raise RuntimeError(f"source marker/row mismatch: {scene}")
        eligible_objects += int(marker.get("eligible_objects", 0))
        if len(frame):
            frames.append(frame)
    if not frames:
        raise RuntimeError("source cohort contains no rows")
    return pd.concat(frames, ignore_index=True), eligible_objects


def validate_and_load_sources(
    *,
    r0_report: Path = R0_REPORT,
    r1_f0_report: Path = R1_F0_REPORT,
    output_report: Path = REPORT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Perform the permitted source-structure smoke and write its audit."""
    r0_progress_path = r0_report / "progress_manifest.json"
    r0_decision_path = r0_report / "decision.json"
    r1_progress_path = r1_f0_report / "progress_manifest.json"
    r1_decision_path = r1_f0_report / "decision.json"
    r0_progress = _read_json(r0_progress_path)
    r0_decision = _read_json(r0_decision_path)
    r1_progress = _read_json(r1_progress_path)
    r1_decision = _read_json(r1_decision_path)
    for value, name in (
        (r0_progress, "R0 progress"),
        (r0_decision, "R0 decision"),
        (r1_progress, "R1-F0 progress"),
        (r1_decision, "R1-F0 decision"),
    ):
        _heldout_locked(value, name)
    if r0_progress.get("completed_scenes") != EXPECTED_SCENES:
        raise RuntimeError("R0 completed scene count changed")
    if r0_progress.get("schema_version") != 1:
        raise RuntimeError("R0 progress schema version changed")
    if r0_progress.get("rows") != EXPECTED_PROTOCOL_ROWS:
        raise RuntimeError("R0 protocol row count changed")
    if r1_progress.get("schema_version") != 2:
        raise RuntimeError("R1-F0 schema version changed")
    if r1_progress.get("completed_scenes") != EXPECTED_SCENES:
        raise RuntimeError("R1-F0 completed scene count changed")
    if r1_progress.get("disagreement_rows") != EXPECTED_DISAGREEMENT_ROWS:
        raise RuntimeError("R1-F0 disagreement row count changed")
    if r1_decision.get("schema_version") != 2:
        raise RuntimeError("R1-F0 decision schema version changed")
    if r1_decision.get("status") != "PASS_R1_RELATIVE_EVIDENCE":
        raise RuntimeError("R1-F0 source Gate did not pass")

    manifest = pd.read_csv(r0_report / "probe_train_manifest.csv")
    if (
        len(manifest) != EXPECTED_SCENES
        or manifest.scene_token.astype(str).nunique() != EXPECTED_SCENES
        or set(manifest.split.astype(str)) != {"probe_train"}
    ):
        raise RuntimeError("R0 419-scene probe-train manifest changed")
    r0_rows, r0_eligible = _load_scene_rows(
        r0_report, manifest, schema_version=1, count_field="rows"
    )
    r1_rows, r1_eligible = _load_scene_rows(
        r1_f0_report, manifest, schema_version=2, count_field="disagreement_rows"
    )
    if len(r0_rows) != EXPECTED_PROTOCOL_ROWS:
        raise RuntimeError("R0 full population is incomplete")
    if len(r1_rows) != EXPECTED_DISAGREEMENT_ROWS:
        raise RuntimeError("R1-F0 disagreement population is incomplete")
    if r0_eligible != EXPECTED_ELIGIBLE_OBJECTS or r1_eligible != EXPECTED_ELIGIBLE_OBJECTS:
        raise RuntimeError("eligible object count changed")
    if set(r0_rows.protocol.astype(str)) != set(PROTOCOLS):
        raise RuntimeError("R0 protocol set changed")
    if set(r1_rows.protocol.astype(str)) != set(PROTOCOLS):
        raise RuntimeError("R1-F0 protocol set changed")
    for flag in (
        "gt_used_as_association_input",
        "clean_future_used_as_association_input",
        "oracle_query_used_as_association_input",
    ):
        if not (r0_rows[flag] == False).all():  # noqa: E712 - exact audit value
            raise RuntimeError(f"R0 leakage flag changed: {flag}")
    if not (r1_rows["feature_computation_frozen_before_oracle"] == True).all():  # noqa: E712
        raise RuntimeError("R1-F0 feature/label ordering changed")
    for flag in (
        "gt_used_as_feature_input",
        "clean_future_used_as_feature_input",
        "oracle_query_used_as_feature_input",
    ):
        if not (r1_rows[flag] == False).all():  # noqa: E712 - exact audit value
            raise RuntimeError(f"R1-F0 leakage flag changed: {flag}")
    if not (
        r1_rows[["p2a0_wins", "lineage_wins", "both_wrong"]]
        .astype(int)
        .sum(axis=1)
        == 1
    ).all():
        raise RuntimeError("R1-F0 outcome labels are not exhaustive")
    finite_model_matrix(r1_rows)
    for protocol in PROTOCOLS:
        if int((r0_rows.protocol.astype(str) == protocol).sum()) != EXPECTED_ELIGIBLE_OBJECTS:
            raise RuntimeError(f"R0 protocol population changed: {protocol}")
    full, disagreement = reconstruct_full_population(r0_rows, r1_rows)
    if len(full) - len(disagreement) + len(disagreement) != EXPECTED_PROTOCOL_ROWS:
        raise RuntimeError("R0 agreement/disagreement accounting changed")
    validation = {
        "schema_version": SCHEMA_VERSION,
        "source_r1_f0_head": SOURCE_HEAD,
        "r0_completed_scenes": EXPECTED_SCENES,
        "r0_eligible_objects": r0_eligible,
        "r0_protocol_rows": len(full),
        "r1_f0_schema_version": 2,
        "r1_f0_completed_scenes": EXPECTED_SCENES,
        "r1_f0_disagreement_rows": len(disagreement),
        "r1_f0_status": r1_decision["status"],
        "r0_r1_disagreement_key_exact": True,
        "r0_r1_identity_fields_exact": True,
        "r1_f0_decision_sha256": _sha256(r1_decision_path),
        "r0_decision_sha256": _sha256(r0_decision_path),
        "r0_progress_sha256": _sha256(r0_progress_path),
        "model_feature_columns_sha256": model_feature_columns_sha256(),
        "probe_val_read": False,
        "probe_test_read": False,
    }
    atomic_json(output_report / "source_validation.json", validation)
    return manifest, full, disagreement, validation


def _ordered_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    minimum = [
        "scene_token", "instance_token", "anchor_frame_idx", "target_frame_idx",
        "protocol", "outer_fold", "p2a0_selected_query", "lineage_child_query",
        "oracle_query_index", "p_lineage", "p_both_wrong", "tau_preference",
        "tau_defer", "arbiter_selected_query", "arbiter_decision",
        "arbiter_exact", "arbiter_wrong", "arbiter_unmatched",
    ]
    remainder = [column for column in frame.columns if column not in minimum]
    return frame.loc[:, minimum + remainder]


def main() -> None:
    manifest, full, disagreement, _validation = validate_and_load_sources()
    outer_folds = assign_scene_folds(
        manifest, n_splits=OUTER_SPLITS, fold_column="outer_fold"
    )
    atomic_csv(REPORT / "outer_scene_folds.csv", outer_folds)
    full_oof, disagreement_oof, thresholds = nested_outer_oof(
        full, disagreement, manifest
    )
    atomic_csv(
        REPORT / "outer_fold_thresholds.csv",
        thresholds.sort_values("outer_fold").reset_index(drop=True),
    )
    atomic_csv(
        REPORT / "nested_oof_disagreement_predictions.csv",
        _ordered_prediction_columns(disagreement_oof),
    )
    atomic_csv(
        REPORT / "nested_oof_full_population.csv",
        _ordered_prediction_columns(full_oof),
    )

    metric_rows = []
    repair_rows = []
    scene_rows = []
    instance_rows = []
    scene_population = manifest.scene_token.astype(str).tolist()
    for protocol in PROTOCOLS:
        group = full_oof.loc[full_oof.protocol.astype(str) == protocol].reset_index(drop=True)
        metrics = protocol_point_metrics(group)
        metric_rows.append({"protocol": protocol, **metrics})
        repair_rows.append({"protocol": protocol, **repair_breakdown(group)})
        scene_summary, scene_redraws = paired_cluster_bootstrap(
            group,
            cluster_columns=("scene_token",),
            cluster_population=scene_population,
            replicates=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED,
        )
        scene_summary.insert(0, "protocol", protocol)
        scene_summary["redraw_count"] = scene_redraws
        scene_rows.append(scene_summary)
        instance_summary, instance_redraws = paired_cluster_bootstrap(
            group,
            cluster_columns=("scene_token", "instance_token"),
            replicates=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED,
        )
        instance_summary.insert(0, "protocol", protocol)
        instance_summary["redraw_count"] = instance_redraws
        instance_rows.append(instance_summary)
    metrics_frame = pd.DataFrame(metric_rows)
    repair_frame = pd.DataFrame(repair_rows)
    scene_frame = pd.concat(scene_rows, ignore_index=True)
    instance_frame = pd.concat(instance_rows, ignore_index=True)
    atomic_csv(REPORT / "protocol_metrics.csv", metrics_frame)
    atomic_csv(REPORT / "repair_breakdown.csv", repair_frame)
    atomic_csv(REPORT / "scene_cluster_bootstrap_summary.csv", scene_frame)
    atomic_csv(REPORT / "instance_cluster_bootstrap_summary.csv", instance_frame)

    discipline = {
        "protocol_used_as_feature": False,
        "protocol_specific_model": False,
        "protocol_specific_threshold": False,
        "model_family_search": False,
        "feature_search": False,
        "calibration_search": False,
        "outer_test_used_for_threshold_selection": False,
        "agreement_bypass_exact": True,
        "probe_val_read": False,
        "probe_test_read": False,
    }
    gate, status = evaluate_final_gate(
        metrics_frame, scene_frame, instance_frame, discipline
    )
    atomic_csv(REPORT / "gate_summary.csv", gate)
    decision = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "development_pre_gate": True,
        "confirmatory_validation": False,
        "final_r1_model_frozen": False,
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "threshold_selection": "nested_inner_oof",
        **discipline,
    }
    atomic_json(REPORT / "decision.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
