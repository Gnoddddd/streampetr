#!/usr/bin/env python3
"""Fit the final R1 deployment artifact after a successful development Gate.

This entry point is intentionally separate from nested scientific evaluation.
It must not be run until the development ``decision.json`` reports GO.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.care3d_p2a2_r1_arbiter import (
    MODEL_PARAMETERS,
    OUTER_SPLITS,
    assign_scene_folds,
    export_head,
    fit_ambiguity_head,
    fit_preference_head,
    select_thresholds,
)
from analysis.care3d_p2a2_r1_features import MODEL_FEATURE_COLUMNS, finite_model_matrix
from scripts.analyze_care3d_p2a2_r1_arbiter import (
    REPORT,
    SCHEMA_VERSION,
    SOURCE_HEAD,
    atomic_json,
    model_feature_columns_sha256,
    validate_and_load_sources,
)


GO_STATUS = "GO_P2A2_R1_DISAGREEMENT_GATED_IDENTITY_ARBITER"


def _require_development_go() -> dict:
    path = REPORT / "decision.json"
    if not path.is_file():
        raise RuntimeError("final freeze is locked pending the development Gate")
    decision = json.loads(path.read_text())
    if decision.get("status") != GO_STATUS:
        raise RuntimeError("final freeze is locked because the development Gate is not GO")
    if decision.get("probe_val_read") is not False or decision.get("probe_test_read") is not False:
        raise RuntimeError("development Gate metadata indicates held-out access")
    return decision


def development_oof_probabilities(
    disagreement: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Create the frozen 5-fold scene-grouped raw probability table."""
    folds = assign_scene_folds(
        manifest, n_splits=OUTER_SPLITS, fold_column="fold"
    )
    scene_to_fold = folds.set_index("scene_token").fold
    row_folds = disagreement.scene_token.astype(str).map(scene_to_fold)
    if row_folds.isna().any():
        raise RuntimeError("final threshold OOF fold coverage is incomplete")
    output = disagreement.loc[:, [
        "scene_token", "instance_token", "anchor_frame_idx", "target_frame_idx", "protocol"
    ]].copy()
    output["inner_p_lineage"] = np.nan
    output["inner_p_both_wrong"] = np.nan
    for fold in range(OUTER_SPLITS):
        validation = row_folds.to_numpy(int) == fold
        train = ~validation
        train_scenes = set(disagreement.loc[train, "scene_token"].astype(str))
        validation_scenes = set(disagreement.loc[validation, "scene_token"].astype(str))
        if train_scenes & validation_scenes:
            raise RuntimeError("final threshold OOF scene leakage")
        preference = fit_preference_head(disagreement.loc[train])
        ambiguity = fit_ambiguity_head(disagreement.loc[train])
        matrix = finite_model_matrix(disagreement.loc[validation])
        output.loc[validation, "inner_p_lineage"] = preference.predict_proba(matrix)[:, 1]
        output.loc[validation, "inner_p_both_wrong"] = ambiguity.predict_proba(matrix)[:, 1]
    if not np.isfinite(output[["inner_p_lineage", "inner_p_both_wrong"]].to_numpy(float)).all():
        raise RuntimeError("final threshold OOF probabilities are incomplete")
    return output


def main() -> None:
    _require_development_go()
    manifest, full, disagreement, source = validate_and_load_sources()
    raw_oof = development_oof_probabilities(disagreement, manifest)
    thresholds = select_thresholds(raw_oof, full)
    preference = fit_preference_head(disagreement)
    ambiguity = fit_ambiguity_head(disagreement)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "source_r1_f0_head": SOURCE_HEAD,
        "source_r1_f0_decision_sha256": source["r1_f0_decision_sha256"],
        "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
        "model_feature_columns_sha256": model_feature_columns_sha256(),
        "preference_model": export_head(preference),
        "ambiguity_model": export_head(ambiguity),
        "tau_preference": float(thresholds["tau_preference"]),
        "tau_defer": float(thresholds["tau_defer"]),
        "solver": MODEL_PARAMETERS["solver"],
        "penalty": MODEL_PARAMETERS["penalty"],
        "C": MODEL_PARAMETERS["C"],
        "class_weight": MODEL_PARAMETERS["class_weight"],
        "random_state": MODEL_PARAMETERS["random_state"],
        "preference_training_rows": int(
            (disagreement.p2a0_wins.astype(bool) | disagreement.lineage_wins.astype(bool)).sum()
        ),
        "ambiguity_training_rows": int(len(disagreement)),
        "probe_val_read": False,
        "probe_test_read": False,
        "confirmatory_validation": False,
    }
    atomic_json(REPORT / "frozen_arbiter.json", artifact)
    print(json.dumps({
        "status": "P2A2_R1_FINAL_ARBITER_FROZEN",
        "tau_preference": artifact["tau_preference"],
        "tau_defer": artifact["tau_defer"],
        "probe_val_read": False,
        "probe_test_read": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
