#!/usr/bin/env python3
"""Fixed 5-fold scene-grouped feature-sufficiency diagnostics for R1-F0."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.care3d_p2a2_r1_features import (
    CHEAP_BASELINE_DECISIVE_AUROC,
    MODEL_FEATURE_COLUMNS,
    finite_model_matrix,
)
from analysis.care3d_p2a_association import PROTOCOLS


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/care3d/p2a2_r1_relative_evidence"
R0_REPORT = ROOT / "reports/care3d/p2a2_memory_lineage_r0"
RANDOM_STATE = 314159
N_SPLITS = 5
# sklearn requires explicitly supplied multiclass ROC labels to be ordered.
CLASSES = ("BOTH_WRONG", "LINEAGE_WINS", "P2A0_WINS")


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def load_rows() -> pd.DataFrame:
    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    if progress.get("completed_scenes") != 419:
        raise RuntimeError("R1-F0 analysis requires all 419 probe-train scenes")
    if progress.get("probe_val_read") is not False or progress.get("probe_test_read") is not False:
        raise RuntimeError("R1-F0 progress indicates held-out access")
    manifest = pd.read_csv(R0_REPORT / "probe_train_manifest.csv")
    if len(manifest) != 419 or set(manifest.split.astype(str)) != {"probe_train"}:
        raise RuntimeError("R1-F0 train-only manifest changed")
    frames = []
    for scene in manifest.scene_token.astype(str):
        prefix = REPORT / "incremental/probe_train" / scene
        marker = json.loads(prefix.with_suffix(".complete.json").read_text())
        frame = pd.read_csv(prefix.with_suffix(".rows.csv"))
        if not all((
            marker.get("complete"),
            marker.get("split") == "probe_train",
            marker.get("r0_assignment_exact") is True,
            marker.get("p2a0_frozen_cost_exact") is True,
            marker.get("feature_non_mutating") is True,
            marker.get("probe_val_read") is False,
            marker.get("probe_test_read") is False,
            int(marker.get("disagreement_rows", -1)) == len(frame),
        )):
            raise RuntimeError(f"invalid R1-F0 scene output: {scene}")
        if len(frame):
            frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    if set(rows.protocol.astype(str)) != set(PROTOCOLS):
        raise RuntimeError("R1-F0 protocol set changed")
    if (rows.p2a0_selected_query == rows.lineage_child_query).any():
        raise RuntimeError("R1-F0 analysis received agreement rows")
    if not (rows[["p2a0_wins", "lineage_wins", "both_wrong"]].astype(int).sum(axis=1) == 1).all():
        raise RuntimeError("R1-F0 offline labels changed")
    for flag in (
        "feature_computation_frozen_before_oracle",
        "gt_used_as_feature_input",
        "clean_future_used_as_feature_input",
        "oracle_query_used_as_feature_input",
    ):
        expected = flag == "feature_computation_frozen_before_oracle"
        if not (rows[flag].astype(bool) == expected).all():
            raise RuntimeError(f"R1-F0 leakage contract changed: {flag}")
    finite_model_matrix(rows)
    return rows


def fold_ids(rows: pd.DataFrame) -> np.ndarray:
    groups = rows.scene_token.astype(str).to_numpy()
    splitter = GroupKFold(n_splits=N_SPLITS)
    folds = np.full(len(rows), -1, dtype=np.int64)
    dummy = np.zeros((len(rows), 1), dtype=np.float64)
    for fold, (_, test) in enumerate(splitter.split(dummy, groups=groups)):
        folds[test] = fold
    if np.any(folds < 0):
        raise RuntimeError("R1-F0 GroupKFold did not assign every row")
    scene_fold = pd.DataFrame({"scene_token": groups, "fold": folds}).drop_duplicates()
    if scene_fold.scene_token.duplicated().any():
        raise RuntimeError("a scene crossed R1-F0 folds")
    return folds


def binary_oof(rows: pd.DataFrame, target: str, folds: np.ndarray) -> np.ndarray:
    x = finite_model_matrix(rows)
    y = rows[target].to_numpy(dtype=np.int64)
    probabilities = np.full(len(rows), np.nan, dtype=np.float64)
    for fold in range(N_SPLITS):
        train = folds != fold
        test = folds == fold
        if set(np.unique(y[train]).tolist()) != {0, 1}:
            raise RuntimeError(f"R1-F0 binary training fold lacks a class: {target} fold={fold}")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        )
        model.fit(x[train], y[train])
        probabilities[test] = model.predict_proba(x[test])[:, 1]
    if not np.isfinite(probabilities).all():
        raise RuntimeError("R1-F0 binary OOF prediction incomplete")
    return probabilities


def multiclass_oof(rows: pd.DataFrame, folds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = finite_model_matrix(rows)
    y = rows.outcome_class.astype(str).to_numpy()
    probabilities = np.full((len(rows), len(CLASSES)), np.nan, dtype=np.float64)
    predictions = np.empty(len(rows), dtype=object)
    for fold in range(N_SPLITS):
        train = folds != fold
        test = folds == fold
        if set(y[train]) != set(CLASSES):
            raise RuntimeError(f"R1-F0 multinomial training fold lacks a class: fold={fold}")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                multi_class="multinomial",
            ),
        )
        model.fit(x[train], y[train])
        classifier = model.named_steps["logisticregression"]
        raw = model.predict_proba(x[test])
        for class_index, name in enumerate(CLASSES):
            source = int(np.flatnonzero(classifier.classes_ == name)[0])
            probabilities[test, class_index] = raw[:, source]
        predictions[test] = model.predict(x[test])
    if not np.isfinite(probabilities).all():
        raise RuntimeError("R1-F0 multinomial OOF prediction incomplete")
    return probabilities, predictions.astype(str)


def main() -> None:
    rows = load_rows()
    all_folds = fold_ids(rows)
    fold_frame = pd.DataFrame({
        "scene_token": rows.scene_token.astype(str),
        "fold": all_folds,
    }).drop_duplicates().sort_values(["fold", "scene_token"])
    atomic_csv(REPORT / "groupkfold_scene_assignment.csv", fold_frame)

    decisive_mask = (rows.p2a0_wins.astype(bool) | rows.lineage_wins.astype(bool)).to_numpy()
    decisive = rows.loc[decisive_mask].reset_index(drop=True)
    decisive_folds = all_folds[decisive_mask]
    decisive_probability = binary_oof(decisive, "lineage_wins", decisive_folds)
    decisive_rows = []
    for protocol in PROTOCOLS:
        mask = decisive.protocol.astype(str).to_numpy() == protocol
        y = decisive.lineage_wins.to_numpy(np.int64)[mask]
        probability = decisive_probability[mask]
        decisive_rows.append({
            "protocol": protocol,
            "rows": int(mask.sum()),
            "lineage_win_rate": float(y.mean()),
            "auroc": float(roc_auc_score(y, probability)),
            "auprc": float(average_precision_score(y, probability)),
            "accuracy_at_0_5": float(accuracy_score(y, probability >= 0.5)),
            "lineage_choice_rate": float((probability >= 0.5).mean()),
        })
    decisive_frame = pd.DataFrame(decisive_rows)
    atomic_csv(REPORT / "decisive_preference_oof.csv", decisive_frame)

    both_wrong_probability = binary_oof(rows, "both_wrong", all_folds)
    both_wrong_rows = []
    for protocol in PROTOCOLS:
        mask = rows.protocol.astype(str).to_numpy() == protocol
        y = rows.both_wrong.to_numpy(np.int64)[mask]
        probability = both_wrong_probability[mask]
        both_wrong_rows.append({
            "protocol": protocol,
            "rows": int(mask.sum()),
            "both_wrong_rate": float(y.mean()),
            "auroc": float(roc_auc_score(y, probability)),
            "auprc": float(average_precision_score(y, probability)),
        })
    atomic_csv(REPORT / "both_wrong_oof.csv", pd.DataFrame(both_wrong_rows))

    multi_probability, multi_prediction = multiclass_oof(rows, all_folds)
    multi_rows = []
    confusion_rows = []
    for protocol in PROTOCOLS:
        mask = rows.protocol.astype(str).to_numpy() == protocol
        y = rows.outcome_class.astype(str).to_numpy()[mask]
        probability = multi_probability[mask]
        prediction = multi_prediction[mask]
        multi_rows.append({
            "protocol": protocol,
            "rows": int(mask.sum()),
            "macro_auroc_ovr": float(
                roc_auc_score(
                    y, probability, labels=list(CLASSES),
                    multi_class="ovr", average="macro",
                )
            ),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        })
        matrix = confusion_matrix(y, prediction, labels=list(CLASSES))
        for actual_index, actual in enumerate(CLASSES):
            for predicted_index, predicted in enumerate(CLASSES):
                confusion_rows.append({
                    "protocol": protocol,
                    "actual": actual,
                    "predicted": predicted,
                    "count": int(matrix[actual_index, predicted_index]),
                })
    atomic_csv(REPORT / "multiclass_oof.csv", pd.DataFrame(multi_rows))
    atomic_csv(REPORT / "multiclass_confusion_matrix.csv", pd.DataFrame(confusion_rows))

    gate_rows = []
    for row in decisive_frame.itertuples(index=False):
        cheap = float(CHEAP_BASELINE_DECISIVE_AUROC[str(row.protocol)])
        gate_rows.append({
            "protocol": str(row.protocol),
            "decisive_auroc": float(row.auroc),
            "min_auroc": 0.85,
            "cheap_feature_baseline_auroc": cheap,
            "auroc_at_least_0_85": bool(float(row.auroc) >= 0.85),
            "beats_cheap_feature_baseline": bool(float(row.auroc) > cheap),
        })
    gate_frame = pd.DataFrame(gate_rows)
    atomic_csv(REPORT / "f0_gate_summary.csv", gate_frame)
    pass_threshold = int(gate_frame.auroc_at_least_0_85.astype(bool).sum()) >= 2
    beats_all = bool(gate_frame.beats_cheap_feature_baseline.astype(bool).all())
    passed = bool(pass_threshold and beats_all)
    result = {
        "schema_version": 1,
        "status": (
            "PASS_R1_RELATIVE_EVIDENCE"
            if passed else "STOP_RELATIVE_EVIDENCE_INSUFFICIENT"
        ),
        "feature_sufficiency_diagnostic": True,
        "final_r1_model": False,
        "group_kfold_splits": N_SPLITS,
        "group": "scene_token",
        "protocol_used_as_feature": False,
        "model_family_search": False,
        "threshold_search": False,
        "protocols_at_or_above_0_85": int(
            gate_frame.auroc_at_least_0_85.astype(bool).sum()
        ),
        "all_protocols_beat_cheap_baseline": beats_all,
        "probe_val_read": False,
        "probe_test_read": False,
    }
    atomic_json(REPORT / "decision.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
