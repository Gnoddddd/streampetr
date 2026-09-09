"""Frozen helpers for the CARE-3D P2-A2-R1 identity arbiter.

The only model inputs are the schema-2 ``MODEL_FEATURE_COLUMNS`` supplied by
``finite_model_matrix``.  Ground-truth identity is accepted only by the
offline evaluation helpers after probabilities and decisions are frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.care3d_p2a2_r1_features import finite_model_matrix
from analysis.care3d_p2a_association import PROTOCOLS


OUTER_SPLITS = 5
INNER_SPLITS = 4
RANDOM_STATE = 314159
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 314159
KEY_COLUMNS = (
    "scene_token",
    "instance_token",
    "anchor_frame_idx",
    "target_frame_idx",
    "protocol",
)
PREFERENCE_THRESHOLDS = tuple(round(value / 100.0, 2) for value in range(5, 96)) + (1.01,)
DEFER_THRESHOLDS = tuple(round(value / 100.0, 2) for value in range(50, 100)) + (1.01,)
MODEL_PARAMETERS = {
    "solver": "lbfgs",
    "penalty": "l2",
    "C": 1.0,
    "max_iter": 2000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}


def make_binary_head() -> Pipeline:
    """Construct either preregistered binary head without family search."""
    return Pipeline([
        ("standardscaler", StandardScaler()),
        ("logisticregression", LogisticRegression(**MODEL_PARAMETERS)),
    ])


def assign_scene_folds(
    manifest: pd.DataFrame,
    *,
    n_splits: int,
    fold_column: str,
) -> pd.DataFrame:
    """Assign every manifest scene to exactly one deterministic GroupKFold."""
    if "scene_token" not in manifest or manifest.scene_token.isna().any():
        raise RuntimeError("scene manifest lacks scene_token")
    scenes = manifest.loc[:, ["scene_token"]].copy()
    scenes["scene_token"] = scenes.scene_token.astype(str)
    if scenes.scene_token.duplicated().any():
        raise RuntimeError("scene manifest contains duplicate scene_token")
    if len(scenes) < int(n_splits):
        raise RuntimeError("scene manifest is too small for GroupKFold")
    groups = scenes.scene_token.to_numpy()
    folds = np.full(len(scenes), -1, dtype=np.int64)
    dummy = np.zeros((len(scenes), 1), dtype=np.float64)
    splitter = GroupKFold(n_splits=int(n_splits))
    for fold, (_, validation) in enumerate(splitter.split(dummy, groups=groups)):
        folds[validation] = fold
    if np.any(folds < 0):
        raise RuntimeError("GroupKFold did not assign every manifest scene")
    scenes[fold_column] = folds
    if scenes.scene_token.nunique() != len(scenes):
        raise RuntimeError("a scene crossed folds")
    return scenes


def _binary_labels(frame: pd.DataFrame, column: str) -> np.ndarray:
    labels = frame[column].to_numpy(dtype=np.int64)
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise RuntimeError(f"binary head training data lacks a class: {column}")
    return labels


def fit_preference_head(disagreement: pd.DataFrame) -> Pipeline:
    decisive = disagreement.loc[
        disagreement.p2a0_wins.astype(bool) | disagreement.lineage_wins.astype(bool)
    ]
    if len(decisive) == 0 or decisive.both_wrong.astype(bool).any():
        raise RuntimeError("Preference Head requires decisive disagreement rows only")
    model = make_binary_head()
    model.fit(finite_model_matrix(decisive), _binary_labels(decisive, "lineage_wins"))
    return model


def fit_ambiguity_head(disagreement: pd.DataFrame) -> Pipeline:
    if len(disagreement) == 0:
        raise RuntimeError("Ambiguity Head requires disagreement rows")
    model = make_binary_head()
    model.fit(
        finite_model_matrix(disagreement),
        _binary_labels(disagreement, "both_wrong"),
    )
    return model


def apply_probabilities(
    p2a0_candidates: Sequence[int],
    lineage_candidates: Sequence[int],
    p_lineage: Sequence[float],
    p_both_wrong: Sequence[float],
    *,
    tau_preference: float,
    tau_defer: float,
) -> pd.DataFrame:
    """Apply agreement bypass, then ambiguity defer, then preference."""
    a = np.asarray(p2a0_candidates, dtype=np.int64)
    lineage = np.asarray(lineage_candidates, dtype=np.int64)
    preference = np.asarray(p_lineage, dtype=np.float64)
    ambiguity = np.asarray(p_both_wrong, dtype=np.float64)
    if any(value.ndim != 1 for value in (a, lineage, preference, ambiguity)):
        raise ValueError("arbiter inputs must be one-dimensional")
    if not (a.shape == lineage.shape == preference.shape == ambiguity.shape):
        raise ValueError("arbiter inputs must be aligned")
    disagreement = a != lineage
    if not np.isfinite(preference[disagreement]).all():
        raise RuntimeError("disagreement p_lineage is incomplete")
    if not np.isfinite(ambiguity[disagreement]).all():
        raise RuntimeError("disagreement p_both_wrong is incomplete")

    selected = a.copy()
    decision = np.full(len(a), "AGREEMENT", dtype=object)
    defer = disagreement & (ambiguity >= float(tau_defer))
    choose_lineage = (
        disagreement
        & ~defer
        & (preference >= float(tau_preference))
    )
    choose_p2a0 = disagreement & ~defer & ~choose_lineage
    selected[defer] = -1
    selected[choose_lineage] = lineage[choose_lineage]
    decision[defer] = "DEFER"
    decision[choose_lineage] = "LINEAGE"
    decision[choose_p2a0] = "P2A0"
    return pd.DataFrame({
        "p_lineage": preference,
        "p_both_wrong": ambiguity,
        "arbiter_selected_query": selected,
        "arbiter_decision": decision,
    })


def runtime_arbiter(
    model_features: np.ndarray,
    p2a0_candidates: Sequence[int],
    lineage_candidates: Sequence[int],
    preference_head: Any,
    ambiguity_head: Any,
    *,
    tau_preference: float,
    tau_defer: float,
) -> pd.DataFrame:
    """Run both heads only on disagreement rows; agreement is an exact bypass."""
    matrix = np.asarray(model_features, dtype=np.float64)
    a = np.asarray(p2a0_candidates, dtype=np.int64)
    lineage = np.asarray(lineage_candidates, dtype=np.int64)
    if matrix.ndim != 2 or len(matrix) != len(a) or a.shape != lineage.shape:
        raise ValueError("runtime feature/candidate layout changed")
    disagreement = a != lineage
    preference = np.full(len(a), np.nan, dtype=np.float64)
    ambiguity = np.full(len(a), np.nan, dtype=np.float64)
    if disagreement.any():
        preference[disagreement] = preference_head.predict_proba(
            matrix[disagreement]
        )[:, 1]
        ambiguity[disagreement] = ambiguity_head.predict_proba(
            matrix[disagreement]
        )[:, 1]
    return apply_probabilities(
        a,
        lineage,
        preference,
        ambiguity,
        tau_preference=tau_preference,
        tau_defer=tau_defer,
    )


def add_offline_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach exhaustive offline outcomes after selected identities are frozen."""
    output = frame.copy()
    selected = output.arbiter_selected_query.to_numpy(dtype=np.int64)
    p2a0 = output.p2a0_selected_query.to_numpy(dtype=np.int64)
    oracle = output.oracle_query_index.to_numpy(dtype=np.int64)
    if np.any(oracle < 0):
        raise RuntimeError("offline oracle identity must be a real query")
    output["p2a0_exact"] = (p2a0 == oracle).astype(np.int8)
    output["p2a0_wrong"] = ((p2a0 >= 0) & (p2a0 != oracle)).astype(np.int8)
    output["p2a0_unmatched"] = (p2a0 == -1).astype(np.int8)
    output["arbiter_exact"] = (selected == oracle).astype(np.int8)
    output["arbiter_wrong"] = ((selected >= 0) & (selected != oracle)).astype(np.int8)
    output["arbiter_unmatched"] = (selected == -1).astype(np.int8)
    for prefix in ("p2a0", "arbiter"):
        total = output[[f"{prefix}_exact", f"{prefix}_wrong", f"{prefix}_unmatched"]].sum(axis=1)
        if not (total == 1).all():
            raise RuntimeError(f"{prefix} outcomes are not exhaustive")
    return output


def _key_index(frame: pd.DataFrame, name: str) -> pd.MultiIndex:
    missing = set(KEY_COLUMNS) - set(frame.columns)
    if missing:
        raise RuntimeError(f"{name} lacks key columns: {sorted(missing)}")
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError(f"{name} contains duplicate identity keys")
    return pd.MultiIndex.from_frame(frame.loc[:, KEY_COLUMNS])


def reconstruct_full_population(
    r0_rows: pd.DataFrame,
    r1_disagreement: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate exact R0/R1 disagreement continuity without dropping R0 rows."""
    full = r0_rows.reset_index(drop=True).copy()
    r1 = r1_disagreement.reset_index(drop=True).copy()
    r0_index = _key_index(full, "R0 full population")
    r1_index = _key_index(r1, "R1-F0 disagreement")
    disagreement = full.p2a0_selected_query.to_numpy(dtype=np.int64) != full.lineage_child_query.to_numpy(dtype=np.int64)
    r0_disagreement = full.loc[disagreement].copy()
    r0_disagreement_index = _key_index(r0_disagreement, "R0 disagreement")
    if set(r0_disagreement_index.tolist()) != set(r1_index.tolist()):
        raise RuntimeError("R0/R1-F0 disagreement key set changed")
    if len(r0_index) != len(full):
        raise RuntimeError("R0 full population key accounting changed")
    r0_compare = r0_disagreement.set_index(list(KEY_COLUMNS)).sort_index()
    r1_compare = r1.set_index(list(KEY_COLUMNS)).sort_index()
    for column in ("p2a0_selected_query", "lineage_child_query", "oracle_query_index"):
        if not np.array_equal(
            r0_compare[column].to_numpy(dtype=np.int64),
            r1_compare[column].to_numpy(dtype=np.int64),
        ):
            raise RuntimeError(f"R0/R1-F0 identity field changed: {column}")
    return full, r1


def _merge_disagreement_probabilities(
    full_population: pd.DataFrame,
    disagreement_predictions: pd.DataFrame,
) -> pd.DataFrame:
    full = full_population.copy()
    _key_index(full, "full population")
    _key_index(disagreement_predictions, "disagreement predictions")
    columns = list(KEY_COLUMNS) + ["p_lineage", "p_both_wrong"]
    merged = full.merge(
        disagreement_predictions.loc[:, columns],
        how="left",
        on=list(KEY_COLUMNS),
        validate="one_to_one",
        sort=False,
    )
    disagreement = merged.p2a0_selected_query.to_numpy(dtype=np.int64) != merged.lineage_child_query.to_numpy(dtype=np.int64)
    if not np.isfinite(merged.loc[disagreement, ["p_lineage", "p_both_wrong"]].to_numpy(float)).all():
        raise RuntimeError("full population lacks disagreement probabilities")
    if merged.loc[~disagreement, ["p_lineage", "p_both_wrong"]].notna().any().any():
        raise RuntimeError("agreement rows unexpectedly received model probabilities")
    return merged


def decide_full_population(
    full_population: pd.DataFrame,
    disagreement_predictions: pd.DataFrame,
    *,
    tau_preference: float,
    tau_defer: float,
) -> pd.DataFrame:
    merged = _merge_disagreement_probabilities(full_population, disagreement_predictions)
    decisions = apply_probabilities(
        merged.p2a0_selected_query,
        merged.lineage_child_query,
        merged.p_lineage,
        merged.p_both_wrong,
        tau_preference=tau_preference,
        tau_defer=tau_defer,
    )
    for column in ("arbiter_selected_query", "arbiter_decision"):
        merged[column] = decisions[column].to_numpy()
    return add_offline_outcomes(merged)


def protocol_point_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    if len(frame) == 0:
        raise ValueError("protocol metrics require rows")
    agreement = frame.p2a0_selected_query.to_numpy(int) == frame.lineage_child_query.to_numpy(int)
    baseline_exact = frame.p2a0_exact.to_numpy(bool)
    baseline_wrong = frame.p2a0_wrong.to_numpy(bool)
    exact = frame.arbiter_exact.to_numpy(bool)
    wrong = frame.arbiter_wrong.to_numpy(bool)
    unmatched = frame.arbiter_unmatched.to_numpy(bool)
    decision = frame.arbiter_decision.astype(str).to_numpy()
    selected = frame.arbiter_selected_query.to_numpy(int)
    a = frame.p2a0_selected_query.to_numpy(int)
    result: dict[str, float | int] = {
        "rows": int(len(frame)),
        "agreement_rows": int(agreement.sum()),
        "disagreement_rows": int((~agreement).sum()),
        "agreement_rate": float(agreement.mean()),
        "disagreement_rate": float((~agreement).mean()),
        "p2a0_exact_rate": float(baseline_exact.mean()),
        "p2a0_wrong_rate": float(baseline_wrong.mean()),
        "p2a0_unmatched_rate": float(frame.p2a0_unmatched.astype(float).mean()),
        "arbiter_exact_rate": float(exact.mean()),
        "arbiter_wrong_rate": float(wrong.mean()),
        "arbiter_unmatched_rate": float(unmatched.mean()),
        "delta_exact_rate": float(exact.mean() - baseline_exact.mean()),
        "delta_wrong_rate": float(wrong.mean() - baseline_wrong.mean()),
        "wrong_reduction": float(baseline_wrong.mean() - wrong.mean()),
        "wrong_repaired_count": int((baseline_wrong & exact).sum()),
        "correct_broken_to_wrong_count": int((baseline_exact & wrong).sum()),
        "correct_broken_to_unmatched_count": int((baseline_exact & unmatched).sum()),
        "wrong_to_unmatched_count": int((baseline_wrong & unmatched).sum()),
        "wrong_remaining_wrong_count": int((baseline_wrong & wrong).sum()),
        "lineage_switch_count": int((decision == "LINEAGE").sum()),
        "lineage_switch_rate_all": float((decision == "LINEAGE").mean()),
        "lineage_switch_rate_disagreement": float((decision[~agreement] == "LINEAGE").mean()) if (~agreement).any() else 0.0,
        "defer_count": int((decision == "DEFER").sum()),
        "defer_rate_all": float((decision == "DEFER").mean()),
        "defer_rate_disagreement": float((decision[~agreement] == "DEFER").mean()) if (~agreement).any() else 0.0,
        "agreement_modified_count": int((selected[agreement] != a[agreement]).sum()),
    }
    if result["agreement_modified_count"] != 0:
        raise RuntimeError("Agreement Bypass modified an agreement row")
    if int(baseline_wrong.sum()) != (
        result["wrong_repaired_count"]
        + result["wrong_to_unmatched_count"]
        + result["wrong_remaining_wrong_count"]
    ):
        raise RuntimeError("wrong-row repair accounting is inconsistent")
    correct_remaining_exact = int((baseline_exact & exact).sum())
    if int(baseline_exact.sum()) != (
        correct_remaining_exact
        + result["correct_broken_to_wrong_count"]
        + result["correct_broken_to_unmatched_count"]
    ):
        raise RuntimeError("correct-row break accounting is inconsistent")
    return result


def repair_breakdown(frame: pd.DataFrame) -> dict[str, int]:
    metrics = protocol_point_metrics(frame)
    names = (
        "wrong_repaired_count",
        "correct_broken_to_wrong_count",
        "correct_broken_to_unmatched_count",
        "wrong_to_unmatched_count",
        "wrong_remaining_wrong_count",
    )
    return {name: int(metrics[name]) for name in names}


def threshold_sort_key(candidate: Mapping[str, float]) -> tuple[float, ...]:
    """The exact preregistered deterministic lexicographic objective."""
    return (
        float(candidate["max_protocol_wrong_rate"]),
        -float(candidate["min_protocol_delta_exact"]),
        float(candidate["mean_protocol_wrong_rate"]),
        -float(candidate["mean_protocol_delta_exact"]),
        float(candidate["max_protocol_unmatched_rate"]),
        float(candidate["mean_protocol_unmatched_rate"]),
        -float(candidate["tau_defer"]),
        abs(float(candidate["tau_preference"]) - 0.5),
        -float(candidate["tau_preference"]),
    )


def select_lexicographic_candidate(candidates: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not candidates:
        raise RuntimeError("threshold feasible set is empty")
    return dict(min(candidates, key=threshold_sort_key))


def _threshold_candidate_metrics(
    merged: pd.DataFrame,
    tau_preference: float,
    tau_defer: float,
) -> dict[str, float] | None:
    decisions = apply_probabilities(
        merged.p2a0_selected_query,
        merged.lineage_child_query,
        merged.p_lineage,
        merged.p_both_wrong,
        tau_preference=tau_preference,
        tau_defer=tau_defer,
    )
    selected = decisions.arbiter_selected_query.to_numpy(int)
    oracle = merged.oracle_query_index.to_numpy(int)
    wrong = (selected >= 0) & (selected != oracle)
    unmatched = selected == -1
    exact = selected == oracle
    baseline_exact = merged.p2a0_selected_query.to_numpy(int) == oracle
    candidate: dict[str, float] = {
        "tau_preference": float(tau_preference),
        "tau_defer": float(tau_defer),
    }
    wrong_rates = []
    unmatched_rates = []
    delta_exact_rates = []
    protocol_labels = merged.protocol.astype(str).to_numpy()
    for protocol in PROTOCOLS:
        mask = protocol_labels == protocol
        if not mask.any():
            raise RuntimeError(f"threshold population lacks protocol: {protocol}")
        protocol_wrong = float(wrong[mask].mean())
        protocol_unmatched = float(unmatched[mask].mean())
        protocol_delta_exact = float(exact[mask].mean() - baseline_exact[mask].mean())
        short = protocol.split("_")[0]
        candidate[f"inner_{short}_wrong"] = protocol_wrong
        candidate[f"inner_{short}_unmatched"] = protocol_unmatched
        candidate[f"inner_{short}_delta_exact"] = protocol_delta_exact
        wrong_rates.append(protocol_wrong)
        unmatched_rates.append(protocol_unmatched)
        delta_exact_rates.append(protocol_delta_exact)
    if any(value > 0.01 for value in unmatched_rates):
        return None
    if any(value < 0.0 for value in delta_exact_rates):
        return None
    candidate.update({
        "max_protocol_wrong_rate": max(wrong_rates),
        "min_protocol_delta_exact": min(delta_exact_rates),
        "mean_protocol_wrong_rate": float(np.mean(wrong_rates)),
        "mean_protocol_delta_exact": float(np.mean(delta_exact_rates)),
        "max_protocol_unmatched_rate": max(unmatched_rates),
        "mean_protocol_unmatched_rate": float(np.mean(unmatched_rates)),
    })
    return candidate


def select_thresholds(
    inner_oof_disagreement_table: pd.DataFrame,
    outer_train_full_population_table: pd.DataFrame,
) -> dict[str, float]:
    """Select one shared threshold pair using inner OOF and outer-train only."""
    renamed = inner_oof_disagreement_table.rename(columns={
        "inner_p_lineage": "p_lineage",
        "inner_p_both_wrong": "p_both_wrong",
    })
    merged = _merge_disagreement_probabilities(
        outer_train_full_population_table,
        renamed,
    )
    # Agreement outcomes are constant for every candidate.  Restrict the grid
    # loop to disagreement arrays and add the fixed agreement counts back into
    # each protocol metric; this is algebraically identical to full replay.
    disagreement = merged.p2a0_selected_query.to_numpy(int) != merged.lineage_child_query.to_numpy(int)
    protocol_labels = merged.protocol.astype(str).to_numpy()
    a = merged.p2a0_selected_query.to_numpy(int)
    lineage = merged.lineage_child_query.to_numpy(int)
    oracle = merged.oracle_query_index.to_numpy(int)
    preference = merged.p_lineage.to_numpy(float)
    ambiguity = merged.p_both_wrong.to_numpy(float)
    protocol_data = {}
    for protocol in PROTOCOLS:
        all_mask = protocol_labels == protocol
        if not all_mask.any():
            raise RuntimeError(f"threshold population lacks protocol: {protocol}")
        changing = all_mask & disagreement
        fixed = all_mask & ~disagreement
        protocol_data[protocol] = {
            "rows": int(all_mask.sum()),
            "a": a[changing],
            "lineage": lineage[changing],
            "oracle": oracle[changing],
            "preference": preference[changing],
            "ambiguity": ambiguity[changing],
            "fixed_exact": int((a[fixed] == oracle[fixed]).sum()),
            "fixed_wrong": int(((a[fixed] >= 0) & (a[fixed] != oracle[fixed])).sum()),
            "fixed_unmatched": int((a[fixed] == -1).sum()),
            "baseline_exact": int((a[all_mask] == oracle[all_mask]).sum()),
        }
    candidates = []
    for tau_defer in DEFER_THRESHOLDS:
        for tau_preference in PREFERENCE_THRESHOLDS:
            candidate: dict[str, float] = {
                "tau_preference": float(tau_preference),
                "tau_defer": float(tau_defer),
            }
            wrong_rates = []
            unmatched_rates = []
            delta_exact_rates = []
            for protocol in PROTOCOLS:
                data = protocol_data[protocol]
                selected = np.where(
                    data["ambiguity"] >= tau_defer,
                    -1,
                    np.where(
                        data["preference"] >= tau_preference,
                        data["lineage"],
                        data["a"],
                    ),
                )
                exact_count = int(data["fixed_exact"] + (selected == data["oracle"]).sum())
                wrong_count = int(
                    data["fixed_wrong"]
                    + ((selected >= 0) & (selected != data["oracle"])).sum()
                )
                unmatched_count = int(data["fixed_unmatched"] + (selected == -1).sum())
                rows = int(data["rows"])
                wrong_rate = wrong_count / rows
                unmatched_rate = unmatched_count / rows
                delta_exact = (exact_count - int(data["baseline_exact"])) / rows
                short = protocol.split("_")[0]
                candidate[f"inner_{short}_wrong"] = float(wrong_rate)
                candidate[f"inner_{short}_unmatched"] = float(unmatched_rate)
                candidate[f"inner_{short}_delta_exact"] = float(delta_exact)
                wrong_rates.append(wrong_rate)
                unmatched_rates.append(unmatched_rate)
                delta_exact_rates.append(delta_exact)
            if any(value > 0.01 for value in unmatched_rates):
                continue
            if any(value < 0.0 for value in delta_exact_rates):
                continue
            candidate.update({
                "max_protocol_wrong_rate": max(wrong_rates),
                "min_protocol_delta_exact": min(delta_exact_rates),
                "mean_protocol_wrong_rate": float(np.mean(wrong_rates)),
                "mean_protocol_delta_exact": float(np.mean(delta_exact_rates)),
                "max_protocol_unmatched_rate": max(unmatched_rates),
                "mean_protocol_unmatched_rate": float(np.mean(unmatched_rates)),
            })
            candidates.append(candidate)
    return select_lexicographic_candidate(candidates)


def inner_oof_predictions(
    outer_train_disagreement: pd.DataFrame,
    outer_train_scene_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Generate one prediction/head/row with 4-fold scene isolation."""
    folds = assign_scene_folds(
        outer_train_scene_manifest,
        n_splits=INNER_SPLITS,
        fold_column="inner_fold",
    )
    scene_to_fold = folds.set_index("scene_token").inner_fold
    row_folds = outer_train_disagreement.scene_token.astype(str).map(scene_to_fold)
    if row_folds.isna().any():
        raise RuntimeError("inner fold assignment does not cover disagreement rows")
    result = outer_train_disagreement.loc[:, KEY_COLUMNS].copy()
    result["inner_fold"] = row_folds.to_numpy(dtype=np.int64)
    result["inner_p_lineage"] = np.nan
    result["inner_p_both_wrong"] = np.nan
    for fold in range(INNER_SPLITS):
        validation = result.inner_fold.to_numpy(int) == fold
        train = ~validation
        train_scenes = set(outer_train_disagreement.loc[train, "scene_token"].astype(str))
        validation_scenes = set(outer_train_disagreement.loc[validation, "scene_token"].astype(str))
        if train_scenes & validation_scenes:
            raise RuntimeError("an inner-validation scene entered inner training")
        preference = fit_preference_head(outer_train_disagreement.loc[train])
        ambiguity = fit_ambiguity_head(outer_train_disagreement.loc[train])
        matrix = finite_model_matrix(outer_train_disagreement.loc[validation])
        result.loc[validation, "inner_p_lineage"] = preference.predict_proba(matrix)[:, 1]
        result.loc[validation, "inner_p_both_wrong"] = ambiguity.predict_proba(matrix)[:, 1]
    probabilities = result[["inner_p_lineage", "inner_p_both_wrong"]].to_numpy(float)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("inner OOF probabilities are incomplete")
    return result


def nested_outer_oof(
    full_population: pd.DataFrame,
    disagreement: pd.DataFrame,
    scene_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run preregistered 5x4 nested scene-grouped development evaluation."""
    outer_folds = assign_scene_folds(
        scene_manifest,
        n_splits=OUTER_SPLITS,
        fold_column="outer_fold",
    )
    scene_to_fold = outer_folds.set_index("scene_token").outer_fold
    full_folds = full_population.scene_token.astype(str).map(scene_to_fold)
    disagreement_folds = disagreement.scene_token.astype(str).map(scene_to_fold)
    if full_folds.isna().any() or disagreement_folds.isna().any():
        raise RuntimeError("outer folds do not cover all rows")
    outputs = []
    threshold_rows = []
    for outer_fold in range(OUTER_SPLITS):
        outer_train_full = full_population.loc[full_folds.to_numpy(int) != outer_fold].copy()
        outer_test_full = full_population.loc[full_folds.to_numpy(int) == outer_fold].copy()
        outer_train_disagreement = disagreement.loc[
            disagreement_folds.to_numpy(int) != outer_fold
        ].copy()
        outer_test_disagreement = disagreement.loc[
            disagreement_folds.to_numpy(int) == outer_fold
        ].copy()
        train_scenes = outer_folds.loc[
            outer_folds.outer_fold != outer_fold, ["scene_token"]
        ].reset_index(drop=True)
        inner = inner_oof_predictions(outer_train_disagreement, train_scenes)
        chosen = select_thresholds(inner, outer_train_full)
        chosen["outer_fold"] = int(outer_fold)
        threshold_rows.append(chosen)

        preference = fit_preference_head(outer_train_disagreement)
        ambiguity = fit_ambiguity_head(outer_train_disagreement)
        matrix = finite_model_matrix(outer_test_disagreement)
        prediction = outer_test_disagreement.loc[:, KEY_COLUMNS].copy()
        prediction["p_lineage"] = preference.predict_proba(matrix)[:, 1]
        prediction["p_both_wrong"] = ambiguity.predict_proba(matrix)[:, 1]
        decided = decide_full_population(
            outer_test_full,
            prediction,
            tau_preference=chosen["tau_preference"],
            tau_defer=chosen["tau_defer"],
        )
        decided["outer_fold"] = int(outer_fold)
        decided["tau_preference"] = float(chosen["tau_preference"])
        decided["tau_defer"] = float(chosen["tau_defer"])
        outputs.append(decided)
    full_oof = pd.concat(outputs, ignore_index=True)
    if len(full_oof) != len(full_population):
        raise RuntimeError("nested OOF did not reconstruct the full population")
    if full_oof.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError("nested OOF contains duplicate rows")
    disagreement_oof = full_oof.loc[
        full_oof.p2a0_selected_query != full_oof.lineage_child_query
    ].copy()
    return full_oof, disagreement_oof, pd.DataFrame(threshold_rows)


def paired_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    cluster_columns: Sequence[str],
    cluster_population: Sequence[Any] | None = None,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, int]:
    """Paired percentile bootstrap from already-frozen nested OOF rows."""
    if int(replicates) != BOOTSTRAP_REPLICATES or int(seed) != BOOTSTRAP_SEED:
        raise ValueError("bootstrap repetitions/seed are frozen at 5000/314159")
    columns = list(cluster_columns)
    if not columns or not set(columns) <= set(frame.columns):
        raise ValueError("bootstrap cluster columns are invalid")
    row_keys = list(frame.loc[:, columns].itertuples(index=False, name=None))
    if cluster_population is None:
        population = list(dict.fromkeys(row_keys))
    else:
        population = [value if isinstance(value, tuple) else (value,) for value in cluster_population]
    if not population or len(population) != len(set(population)):
        raise RuntimeError("bootstrap cluster population is empty or duplicated")
    index = {key: position for position, key in enumerate(population)}
    if any(key not in index for key in row_keys):
        raise RuntimeError("bootstrap rows are outside the cluster population")
    metric_names = (
        "delta_exact_rate",
        "delta_wrong_rate",
        "arbiter_exact_rate",
        "arbiter_wrong_rate",
        "arbiter_unmatched_rate",
    )
    metric_values = np.column_stack([
        frame.arbiter_exact.to_numpy(float) - frame.p2a0_exact.to_numpy(float),
        frame.arbiter_wrong.to_numpy(float) - frame.p2a0_wrong.to_numpy(float),
        frame.arbiter_exact.to_numpy(float),
        frame.arbiter_wrong.to_numpy(float),
        frame.arbiter_unmatched.to_numpy(float),
    ])
    cluster_sums = np.zeros((len(population), len(metric_names)), dtype=np.float64)
    cluster_counts = np.zeros(len(population), dtype=np.int64)
    for row, key in enumerate(row_keys):
        position = index[key]
        cluster_sums[position] += metric_values[row]
        cluster_counts[position] += 1
    rng = np.random.default_rng(int(seed))
    samples = np.empty((int(replicates), len(metric_names)), dtype=np.float64)
    valid = 0
    redraws = 0
    batch_size = 64
    while valid < int(replicates):
        count = min(batch_size, int(replicates) - valid)
        draws = rng.integers(0, len(population), size=(count, len(population)))
        denominators = cluster_counts[draws].sum(axis=1)
        numerators = cluster_sums[draws].sum(axis=1)
        good = denominators > 0
        accepted = int(good.sum())
        if accepted:
            samples[valid:valid + accepted] = numerators[good] / denominators[good, None]
            valid += accepted
        redraws += int((~good).sum())
    points = metric_values.mean(axis=0)
    rows = []
    for index_, metric in enumerate(metric_names):
        rows.append({
            "metric": metric,
            "point": float(points[index_]),
            "ci_low": float(np.percentile(samples[:, index_], 2.5)),
            "ci_high": float(np.percentile(samples[:, index_], 97.5)),
            "replicates": int(replicates),
            "seed": int(seed),
        })
    return pd.DataFrame(rows), redraws


def evaluate_final_gate(
    protocol_metrics: pd.DataFrame,
    scene_bootstrap: pd.DataFrame,
    instance_bootstrap: pd.DataFrame,
    discipline: Mapping[str, bool],
) -> tuple[pd.DataFrame, str]:
    required_false = (
        "protocol_used_as_feature",
        "protocol_specific_model",
        "protocol_specific_threshold",
        "model_family_search",
        "feature_search",
        "calibration_search",
        "outer_test_used_for_threshold_selection",
        "probe_val_read",
        "probe_test_read",
    )
    clean = all(discipline.get(name) is False for name in required_false)
    gate_rows = []
    for protocol in PROTOCOLS:
        metric_rows = protocol_metrics.loc[protocol_metrics.protocol.astype(str) == protocol]
        if len(metric_rows) != 1:
            raise RuntimeError(f"Gate requires one metric row for {protocol}")
        metric = metric_rows.iloc[0]
        scene = scene_bootstrap.loc[scene_bootstrap.protocol.astype(str) == protocol].set_index("metric")
        instance = instance_bootstrap.loc[instance_bootstrap.protocol.astype(str) == protocol].set_index("metric")
        gates = {
            "G1_wrong_match": bool(float(metric.arbiter_wrong_rate) <= 0.10),
            "G2_unmatched_budget": bool(float(metric.arbiter_unmatched_rate) <= 0.01),
            "G3_positive_exact_gain": bool(float(metric.delta_exact_rate) > 0.0),
            "G4_scene_exact_gain": bool(float(scene.loc["delta_exact_rate", "ci_low"]) > 0.0),
            "G5_instance_exact_gain": bool(float(instance.loc["delta_exact_rate", "ci_low"]) > 0.0),
            "G6_wrong_reduction": bool(float(metric.delta_wrong_rate) < 0.0),
            "G7_scene_wrong_reduction": bool(float(scene.loc["delta_wrong_rate", "ci_high"]) < 0.0),
            "G8_instance_wrong_reduction": bool(float(instance.loc["delta_wrong_rate", "ci_high"]) < 0.0),
            "G9_agreement_invariance": bool(int(metric.agreement_modified_count) == 0),
            "G10_leakage_search_discipline": bool(clean),
        }
        gate_rows.append({
            "protocol": protocol,
            **gates,
            "protocol_pass": bool(all(gates.values())),
        })
    gate = pd.DataFrame(gate_rows)
    passed = bool(len(gate) == len(PROTOCOLS) and gate.protocol_pass.astype(bool).all())
    status = (
        "GO_P2A2_R1_DISAGREEMENT_GATED_IDENTITY_ARBITER"
        if passed
        else "NO_GO_P2A2_R1_DISAGREEMENT_GATED_IDENTITY_ARBITER"
    )
    return gate, status


def export_head(model: Pipeline) -> dict[str, list[float]]:
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    return {
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "coef": classifier.coef_[0].astype(float).tolist(),
        "intercept": classifier.intercept_.astype(float).tolist(),
    }


@dataclass(frozen=True)
class FrozenIdentityArbiter:
    """Pure NumPy replay of the two frozen standardized logistic heads."""

    preference_model: Mapping[str, Sequence[float]]
    ambiguity_model: Mapping[str, Sequence[float]]
    tau_preference: float
    tau_defer: float

    @staticmethod
    def _probability(matrix: np.ndarray, model: Mapping[str, Sequence[float]]) -> np.ndarray:
        x = np.asarray(matrix, dtype=np.float64)
        mean = np.asarray(model["scaler_mean"], dtype=np.float64)
        scale = np.asarray(model["scaler_scale"], dtype=np.float64)
        coefficient = np.asarray(model["coef"], dtype=np.float64)
        intercept = np.asarray(model["intercept"], dtype=np.float64).reshape(-1)
        if x.ndim != 2 or x.shape[1] != len(mean):
            raise ValueError("frozen arbiter feature layout changed")
        if not (mean.shape == scale.shape == coefficient.shape) or intercept.shape != (1,):
            raise RuntimeError("frozen logistic artifact layout changed")
        score = ((x - mean) / scale) @ coefficient + intercept[0]
        output = np.empty_like(score)
        positive = score >= 0
        output[positive] = 1.0 / (1.0 + np.exp(-score[positive]))
        exponential = np.exp(score[~positive])
        output[~positive] = exponential / (1.0 + exponential)
        return output

    def predict(
        self,
        model_features: np.ndarray,
        p2a0_candidates: Sequence[int],
        lineage_candidates: Sequence[int],
    ) -> pd.DataFrame:
        class _FrozenHead:
            def __init__(self, owner: "FrozenIdentityArbiter", model: Mapping[str, Sequence[float]]):
                self.owner = owner
                self.model = model

            def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
                probability = self.owner._probability(matrix, self.model)
                return np.column_stack([1.0 - probability, probability])

        return runtime_arbiter(
            model_features,
            p2a0_candidates,
            lineage_candidates,
            _FrozenHead(self, self.preference_model),
            _FrozenHead(self, self.ambiguity_model),
            tau_preference=self.tau_preference,
            tau_defer=self.tau_defer,
        )
