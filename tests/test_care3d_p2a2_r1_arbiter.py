import inspect

import numpy as np
import pandas as pd
import pytest

import analysis.care3d_p2a2_r1_arbiter as arbiter
from analysis.care3d_p2a2_r1_arbiter import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DEFER_THRESHOLDS,
    INNER_SPLITS,
    MODEL_PARAMETERS,
    OUTER_SPLITS,
    PREFERENCE_THRESHOLDS,
    FrozenIdentityArbiter,
    _threshold_candidate_metrics,
    add_offline_outcomes,
    apply_probabilities,
    assign_scene_folds,
    evaluate_final_gate,
    export_head,
    inner_oof_predictions,
    make_binary_head,
    paired_cluster_bootstrap,
    protocol_point_metrics,
    repair_breakdown,
    runtime_arbiter,
    select_lexicographic_candidate,
    select_thresholds,
    threshold_sort_key,
)
from analysis.care3d_p2a2_r1_features import MODEL_FEATURE_COLUMNS
from analysis.care3d_p2a_association import PROTOCOLS


class NeverCalled:
    def predict_proba(self, _matrix):
        raise AssertionError("agreement bypass called a model")


class ConstantHead:
    def __init__(self, probability):
        self.probability = float(probability)

    def predict_proba(self, matrix):
        probability = np.full(len(matrix), self.probability)
        return np.column_stack([1.0 - probability, probability])


def test_agreement_bypass_never_calls_heads_and_never_defers():
    result = runtime_arbiter(
        np.zeros((2, 3)), [7, 9], [7, 9], NeverCalled(), NeverCalled(),
        tau_preference=0.5, tau_defer=0.5,
    )
    assert result.arbiter_selected_query.tolist() == [7, 9]
    assert result.arbiter_decision.tolist() == ["AGREEMENT", "AGREEMENT"]
    assert result.p_lineage.isna().all()
    assert result.p_both_wrong.isna().all()


def test_ambiguity_defer_has_priority_over_high_preference():
    result = runtime_arbiter(
        np.zeros((1, 2)), [4], [8], ConstantHead(0.99), ConstantHead(0.90),
        tau_preference=0.5, tau_defer=0.8,
    )
    assert result.arbiter_selected_query.tolist() == [-1]
    assert result.arbiter_decision.tolist() == ["DEFER"]


@pytest.mark.parametrize(
    "probability,expected_query,expected_decision",
    [(0.7, 8, "LINEAGE"), (0.69, 4, "P2A0")],
)
def test_preference_applies_only_when_not_deferred(
    probability, expected_query, expected_decision
):
    result = apply_probabilities(
        [4], [8], [probability], [0.1],
        tau_preference=0.7, tau_defer=0.8,
    )
    assert result.arbiter_selected_query.tolist() == [expected_query]
    assert result.arbiter_decision.tolist() == [expected_decision]


def test_model_family_and_threshold_grids_are_exactly_frozen():
    model = make_binary_head()
    classifier = model.named_steps["logisticregression"]
    assert isinstance(model.named_steps["standardscaler"], arbiter.StandardScaler)
    assert classifier.solver == "lbfgs"
    assert classifier.penalty == "l2"
    assert classifier.C == 1.0
    assert classifier.max_iter == 2000
    assert classifier.class_weight == "balanced"
    assert classifier.random_state == 314159
    assert MODEL_PARAMETERS["random_state"] == 314159
    assert PREFERENCE_THRESHOLDS == tuple(round(x / 100, 2) for x in range(5, 96)) + (1.01,)
    assert DEFER_THRESHOLDS == tuple(round(x / 100, 2) for x in range(50, 100)) + (1.01,)


def test_model_input_interfaces_have_no_label_or_protocol_argument():
    forbidden = {
        "protocol", "gt", "ground_truth", "oracle", "oracle_query",
        "clean_future", "future_clean",
    }
    for function in (
        arbiter.fit_preference_head,
        arbiter.fit_ambiguity_head,
        runtime_arbiter,
    ):
        assert set(inspect.signature(function).parameters).isdisjoint(forbidden)
    assert "protocol" not in MODEL_FEATURE_COLUMNS


def test_outer_group_assignment_covers_all_419_scenes_without_crossing():
    manifest = pd.DataFrame({"scene_token": [f"scene-{index:03d}" for index in range(419)]})
    folds = assign_scene_folds(manifest, n_splits=OUTER_SPLITS, fold_column="outer_fold")
    assert len(folds) == 419
    assert folds.scene_token.nunique() == 419
    assert set(folds.outer_fold) == set(range(5))
    rows = pd.DataFrame({
        "scene_token": np.repeat(manifest.scene_token.to_numpy(), 6),
        "protocol": list(PROTOCOLS) * (419 * 2),
    }).merge(folds, on="scene_token", validate="many_to_one")
    assert (rows.groupby("scene_token").outer_fold.nunique() == 1).all()


def _feature_rows(scene_count=8):
    rows = []
    outcomes = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    for scene in range(scene_count):
        for index, (p2a0, lineage, both) in enumerate(outcomes):
            row = {column: float((scene + index) % 5) for column in MODEL_FEATURE_COLUMNS}
            row.update({
                "scene_token": f"s{scene}",
                "instance_token": f"i{scene}-{index}",
                "anchor_frame_idx": 2,
                "target_frame_idx": 3,
                "protocol": PROTOCOLS[index],
                "p2a0_wins": p2a0,
                "lineage_wins": lineage,
                "both_wrong": both,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def test_inner_group_isolation_applies_to_both_model_fits(monkeypatch):
    rows = _feature_rows()
    manifest = pd.DataFrame({"scene_token": sorted(rows.scene_token.unique())})
    fitted_scene_sets = []

    def fake_fit(frame):
        fitted_scene_sets.append(set(frame.scene_token.astype(str)))
        return ConstantHead(0.4)

    monkeypatch.setattr(arbiter, "fit_preference_head", fake_fit)
    monkeypatch.setattr(arbiter, "fit_ambiguity_head", fake_fit)
    output = inner_oof_predictions(rows, manifest)
    assert len(output) == len(rows)
    assert np.isfinite(output[["inner_p_lineage", "inner_p_both_wrong"]]).all().all()
    assert len(fitted_scene_sets) == INNER_SPLITS * 2
    for fold in range(INNER_SPLITS):
        validation_scenes = set(output.loc[output.inner_fold == fold, "scene_token"])
        assert fitted_scene_sets[fold * 2].isdisjoint(validation_scenes)
        assert fitted_scene_sets[fold * 2 + 1].isdisjoint(validation_scenes)


def _population_with_probabilities(
    *, crash_defer=False, dark_bad_exact=False
):
    full_rows = []
    prediction_rows = []
    for protocol_index, protocol in enumerate(PROTOCOLS):
        key_base = {
            "scene_token": f"s{protocol_index}",
            "anchor_frame_idx": 2,
            "target_frame_idx": 3,
            "protocol": protocol,
        }
        full_rows.extend([
            {
                **key_base, "instance_token": "agreement",
                "p2a0_selected_query": 1, "lineage_child_query": 1,
                "oracle_query_index": 1,
            },
            {
                **key_base, "instance_token": "disagreement",
                "p2a0_selected_query": 2, "lineage_child_query": 3,
                "oracle_query_index": 2 if (dark_bad_exact and protocol == "dark_back") else 3,
            },
        ])
        prediction_rows.append({
            **key_base, "instance_token": "disagreement",
            "inner_p_lineage": 0.9,
            "inner_p_both_wrong": 0.9 if (crash_defer and protocol == "crash_back") else 0.1,
        })
    return pd.DataFrame(full_rows), pd.DataFrame(prediction_rows)


def test_threshold_selector_signature_cannot_receive_outer_test_data():
    assert tuple(inspect.signature(select_thresholds).parameters) == (
        "inner_oof_disagreement_table", "outer_train_full_population_table"
    )


def test_baseline_sentinel_thresholds_exactly_replay_p2a0():
    result = apply_probabilities(
        [1, 2], [1, 3], [np.nan, 0.99], [np.nan, 0.99],
        tau_preference=1.01, tau_defer=1.01,
    )
    assert result.arbiter_selected_query.tolist() == [1, 2]
    assert result.arbiter_decision.tolist() == ["AGREEMENT", "P2A0"]


def test_unmatched_budget_and_nonnegative_exact_are_hard_feasibility_rules():
    full, predictions = _population_with_probabilities(crash_defer=True)
    merged = arbiter._merge_disagreement_probabilities(
        full,
        predictions.rename(columns={
            "inner_p_lineage": "p_lineage", "inner_p_both_wrong": "p_both_wrong"
        }),
    )
    assert _threshold_candidate_metrics(merged, 0.5, 0.5) is None
    full, predictions = _population_with_probabilities(dark_bad_exact=True)
    merged = arbiter._merge_disagreement_probabilities(
        full,
        predictions.rename(columns={
            "inner_p_lineage": "p_lineage", "inner_p_both_wrong": "p_both_wrong"
        }),
    )
    assert _threshold_candidate_metrics(merged, 0.5, 1.01) is None


def test_threshold_selection_returns_one_shared_pair_and_is_deterministic():
    full, predictions = _population_with_probabilities()
    first = select_thresholds(predictions, full)
    second = select_thresholds(predictions, full)
    assert first == second
    assert "tau_preference" in first and "tau_defer" in first
    assert not any(name.startswith(("blur_tau", "crash_tau", "dark_tau")) for name in first)


def _candidate(**overrides):
    value = {
        "max_protocol_wrong_rate": 0.2,
        "min_protocol_delta_exact": 0.1,
        "mean_protocol_wrong_rate": 0.15,
        "mean_protocol_delta_exact": 0.12,
        "max_protocol_unmatched_rate": 0.01,
        "mean_protocol_unmatched_rate": 0.005,
        "tau_defer": 0.8,
        "tau_preference": 0.4,
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize(
    "better,worse",
    [
        ({"max_protocol_wrong_rate": 0.19}, {}),
        ({"min_protocol_delta_exact": 0.11}, {}),
        ({"mean_protocol_wrong_rate": 0.14}, {}),
        ({"mean_protocol_delta_exact": 0.13}, {}),
        ({"max_protocol_unmatched_rate": 0.009}, {}),
        ({"mean_protocol_unmatched_rate": 0.004}, {}),
        ({"tau_defer": 0.81}, {}),
        ({"tau_preference": 0.45}, {}),
        ({"tau_preference": 0.6}, {"tau_preference": 0.4}),
    ],
)
def test_lexicographic_objective_each_level(better, worse):
    first = _candidate(**better)
    second = _candidate(**worse)
    # The final case is equidistant from 0.5 and must choose larger tau.
    assert threshold_sort_key(first) < threshold_sort_key(second)
    assert select_lexicographic_candidate([second, first]) == first


def _accounting_frame():
    frame = pd.DataFrame({
        "p2a0_selected_query": [1, 2, 3, 4, 5, 6],
        "lineage_child_query": [1, 20, 30, 40, 50, 60],
        "oracle_query_index": [1, 20, 30, 4, 50, 6],
        "arbiter_selected_query": [1, 20, 31, 41, -1, -1],
        "arbiter_decision": ["AGREEMENT", "LINEAGE", "LINEAGE", "LINEAGE", "DEFER", "DEFER"],
    })
    return add_offline_outcomes(frame)


def test_full_population_and_repair_accounting_are_exhaustive():
    frame = _accounting_frame()
    assert ((frame.arbiter_exact + frame.arbiter_wrong + frame.arbiter_unmatched) == 1).all()
    metrics = protocol_point_metrics(frame)
    assert metrics["agreement_rows"] + metrics["disagreement_rows"] == len(frame)
    assert metrics["wrong_repaired_count"] == 1
    assert metrics["correct_broken_to_wrong_count"] == 1
    assert metrics["correct_broken_to_unmatched_count"] == 1
    assert metrics["wrong_to_unmatched_count"] == 1
    assert metrics["wrong_remaining_wrong_count"] == 1
    assert repair_breakdown(frame)["wrong_repaired_count"] == 1


def test_bootstrap_is_paired_and_deterministic_with_zero_row_scene_population():
    frame = _accounting_frame().assign(
        scene_token=["s1", "s1", "s2", "s2", "s3", "s3"]
    )
    first, redraws_first = paired_cluster_bootstrap(
        frame,
        cluster_columns=("scene_token",),
        cluster_population=("s0", "s1", "s2", "s3"),
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    second, redraws_second = paired_cluster_bootstrap(
        frame,
        cluster_columns=("scene_token",),
        cluster_population=("s0", "s1", "s2", "s3"),
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    pd.testing.assert_frame_equal(first, second)
    assert redraws_first == redraws_second
    point = first.set_index("metric").loc["delta_exact_rate", "point"]
    expected = (frame.arbiter_exact - frame.p2a0_exact).mean()
    assert point == pytest.approx(expected)


def _passing_gate_inputs():
    metrics = pd.DataFrame([
        {
            "protocol": protocol,
            "arbiter_wrong_rate": 0.09,
            "arbiter_unmatched_rate": 0.005,
            "delta_exact_rate": 0.01,
            "delta_wrong_rate": -0.01,
            "agreement_modified_count": 0,
        }
        for protocol in PROTOCOLS
    ])
    summaries = []
    for protocol in PROTOCOLS:
        summaries.extend([
            {"protocol": protocol, "metric": "delta_exact_rate", "ci_low": 0.001, "ci_high": 0.02},
            {"protocol": protocol, "metric": "delta_wrong_rate", "ci_low": -0.02, "ci_high": -0.001},
        ])
    discipline = {
        name: False
        for name in (
            "protocol_used_as_feature", "protocol_specific_model",
            "protocol_specific_threshold", "model_family_search", "feature_search",
            "calibration_search", "outer_test_used_for_threshold_selection",
            "probe_val_read", "probe_test_read",
        )
    }
    return metrics, pd.DataFrame(summaries), discipline


def test_gate_requires_all_three_protocols_including_crash():
    metrics, summary, discipline = _passing_gate_inputs()
    gate, status = evaluate_final_gate(metrics, summary, summary, discipline)
    assert gate.protocol_pass.all()
    assert status == "GO_P2A2_R1_DISAGREEMENT_GATED_IDENTITY_ARBITER"
    metrics.loc[metrics.protocol == "crash_back", "arbiter_wrong_rate"] = 0.1001
    gate, status = evaluate_final_gate(metrics, summary, summary, discipline)
    assert not gate.loc[gate.protocol == "crash_back", "protocol_pass"].item()
    assert status == "NO_GO_P2A2_R1_DISAGREEMENT_GATED_IDENTITY_ARBITER"


def test_gate_unmatched_and_zero_ci_boundaries_fail_strictly():
    metrics, scene, discipline = _passing_gate_inputs()
    metrics.loc[metrics.protocol == "blur_back", "arbiter_unmatched_rate"] = 0.0101
    gate, status = evaluate_final_gate(metrics, scene, scene, discipline)
    assert status.startswith("NO_GO")
    metrics.loc[metrics.protocol == "blur_back", "arbiter_unmatched_rate"] = 0.01
    scene.loc[
        (scene.protocol == "blur_back") & (scene.metric == "delta_exact_rate"), "ci_low"
    ] = 0.0
    instance = scene.copy()
    instance.loc[
        (instance.protocol == "blur_back") & (instance.metric == "delta_exact_rate"), "ci_low"
    ] = 0.001
    instance.loc[
        (instance.protocol == "dark_back") & (instance.metric == "delta_wrong_rate"), "ci_high"
    ] = 0.0
    gate, status = evaluate_final_gate(metrics, scene, instance, discipline)
    assert not gate.loc[gate.protocol == "blur_back", "G4_scene_exact_gain"].item()
    assert not gate.loc[gate.protocol == "dark_back", "G8_instance_wrong_reduction"].item()
    assert status.startswith("NO_GO")


def test_frozen_numpy_replay_matches_both_sklearn_heads_to_1e_12():
    rng = np.random.default_rng(19)
    training = rng.normal(size=(80, 7))
    labels_a = (training[:, 0] + 0.2 * training[:, 1] > 0).astype(int)
    labels_b = (training[:, 2] - 0.3 * training[:, 3] > 0).astype(int)
    preference = make_binary_head().fit(training, labels_a)
    ambiguity = make_binary_head().fit(training, labels_b)
    frozen = FrozenIdentityArbiter(
        export_head(preference), export_head(ambiguity), 0.5, 0.8
    )
    test = rng.normal(size=(31, 7))
    preference_difference = np.max(np.abs(
        frozen._probability(test, frozen.preference_model)
        - preference.predict_proba(test)[:, 1]
    ))
    ambiguity_difference = np.max(np.abs(
        frozen._probability(test, frozen.ambiguity_model)
        - ambiguity.predict_proba(test)[:, 1]
    ))
    assert preference_difference <= 1e-12, preference_difference
    assert ambiguity_difference <= 1e-12, ambiguity_difference
