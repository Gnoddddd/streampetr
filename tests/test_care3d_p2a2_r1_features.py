import inspect

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import roc_auc_score

from analysis.care3d_p2a2_r1_features import (
    EVIDENCE_FEATURE_COLUMNS,
    HARD_AUDIT_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    finite_model_matrix,
    offline_disagreement_labels,
    relative_candidate_features,
    ungated_weighted_cost,
)
from analysis.care3d_p2a2_lineage import FROZEN_P2A0_CONFIG
from analysis.care3d_p2a_association import hungarian_with_unmatched, weighted_cost
from scripts.analyze_care3d_p2a2_r1_features import (
    CLASSES,
    binary_oof,
    fold_ids,
    multiclass_oof,
)
from scripts.export_care3d_p2a2_r1_features import parse_args, parse_r1_split


def fixture_inputs():
    shape = (3, 900)
    components = {
        "geometry": torch.full(shape, 0.8),
        "embedding": torch.full(shape, 0.6),
        "class": torch.full(shape, 0.5),
        "distance_m": torch.full(shape, 9.6),
        "geometry_allowed": torch.ones(shape, dtype=torch.bool),
    }
    # Row 0: A=1, L=2; query 3 determines row margin for A.  Every
    # geometry-eligible frozen value is produced by the frozen weighted cost.
    for row, query, value in (
        (0, 3, 0.25),
        (1, 1, 0.20), (1, 2, 0.10), (1, 4, 0.15),
        (2, 1, 0.40), (2, 2, 0.50), (2, 5, 0.05),
    ):
        components["geometry"][row, query] = value
        components["embedding"][row, query] = value
        components["class"][row, query] = value
    components["geometry"][0, 1], components["geometry"][0, 2] = 0.1, 0.4
    components["embedding"][0, 1], components["embedding"][0, 2] = 0.2, 0.3
    components["class"][0, 1], components["class"][0, 2] = 0.25, 0.5
    components["distance_m"][0, 1], components["distance_m"][0, 2] = 1.2, 4.8
    cost = weighted_cost(components, FROZEN_P2A0_CONFIG)
    logits = torch.full((900, 3), -4.0)
    logits[1] = torch.tensor([0.0, 2.0, -1.0])
    logits[2] = torch.tensor([3.0, 1.0, -2.0])
    return components, cost, logits


def test_relative_candidate_values_and_deltas_are_exact():
    components, cost, logits = fixture_inputs()
    result = relative_candidate_features(
        components, cost, logits,
        torch.tensor([1, 0, 2]),
        [650, 20, 30],
        [1, 4, 5],
        [2, 4, 5],
        [10, 20, 30],
        target_frame_idx=6,
    )
    # Only row 0 disagrees.
    assert result["source_row_index"].tolist() == [0]
    assert result["A_geometry"][0] == pytest.approx(0.1)
    assert result["L_geometry"][0] == pytest.approx(0.4)
    assert result["delta_geometry"][0] == pytest.approx(0.3)
    assert result["A_total_cost"][0] == pytest.approx(0.17)
    assert result["L_total_cost"][0] == pytest.approx(0.38)
    assert result["delta_total_cost"][0] == pytest.approx(0.21)
    assert result["A_ungated_total_cost"][0] == result["A_total_cost"][0]
    assert result["L_ungated_total_cost"][0] == result["L_total_cost"][0]
    assert result["A_anchor_class_probability"][0] == pytest.approx(0.75)
    assert result["L_anchor_class_probability"][0] == pytest.approx(0.5)
    assert result["delta_anchor_class_probability"][0] == pytest.approx(-0.25)
    assert result["A_predicted_class"].tolist() == [1]
    assert result["L_predicted_class"].tolist() == [0]
    assert result["A_class_matches_anchor"].tolist() == [1]
    assert result["L_class_matches_anchor"].tolist() == [0]


def test_row_local_rank_and_margin_ignore_ineligible_queries():
    components, cost, logits = fixture_inputs()
    result = relative_candidate_features(
        components, cost, logits, [1, 0, 2], [650, 20, 30],
        [1, 4, 5], [2, 4, 5], [10, 20, 30], target_frame_idx=3,
    )
    assert result["A_row_rank"].tolist() == [1]
    assert result["L_row_rank"].tolist() == [3]
    assert result["delta_row_rank"].tolist() == [2]
    assert result["A_row_margin"][0] == pytest.approx(0.08)
    assert result["L_row_margin"][0] == pytest.approx(-0.21)
    assert result["delta_row_margin"][0] == pytest.approx(-0.29)


def test_column_ownership_margin_sign_encodes_mutuality():
    components, cost, logits = fixture_inputs()
    result = relative_candidate_features(
        components, cost, logits, [1, 0, 2], [650, 20, 30],
        [1, 4, 5], [2, 4, 5], [10, 20, 30], target_frame_idx=3,
    )
    assert result["A_mutual"].tolist() == [1]
    assert result["L_mutual"].tolist() == [0]
    assert result["A_column_margin"][0] == pytest.approx(0.03)
    assert result["L_column_margin"][0] == pytest.approx(-0.28)
    assert result["delta_column_margin"][0] == pytest.approx(-0.31)


def test_only_disagreement_rows_are_exported():
    components, cost, logits = fixture_inputs()
    result = relative_candidate_features(
        components, cost, logits, [1, 0, 2], [650, 20, 30],
        [1, 4, 5], [2, 4, 5], [10, 20, 30], target_frame_idx=3,
    )
    assert len(result["source_row_index"]) == 1
    assert result["p2a0_selected_query"][0] != result["lineage_child_query"][0]


def test_context_normalization_is_frozen_and_protocol_free():
    components, cost, logits = fixture_inputs()
    result = relative_candidate_features(
        components, cost, logits, [1, 0, 2], [650, 20, 30],
        [1, 4, 5], [2, 4, 5], [255, 20, 30], target_frame_idx=12,
    )
    assert result["anchor_is_propagated"].tolist() == [1]
    assert result["p2a0_is_propagated"].tolist() == [0]
    assert result["lineage_position_norm"].tolist() == [1.0]
    assert result["target_frame_norm"].tolist() == [1.0]
    assert "protocol" not in MODEL_FEATURE_COLUMNS


def test_raw_predicted_classes_are_exported_but_not_continuous_model_features():
    assert "A_predicted_class" in EVIDENCE_FEATURE_COLUMNS
    assert "L_predicted_class" in EVIDENCE_FEATURE_COLUMNS
    assert "A_predicted_class" not in MODEL_FEATURE_COLUMNS
    assert "L_predicted_class" not in MODEL_FEATURE_COLUMNS
    assert "A_class_matches_anchor" in MODEL_FEATURE_COLUMNS
    assert "L_class_matches_anchor" in MODEL_FEATURE_COLUMNS
    assert not (HARD_AUDIT_FEATURE_COLUMNS & set(MODEL_FEATURE_COLUMNS))
    assert "A_geometry_eligible" in MODEL_FEATURE_COLUMNS
    assert "L_geometry_eligible" in MODEL_FEATURE_COLUMNS
    assert "A_ungated_total_cost" in MODEL_FEATURE_COLUMNS
    assert "L_ungated_mutual" in MODEL_FEATURE_COLUMNS


def test_feature_computation_has_no_gt_oracle_clean_future_or_protocol_argument():
    parameters = set(inspect.signature(relative_candidate_features).parameters)
    assert parameters.isdisjoint({
        "gt", "ground_truth", "oracle", "oracle_queries",
        "clean_future", "clean_output", "protocol",
    })
    assert "oracle_queries" in inspect.signature(offline_disagreement_labels).parameters


def test_offline_labels_are_mutually_exclusive_and_exhaustive():
    labels = offline_disagreement_labels(
        [1, 4, 7], [2, 5, 8], [1, 5, 9]
    )
    assert labels["outcome_class"].tolist() == [
        "P2A0_WINS", "LINEAGE_WINS", "BOTH_WRONG"
    ]
    total = labels["p2a0_wins"] + labels["lineage_wins"] + labels["both_wrong"]
    assert total.tolist() == [1, 1, 1]
    with pytest.raises(RuntimeError):
        offline_disagreement_labels([1], [1], [1])


def test_feature_extraction_does_not_mutate_frozen_tensors():
    components, cost, logits = fixture_inputs()
    before_cost = cost.clone()
    before_components = {key: value.clone() for key, value in components.items()}
    relative_candidate_features(
        components, cost, logits, [1, 0, 2], [650, 20, 30],
        [1, 4, 5], [2, 4, 5], [10, 20, 30], target_frame_idx=3,
    )
    assert torch.equal(cost, before_cost)
    assert all(torch.equal(value, before_components[key]) for key, value in components.items())


def test_geometry_ineligible_lineage_has_finite_ungated_evidence_and_model_matrix():
    components, cost, logits = fixture_inputs()
    components["geometry_allowed"][0, 2] = False
    components["distance_m"][0, 2] = 13.0
    cost = weighted_cost(components, FROZEN_P2A0_CONFIG)
    assignment_before = hungarian_with_unmatched(cost, FROZEN_P2A0_CONFIG.max_cost)
    selected = assignment_before["selected_query"]
    lineage = selected.copy()
    lineage[0] = 2
    result = relative_candidate_features(
        components, cost, logits, [1, 0, 2], [650, 20, 30],
        selected, lineage, [10, 20, 30], target_frame_idx=3,
    )
    assignment_after = hungarian_with_unmatched(cost, FROZEN_P2A0_CONFIG.max_cost)
    assert np.array_equal(
        assignment_before["selected_query"], assignment_after["selected_query"]
    )
    assert np.array_equal(
        assignment_before["selected_cost"], assignment_after["selected_cost"]
    )
    assert result["A_geometry_eligible"].tolist() == [1]
    assert result["L_geometry_eligible"].tolist() == [0]
    assert np.isfinite(result["A_total_cost"][0])
    assert np.isposinf(result["L_total_cost"][0])
    assert result["L_row_rank"].tolist() == [0]
    assert np.isnan(result["L_row_margin"][0])
    assert result["L_mutual"].tolist() == [0]
    assert np.isnan(result["L_column_margin"][0])
    for name in (
        "A_ungated_total_cost", "L_ungated_total_cost", "delta_ungated_total_cost",
        "A_ungated_row_rank", "L_ungated_row_rank", "delta_ungated_row_rank",
        "A_ungated_row_margin", "L_ungated_row_margin", "delta_ungated_row_margin",
        "A_ungated_column_margin", "L_ungated_column_margin",
        "delta_ungated_column_margin", "A_ungated_mutual", "L_ungated_mutual",
    ):
        assert np.isfinite(result[name]).all(), name
    assert np.isfinite(finite_model_matrix(pd.DataFrame(result))).all()


def test_ungated_cost_exactly_matches_frozen_cost_inside_geometry_gate():
    components, _, _ = fixture_inputs()
    components["geometry_allowed"][0, 2] = False
    frozen = weighted_cost(components, FROZEN_P2A0_CONFIG)
    soft = ungated_weighted_cost(components, frozen)
    allowed = components["geometry_allowed"]
    assert torch.equal(soft[allowed], frozen[allowed])
    assert torch.isfinite(soft).all()
    assert torch.isposinf(frozen[~allowed]).all()


def test_geometry_ineligible_p2a0_candidate_is_rejected():
    components, _, logits = fixture_inputs()
    components["geometry_allowed"][0, 1] = False
    cost = weighted_cost(components, FROZEN_P2A0_CONFIG)
    with pytest.raises(RuntimeError, match="P2-A0 candidate is geometry-ineligible"):
        relative_candidate_features(
            components, cost, logits, [1, 0, 2], [650, 20, 30],
            [1, 4, 5], [2, 4, 5], [10, 20, 30], target_frame_idx=3,
        )


@pytest.mark.parametrize(
    "a_margin,l_margin,raw_delta,expected",
    [
        (float("inf"), float("inf"), float("nan"), (1.0, 1.0, 0.0)),
        (float("inf"), 0.2, float("-inf"), (1.0, 0.2, -0.8)),
        (0.2, float("inf"), float("inf"), (0.2, 1.0, 0.8)),
    ],
)
def test_fixed_model_recomputes_encoded_column_margin_delta(
    a_margin, l_margin, raw_delta, expected
):
    frame = pd.DataFrame({column: [0.0] for column in EVIDENCE_FEATURE_COLUMNS})
    frame["A_ungated_column_margin"] = [a_margin]
    frame["L_ungated_column_margin"] = [l_margin]
    frame["delta_ungated_column_margin"] = [raw_delta]
    matrix = finite_model_matrix(frame)
    columns = {name: index for index, name in enumerate(MODEL_FEATURE_COLUMNS)}
    observed = tuple(
        matrix[0, columns[name]]
        for name in (
            "A_ungated_column_margin", "L_ungated_column_margin",
            "delta_ungated_column_margin",
        )
    )
    assert observed == pytest.approx(expected)


@pytest.mark.parametrize(
    "name", ["A_ungated_column_margin", "L_ungated_column_margin"]
)
@pytest.mark.parametrize("invalid", [float("nan"), float("-inf")])
def test_fixed_model_rejects_invalid_candidate_column_margins(name, invalid):
    frame = pd.DataFrame({column: [0.0] for column in EVIDENCE_FEATURE_COLUMNS})
    frame[name] = [invalid]
    with pytest.raises(RuntimeError, match="invalid no-competitor encoding input"):
        finite_model_matrix(frame)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_fixed_model_rejects_nonfinite_other_model_features(invalid):
    frame = pd.DataFrame({column: [0.0] for column in EVIDENCE_FEATURE_COLUMNS})
    frame["A_ungated_total_cost"] = [invalid]
    with pytest.raises(RuntimeError, match="outside column margins"):
        finite_model_matrix(frame)


def test_fixed_model_ignores_nonfinite_hard_gate_audit_features():
    frame = pd.DataFrame({column: [0.0] for column in EVIDENCE_FEATURE_COLUMNS})
    frame["L_total_cost"] = [float("inf")]
    frame["L_row_margin"] = [float("nan")]
    frame["L_column_margin"] = [float("nan")]
    assert np.isfinite(finite_model_matrix(frame)).all()


def test_cli_rejects_probe_val_and_probe_test():
    assert parse_r1_split("probe_train") == "probe_train"
    with pytest.raises(Exception):
        parse_r1_split("probe_val")
    with pytest.raises(Exception):
        parse_r1_split("probe_test")
    assert parse_args(["--split", "probe_train"]).split == "probe_train"
    with pytest.raises(SystemExit):
        parse_args(["--split", "probe_val"])
    with pytest.raises(SystemExit):
        parse_args(["--split", "probe_test"])


def test_groupkfold_keeps_all_protocol_rows_of_a_scene_together():
    rows = pd.DataFrame({
        "scene_token": [f"s{scene}" for scene in range(10) for _ in range(3)],
        "protocol": [protocol for _ in range(10) for protocol in ("b", "c", "d")],
    })
    first = fold_ids(rows)
    second = fold_ids(rows)
    assert np.array_equal(first, second)
    grouped = pd.DataFrame({"scene": rows.scene_token, "fold": first}).groupby("scene")
    assert (grouped.fold.nunique() == 1).all()
    assert set(first.tolist()) == set(range(5))


def test_fixed_binary_and_multinomial_logistic_oof_are_executable():
    rng = np.random.default_rng(17)
    count = 45
    rows = pd.DataFrame({
        column: rng.normal(size=count) for column in EVIDENCE_FEATURE_COLUMNS
    })
    rows["scene_token"] = [f"s{index // 3}" for index in range(count)]
    rows["binary"] = np.arange(count) % 2
    classes = np.asarray(["P2A0_WINS", "LINEAGE_WINS", "BOTH_WRONG"])
    rows["outcome_class"] = classes[np.arange(count) % 3]
    folds = fold_ids(rows)
    first = binary_oof(rows, "binary", folds)
    second = binary_oof(rows, "binary", folds)
    assert np.allclose(first, second, rtol=0.0, atol=0.0)
    assert ((0.0 <= first) & (first <= 1.0)).all()
    probabilities, predictions = multiclass_oof(rows, folds)
    assert probabilities.shape == (count, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert set(predictions) <= set(classes)
    assert np.isfinite(roc_auc_score(
        rows.outcome_class.astype(str), probabilities, labels=list(CLASSES),
        multi_class="ovr", average="macro",
    ))
