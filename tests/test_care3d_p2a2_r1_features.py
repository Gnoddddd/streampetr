import inspect

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import roc_auc_score

from analysis.care3d_p2a2_r1_features import (
    EVIDENCE_FEATURE_COLUMNS,
    finite_model_matrix,
    offline_disagreement_labels,
    relative_candidate_features,
)
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
    cost = torch.full(shape, float("inf"))
    # Row 0: A=1, L=2; query 3 determines row margin for A.
    cost[0, 1], cost[0, 2], cost[0, 3] = 0.10, 0.30, 0.20
    cost[1, 1], cost[1, 2], cost[1, 4] = 0.20, 0.10, 0.15
    cost[2, 1], cost[2, 2], cost[2, 5] = 0.40, 0.50, 0.05
    components["geometry"][0, 1], components["geometry"][0, 2] = 0.1, 0.4
    components["embedding"][0, 1], components["embedding"][0, 2] = 0.2, 0.3
    components["class"][0, 1], components["class"][0, 2] = 0.25, 0.5
    components["distance_m"][0, 1], components["distance_m"][0, 2] = 1.2, 4.8
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
    assert result["A_total_cost"][0] == pytest.approx(0.1)
    assert result["L_total_cost"][0] == pytest.approx(0.3)
    assert result["delta_total_cost"][0] == pytest.approx(0.2)
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
    assert result["A_row_margin"][0] == pytest.approx(0.10)
    assert result["L_row_margin"][0] == pytest.approx(-0.20)
    assert result["delta_row_margin"][0] == pytest.approx(-0.30)


def test_column_ownership_margin_sign_encodes_mutuality():
    components, cost, logits = fixture_inputs()
    result = relative_candidate_features(
        components, cost, logits, [1, 0, 2], [650, 20, 30],
        [1, 4, 5], [2, 4, 5], [10, 20, 30], target_frame_idx=3,
    )
    assert result["A_mutual"].tolist() == [1]
    assert result["L_mutual"].tolist() == [0]
    assert result["A_column_margin"][0] == pytest.approx(0.10)
    assert result["L_column_margin"][0] == pytest.approx(-0.20)
    assert result["delta_column_margin"][0] == pytest.approx(-0.30)


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
    assert "protocol" not in EVIDENCE_FEATURE_COLUMNS


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


def test_ineligible_candidate_is_not_given_a_finite_rank_or_margin():
    components, cost, logits = fixture_inputs()
    cost[0, 2] = float("inf")
    result = relative_candidate_features(
        components, cost, logits, [1, 0, 2], [650, 20, 30],
        [1, 4, 5], [2, 4, 5], [10, 20, 30], target_frame_idx=3,
    )
    assert result["L_row_rank"].tolist() == [0]
    assert np.isnan(result["L_row_margin"][0])
    assert result["L_mutual"].tolist() == [0]
    assert np.isnan(result["L_column_margin"][0])


def test_fixed_model_encoding_handles_only_no_column_competitor_case():
    frame = pd.DataFrame({column: [0.0] for column in EVIDENCE_FEATURE_COLUMNS})
    frame["A_column_margin"] = [float("inf")]
    frame["L_column_margin"] = [float("inf")]
    frame["delta_column_margin"] = [float("nan")]
    matrix = finite_model_matrix(frame)
    columns = {name: index for index, name in enumerate(EVIDENCE_FEATURE_COLUMNS)}
    assert matrix[0, columns["A_column_margin"]] == 1.0
    assert matrix[0, columns["L_column_margin"]] == 1.0
    assert matrix[0, columns["delta_column_margin"]] == 0.0
    frame["A_total_cost"] = [float("inf")]
    with pytest.raises(RuntimeError, match="outside column margins"):
        finite_model_matrix(frame)


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
