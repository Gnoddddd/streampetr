import numpy as np
import pytest

from analysis.fault_boundary_root_cause import (
    auroc,
    candidate_pool_statistics,
    count_matched_clean_max,
    projected_box_visibility,
    regression_cost,
    rescue_category,
    spearman,
)


def test_candidate_pool_uses_flattened_boundary_and_gt_near_best():
    probabilities = np.asarray([
        [0.90, 0.10],
        [0.80, 0.70],
        [0.60, 0.50],
    ])
    logits = np.log(probabilities / (1.0 - probabilities))
    boxes = np.asarray([[0.5, 0, 0], [1.0, 0, 0], [3.0, 0, 0]])
    result = candidate_pool_statistics(logits, boxes, [0, 0, 0], 1, topk=3)
    assert result["queries"].tolist() == [1, 0]
    assert result["scores"].tolist() == pytest.approx([0.70, 0.10])
    assert result["s_k"] == pytest.approx(0.70)
    assert result["rank"] == 3
    assert result["margin"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "target,boundary,expected",
    [(1, -1, "target-driven"), (-1, 1, "competitor-driven"),
     (1, 1, "mixed"), (0, 0, "neither")],
)
def test_rescue_category_is_strict(target, boundary, expected):
    assert rescue_category(target, boundary) == expected


def test_count_matched_counterfactual_is_reproducible_and_exact_at_full_count():
    left = count_matched_clean_max([0.1, 0.2, 0.9], 2, seed=7, repeats=200)
    right = count_matched_clean_max([0.1, 0.2, 0.9], 2, seed=7, repeats=200)
    assert left == right
    assert 0.2 <= left["expected_max"] <= 0.9
    exact = count_matched_clean_max([0.1, 0.9], 2, seed=99)
    assert exact["expected_max"] == pytest.approx(0.9)
    assert exact["effective_repeats"] == 1


def test_auroc_and_spearman_handle_ties():
    assert auroc([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert auroc([1, 1, 0, 0], [0, 0, 1, 1]) == pytest.approx(0.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_projected_box_visibility_uses_clipped_positive_area():
    corners = np.asarray([
        [1, 1, 2], [3, 1, 2], [1, 3, 2], [3, 3, 2],
        [1, 1, 4], [3, 1, 4], [1, 3, 4], [3, 3, 4],
    ])
    matrices = np.stack([np.eye(4), -np.eye(4)])
    result = projected_box_visibility(corners, matrices, [10, 10])
    assert result["visible"].tolist() == [True, False]
    assert result["area_fraction"][0] > 0


def test_decomposition_identity_with_independent_candidate_selection():
    clean = {"s_pos": 0.8, "s_k": 0.3, "margin": 0.5}
    fault = {"s_pos": 0.45, "s_k": 0.35, "margin": 0.10}
    delta_margin = fault["margin"] - clean["margin"]
    decomposed = ((fault["s_pos"] - clean["s_pos"])
                  - (fault["s_k"] - clean["s_k"]))
    assert delta_margin == pytest.approx(decomposed, abs=1e-12)


def test_regression_cost_combines_center_size_and_wrapped_yaw():
    exact = regression_cost([0, 0, 0, 2, 4, 1, 0], [0, 0, 0], [2, 4, 1], 0)
    assert exact == pytest.approx(0.0)
    shifted = regression_cost([1, 0, 0, 2, 4, 1, 2 * np.pi - 0.1],
                              [0, 0, 0], [2, 4, 1], 0)
    assert shifted == pytest.approx(0.5 + 0.1 / np.pi)
