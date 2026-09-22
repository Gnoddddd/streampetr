import math

import pytest
import torch

from analysis.geocorr_correspondence_audit import (
    confidence_group_indices,
    correspondence_audit_metrics,
    select_object_group,
    shuffle_history_objects,
    summarize_correspondence_audit,
    summarize_valid_view_groups,
)


OFFSETS = [(1, 0), (0, 0), (-1, 0)]


def _probabilities():
    probabilities = torch.tensor(
        [[[[[[0.20, 0.10]], [[0.30, 0.10]], [[0.20, 0.10]]],
            [[[0.00, 0.00]], [[0.00, 0.00]], [[0.00, 0.00]]]]]]
    )
    mask = probabilities > 0
    return probabilities, mask


def test_position_and_view_marginals_normalize_and_center_is_not_index_four():
    probabilities, mask = _probabilities()
    metrics = correspondence_audit_metrics(probabilities, mask, OFFSETS)
    assert torch.allclose(metrics["position_marginal"][0, 0, 0].sum(), torch.tensor(1.0))
    assert torch.allclose(metrics["view_marginal"][0, 0, 0].sum(), torch.tensor(1.0))
    assert torch.allclose(metrics["center_candidate_mass"][0, 0, 0], torch.tensor(0.4))


def test_top1_margin_and_offset_are_correct():
    probabilities, mask = _probabilities()
    metrics = correspondence_audit_metrics(probabilities, mask, OFFSETS)
    assert torch.allclose(
        metrics["position_top1_minus_top2_margin"][0, 0, 0], torch.tensor(0.1)
    )
    assert torch.equal(metrics["position_top1_offset"][0, 0, 0], torch.tensor([0.0, 0.0]))


def test_same_view_uses_matching_history_camera_and_renormalizes():
    probabilities, mask = _probabilities()
    metrics = correspondence_audit_metrics(probabilities, mask, OFFSETS)
    expected_center = 0.30 / (0.20 + 0.30 + 0.20)
    assert math.isclose(
        metrics["same_view_center_mass"][0, 0, 0].item(), expected_center, rel_tol=1e-6
    )
    assert not metrics["same_view_query_valid"][0, 0, 1]


def test_shuffle_history_rolls_valid_objects_and_skips_singletons():
    tokens = torch.arange(3.0).reshape(1, 3, 1, 1, 1, 1)
    valid = torch.ones(1, 3, 1, 1, 1, dtype=torch.bool)
    shuffled, shuffled_valid, permutation, comparable = shuffle_history_objects(
        tokens, valid, torch.tensor([[True, True, False]])
    )
    assert permutation.tolist() == [[1, 0, 2]]
    assert shuffled[:, :2].flatten().tolist() == [1.0, 0.0]
    assert shuffled_valid.all()
    assert comparable.tolist() == [[True, True, False]]


def test_invalid_query_is_excluded_and_statistics_remain_finite():
    probabilities, mask = _probabilities()
    metrics = correspondence_audit_metrics(probabilities, mask, OFFSETS)
    assert metrics["query_valid"].tolist() == [[[True, False]]]
    for name, value in metrics.items():
        if value.dtype.is_floating_point:
            assert torch.isfinite(value).all(), name
    summary = summarize_correspondence_audit(metrics, OFFSETS)
    assert summary["num_valid_queries"] == 1
    assert summary["num_objects"] == 1


def test_score_and_object_indices_remain_aligned():
    scores = torch.tensor([0.2, 0.9, 0.4])
    objects = torch.tensor([[20, 90, 40]])
    indices = confidence_group_indices(scores)["top25"]
    assert select_object_group(objects, indices).tolist() == [[90, 40, 20]]
    assert scores.index_select(0, indices).tolist() == pytest.approx([0.9, 0.4, 0.2])


def test_topk_groups_contain_highest_scores_and_are_nested():
    scores = torch.arange(60.0)
    groups = confidence_group_indices(scores)
    assert groups["top25"].tolist() == list(range(59, 34, -1))
    assert set(groups["top25"].tolist()).issubset(set(groups["top50"].tolist()))


def test_quartiles_have_no_duplicates_or_omissions():
    groups = confidence_group_indices(torch.tensor([0.5, 0.5, 0.2, 0.9, 0.1]))
    quartiles = torch.cat(list(groups["quartiles"].values()))
    assert quartiles.tolist() == [3, 0, 1, 2, 4]
    assert sorted(quartiles.tolist()) == list(range(5))


def test_shuffle_after_selection_cannot_leave_confidence_group():
    tokens = torch.arange(6.0).reshape(1, 6, 1, 1, 1, 1)
    valid = torch.ones(1, 6, 1, 1, 1, dtype=torch.bool)
    group = torch.tensor([5, 3, 1])
    selected = select_object_group(tokens, group)
    shuffled, _, _, _ = shuffle_history_objects(
        selected, select_object_group(valid, group)
    )
    assert set(shuffled.flatten().tolist()) == {1.0, 3.0, 5.0}


def test_center_as_top1_rate_is_reported_for_all_and_same_view():
    probabilities, mask = _probabilities()
    metrics = correspondence_audit_metrics(probabilities, mask, OFFSETS)
    summary = summarize_correspondence_audit(metrics, OFFSETS)
    assert summary["center_as_top1_rate"] == 1.0
    assert summary["same_view_center_as_top1_rate"] == 1.0


def test_shuffle_delta_directions_follow_metric_semantics():
    probabilities, mask = _probabilities()
    correct = correspondence_audit_metrics(probabilities, mask, OFFSETS)
    shuffled_probabilities = probabilities.roll(1, dims=3)
    shuffled = correspondence_audit_metrics(shuffled_probabilities, mask.roll(1, dims=3), OFFSETS)
    summary = summarize_correspondence_audit(
        correct, OFFSETS, shuffled, torch.tensor([[True]])
    )["shuffled_history_control"]
    entropy = summary["normalized_position_entropy"]
    center = summary["center_candidate_mass"]
    assert entropy["delta"]["mean"] == pytest.approx(
        entropy["shuffled"]["mean"] - entropy["correct"]["mean"]
    )
    assert center["delta"]["mean"] == pytest.approx(
        center["correct"]["mean"] - center["shuffled"]["mean"]
    )
    assert summary["delta_position_entropy"] == entropy["delta"]["mean"]
    assert summary["delta_center_mass"] == center["delta"]["mean"]


def test_valid_history_view_count_grouping_is_exhaustive():
    shape = (1, 4, 1)
    metrics = {
        "query_valid": torch.ones(shape, dtype=torch.bool),
        "valid_historical_view_count": torch.tensor([[[1], [2], [3], [5]]]),
        "normalized_position_entropy": torch.tensor([[[0.1], [0.2], [0.3], [0.4]]]),
        "position_top1_probability": torch.tensor([[[0.4], [0.5], [0.6], [0.7]]]),
        "center_candidate_mass": torch.tensor([[[0.9], [0.8], [0.7], [0.6]]]),
        "position_top1_minus_top2_margin": torch.tensor(
            [[[0.1], [0.2], [0.3], [0.4]]]
        ),
        "position_top1_index": torch.tensor([[[1], [1], [0], [1]]]),
    }
    groups = summarize_valid_view_groups(metrics, OFFSETS)
    assert [groups[name]["num_valid_queries"] for name in groups] == [1, 1, 2]
    assert [groups[name]["query_count"] for name in groups] == [1, 1, 2]
    assert sum(group["num_valid_queries"] for group in groups.values()) == 4
    assert groups["3_or_more_views"]["center_as_top1_rate"] == 0.5


def test_empty_confidence_group_is_safe_and_json_finite():
    probabilities = torch.zeros(1, 0, 1, 3, 1, 1)
    mask = probabilities.bool()
    metrics = correspondence_audit_metrics(probabilities, mask, OFFSETS)
    summary = summarize_correspondence_audit(
        metrics, OFFSETS, scores=torch.empty(0)
    )
    assert summary["num_objects"] == 0
    assert summary["num_valid_queries"] == 0
    assert summary["score"] == {"min": 0.0, "median": 0.0, "max": 0.0}
    _assert_finite(summary)


def _assert_finite(value):
    if isinstance(value, dict):
        for child in value.values():
            _assert_finite(child)
    elif isinstance(value, float):
        assert math.isfinite(value)


def test_all_summary_values_have_no_nan_or_inf():
    probabilities, mask = _probabilities()
    metrics = correspondence_audit_metrics(probabilities, mask, OFFSETS)
    summary = summarize_correspondence_audit(
        metrics, OFFSETS, metrics, torch.tensor([[True]]), scores=torch.tensor([0.8])
    )
    _assert_finite(summary)
