import numpy as np

from analysis.paired_rank_margin import (
    bootstrap_difference,
    fixed_query_statistics,
    query_margin_statistics,
    stable_rank,
)


def test_query_margin_uses_highest_gt_class_score_within_two_metres():
    logits = np.full((4, 2), -3.0)
    logits[1, 1] = 1.0
    logits[2, 1] = 2.0
    boxes = np.zeros((4, 9))
    boxes[0, :3] = [9, 0, 0]
    boxes[1, :3] = [1, 0, 0]
    boxes[2, :3] = [1.5, 0, 0]
    boxes[3, :3] = [8, 0, 0]
    result = query_margin_statistics(logits, boxes, [0, 0, 0], 1, topk=2)
    assert result["best_query"] == 2
    assert result["geometry_best_query"] == 1
    assert result["near_count"] == 2
    assert result["margin"] == result["score"] - result["s_k"]


def test_no_near_query_is_missing_not_boundary_crossing():
    result = query_margin_statistics(
        np.zeros((2, 2)), np.asarray([[3, 0, 0] * 3, [4, 0, 0] * 3]),
        [0, 0, 0], 0, topk=2,
    )
    assert not result["candidate_available"]
    assert result["rank"] == -1
    assert np.isnan(result["margin"])


def test_fixed_query_tracks_same_lineage_after_fault():
    logits = np.asarray([[0.0, 1.0], [2.0, -1.0]])
    boxes = np.zeros((2, 9))
    result = fixed_query_statistics(logits, boxes, 0, [0, 0, 0], 1)
    assert result["geometry_qualified"]
    assert result["rank"] == stable_rank(
        1 / (1 + np.exp(-logits.reshape(-1))), 1
    )


def test_bootstrap_difference_is_reproducible_and_directional():
    first = bootstrap_difference([-3, -2, -1], [0, 1, 2], np.median, 314159, 1000)
    second = bootstrap_difference([-3, -2, -1], [0, 1, 2], np.median, 314159, 1000)
    assert first == second
    assert first["estimate"] < 0
    assert first["ci_high"] <= 0

