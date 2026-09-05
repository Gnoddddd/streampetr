import numpy as np

from analysis.temporal_representation_localization import (
    geometry_candidates,
    geometry_match,
    local_non_gt_candidates,
    matched_representation_metrics,
)


def test_candidates_and_matching_are_geometry_only_and_deterministic():
    source = np.array([[0.2, 0, 0], [1.4, 0, 0], [4, 0, 0]], float)
    target = np.array([[1.3, 0, 0], [0.1, 0, 0], [5, 0, 0]], float)
    si = geometry_candidates(source, [0, 0, 0])
    ti = geometry_candidates(target, [0, 0, 0])
    assert si.tolist() == [0, 1]
    assert ti.tolist() == [1, 0]
    assert [(a, b) for a, b, _ in geometry_match(source, target, si, ti)] == [(1, 0), (0, 1)]


def test_representation_metrics_use_all_pairs():
    source = np.array([[1, 0], [0, 1]], float)
    target = np.array([[1, 0], [1, 0]], float)
    result = matched_representation_metrics(source, target, [(0, 0, 0), (1, 1, 0)])
    assert result["matched_pair_count"] == 2
    assert np.isclose(result["cosine_distance"], 0.5)
    assert np.isclose(result["normalized_l2"], np.sqrt(2) / 2)


def test_non_gt_control_is_same_count_local_and_excludes_all_gt():
    boxes = np.array([[0, 0, 0], [2.1, 0, 0], [3, 0, 0], [10, 0, 0]], float)
    selected = local_non_gt_candidates(boxes, np.array([[0, 0, 0]]), [0, 0, 0], 2)
    assert selected.tolist() == [1, 2]
