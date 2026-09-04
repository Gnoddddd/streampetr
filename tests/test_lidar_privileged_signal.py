import math

import numpy as np
import torch

from analysis.lidar_privileged_signal import (
    circular_error,
    cross_modal_signal_decision,
    greedy_class_center_match,
    sample_bev_features,
    teacher_coverage_decision,
)


def test_greedy_match_is_class_aware_and_one_to_one():
    gt = [{"token": "a", "label": 0, "center": [0, 0, 0]},
          {"token": "b", "label": 0, "center": [0.5, 0, 0]},
          {"token": "c", "label": 1, "center": [0, 0, 0]}]
    result = greedy_class_center_match(
        gt, [0, 1], [[0.1, 0, 0], [0.1, 0, 0]], [.9, .8])
    assert result == {"a": 0, "c": 1}


def test_bev_sampling_respects_xy_grid():
    feature = torch.arange(16.0).view(1, 1, 4, 4)
    sampled = sample_bev_features(
        feature, np.asarray([[.5, .5, 0]]), point_cloud_range=(0, 0, 4, 4),
        stride=1.0)
    assert sampled.shape == (1, 1)
    assert sampled[0, 0] == 0.0


def test_circular_error_wraps_pi_boundary():
    assert math.isclose(circular_error(math.pi - .1, -math.pi + .1), .2)


def test_teacher_gate_requires_all_three_families():
    protocol = {name: {"lost_match_rate": .8, "lost_median_score": .4,
                       "lost_median_xy_error": .8}
                for name in ("dark", "blur", "crash")}
    pooled = {"lost_match_rate": .85, "retained_match_rate": .9,
              "lost_median_score": .4, "lost_median_xy_error": .8,
              "lost_median_relative_size_l1": .2,
              "lost_median_yaw_error_deg": 20}
    temporal = {"lost_pair_count": 10, "median_abs_score_delta": .1,
                "median_abs_center_error_delta": .2,
                "median_representation_cosine_distance": .1}
    assert teacher_coverage_decision(protocol, pooled, temporal)[
        "teacher_coverage_pass"]
    temporal["median_representation_cosine_distance"] = .4
    assert not teacher_coverage_decision(protocol, pooled, temporal)[
        "teacher_coverage_pass"]


def test_cross_modal_gate_requires_lost_gap_and_lidar_superiority():
    gap = {"score_median": .1, "score_ci_low": .02,
           "score_cross_protocol": True, "score_enrichment_median": .05,
           "score_enrichment_ci_low": .01,
           "representation_median": .05, "representation_ci_low": .01,
           "representation_cross_protocol": True,
           "representation_enrichment_median": .02,
           "representation_enrichment_ci_low": .005,
           "center_median": .2, "center_ci_low": .05,
           "center_cross_protocol": True}
    superiority = {"score_strength": False, "geometry_strength": True,
                   "temporal_representation": True,
                   "temporal_score_or_geometry": True}
    assert cross_modal_signal_decision(True, gap, superiority)["signal_pass"]
    superiority["temporal_representation"] = False
    result = cross_modal_signal_decision(True, gap, superiority)
    assert result["decision"] == "NO_GO_LIDAR_NOT_BETTER_THAN_CLEAN_TEACHER"
