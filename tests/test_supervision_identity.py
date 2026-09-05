import math

from analysis.supervision_identity import (
    assignment_identity,
    bootstrap_rate_difference,
    trajectory_statistics,
    wilson_interval,
)


def test_assignment_identity_distinguishes_same_other_and_background():
    assert assignment_identity(2, 2) == "same-GT positive"
    assert assignment_identity(1, 2) == "other-GT matched"
    assert assignment_identity(-1, 2) == "unmatched-background"
    assert assignment_identity(0, None) == "other-GT matched"


def test_trajectory_statistics_tracks_longitudinal_supervision():
    result = trajectory_statistics([
        "unmatched-background",
        "same-GT positive",
        "same-GT positive",
        "other-GT matched",
    ])
    assert result["same_gt_layer_count"] == 2
    assert result["other_gt_layer_count"] == 1
    assert result["background_layer_count"] == 1
    assert result["same_gt_layer_fraction"] == 0.5
    assert result["ever_same_gt"] and not result["always_same_gt"]
    assert not result["never_same_gt"]
    assert result["first_same_gt_layer"] == 1
    assert result["last_same_gt_layer"] == 2
    assert result["identity_switch_count"] == 2


def test_wilson_and_bootstrap_rate_difference_are_bounded_and_seeded():
    low, high = wilson_interval(7, 10)
    assert 0.0 < low < 0.7 < high < 1.0
    first = bootstrap_rate_difference([True, True, False], [False, False], 17, 100)
    second = bootstrap_rate_difference([True, True, False], [False, False], 17, 100)
    assert first == second
    assert math.isclose(first["estimate"], 2.0 / 3.0)
