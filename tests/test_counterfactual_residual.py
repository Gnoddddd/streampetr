import json
from pathlib import Path

import numpy as np

from analysis.counterfactual_residual import (
    fault_key,
    independent_match,
    residual_target,
    wrap_yaw,
)


def test_wrapped_yaw_is_periodic_and_bounded():
    values = np.asarray([-4 * np.pi, -np.pi - 0.2, 0.3, np.pi + 0.2, 4 * np.pi])
    wrapped = wrap_yaw(values)
    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped <= np.pi)
    assert np.isclose(wrapped[1], np.pi - 0.2)
    assert np.isclose(wrapped[3], -np.pi + 0.2)


def test_independent_matching_uses_geometry_not_query_index():
    gt_center = np.asarray([[0, 0, 0], [10, 0, 0]], dtype=float)
    gt_label = np.asarray([0, 1])
    prediction_center = np.asarray([[10.1, 0, 0], [0.1, 0, 0]], dtype=float)
    prediction_label = np.asarray([1, 0])
    assert independent_match(
        gt_center, gt_label, prediction_center, prediction_label
    ) == {0: 1, 1: 0}


def test_residual_target_has_full_minus_available_sign():
    available = {
        "logits": np.zeros(10),
        "center": np.asarray([1.0, 2.0, 3.0]),
        "size": np.asarray([2.0, 3.0, 4.0]),
        "yaw": np.pi - 0.1,
        "velocity": np.asarray([1.0, -1.0]),
    }
    full = {
        "logits": np.ones(10),
        "center": np.asarray([2.0, 4.0, 6.0]),
        "size": np.asarray([4.0, 3.0, 2.0]),
        "yaw": -np.pi + 0.1,
        "velocity": np.asarray([2.0, 1.0]),
    }
    target = residual_target(full, available)
    assert np.allclose(target[:10], 1)
    assert np.allclose(available["center"] + target[10:13], full["center"])
    assert np.isclose(target[16], 0.2)
    assert np.allclose(available["velocity"] + target[17:19], full["velocity"])


def test_fault_keys_separate_seen_and_unseen_camera_sets():
    assert fault_key(np.asarray([0, 1, 1, 1, 1, 1]), 3) == "single_duration_3"
    assert fault_key(np.asarray([0, 0, 1, 1, 1, 1]), 5) == (
        "adjacent_double_duration_5"
    )
    assert fault_key(np.asarray([0, 1, 1, 0, 1, 1]), 5) == (
        "nonadjacent_double_duration_5"
    )
    assert fault_key(np.asarray([0, 1, 0, 1, 0, 1]), 2) == (
        "three_camera_duration_other"
    )


def test_counterfactual_manifests_are_split_safe_and_frozen():
    root = Path(__file__).resolve().parents[1]
    protocol_root = root / "protocols/counterfactual_view_deficit"
    train = json.loads((protocol_root / "train_seen.json").read_text())
    seen = json.loads((protocol_root / "val_seen.json").read_text())
    assert len(train["scenes"]) == 8
    assert len(seen["scenes"]) == 2
    assert set(train["scenes"]).isdisjoint(seen["scenes"])
    for events in train["scenes"].values():
        for event in events:
            duration = event["end_frame"] - event["start_frame"] + 1
            assert duration in (1, 3, 5)
            assert len(event["failed_cameras"]) in (1, 2)
