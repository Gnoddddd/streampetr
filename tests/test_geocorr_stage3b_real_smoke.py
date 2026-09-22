import math

import pytest
import torch

from datasets.temporal_occ_nuscenes import NUSCENES_CAMERA_ORDER
from evaluation.geocorr_stage3b_smoke import (
    descriptor_l2_mean,
    masked_statistics,
    probability_entropy_mean,
    probability_sum_error,
    select_manifest_pairs,
    tensor_nonfinite_counts,
)


def _record(condition="Dirt/0.3_dirt"):
    return {
        "raw_condition": condition,
        "camera_names": list(NUSCENES_CAMERA_ORDER),
    }


def test_manifest_pair_selection_is_condition_filtered_bounded_and_ordered():
    records = [
        _record("Dirt/0.1_dirt"),
        _record("Water-blur/0.3_water-blur"),
        _record("Dirt/0.3_dirt"),
    ]
    selected = select_manifest_pairs(
        records, ("Water-blur/0.3_water-blur", "Dirt/0.3_dirt"), 1
    )

    assert selected == [records[1]]


def test_manifest_pair_selection_rejects_camera_order_drift():
    record = _record()
    record["camera_names"] = list(reversed(NUSCENES_CAMERA_ORDER))

    with pytest.raises(ValueError, match="camera order"):
        select_manifest_pairs([record], None, 1)


def test_nonfinite_counter_covers_nan_and_both_infinities():
    nan_count, inf_count = tensor_nonfinite_counts({
        "a": torch.tensor([0.0, float("nan")]),
        "b": torch.tensor([float("inf"), -float("inf")]),
    })

    assert nan_count == 1
    assert inf_count == 2


def test_masked_confidence_statistics_ignore_invalid_queries():
    values = torch.tensor([[[[0.1], [0.5], [0.9]]]])
    valid = torch.tensor([[[True, False, True]]])
    statistics = masked_statistics(values, valid)

    assert statistics == pytest.approx(
        {"min": 0.1, "mean": 0.5, "median": 0.1, "max": 0.9}
    )


def test_probability_sum_error_uses_one_for_valid_and_zero_for_invalid():
    probabilities = torch.tensor([1.0, 0.0, 0.0, 0.0]).reshape(1, 1, 2, 1, 1, 2)
    valid = torch.tensor([[[True, False]]])

    assert probability_sum_error(probabilities, valid) == 0.0


def test_entropy_and_descriptor_l2_summaries_are_finite():
    probabilities = torch.tensor([0.5, 0.5]).reshape(1, 1, 1, 1, 1, 2)
    descriptor = torch.tensor([3.0, 4.0]).reshape(1, 1, 1, 2)
    valid = torch.ones(1, 1, 1, dtype=torch.bool)

    assert math.isclose(
        probability_entropy_mean(probabilities, valid), math.log(2.0), rel_tol=1e-6
    )
    assert descriptor_l2_mean(descriptor, valid) == 5.0
