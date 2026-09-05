import math

import numpy as np
import torch

from analysis.dark_target_recoverability import (
    background_ring_mask, bootstrap_median, centroid_separability,
    darken_normalized_image, destructive_fraction, match_retained_controls,
    projected_roi, recovery_fraction, replace_local_feature, roi_cell_mask,
)


def test_darkening_matches_raw_pixel_multiplication():
    raw = torch.tensor([[[100.]], [[80.]], [[60.]]])
    mean, std = [10., 20., 30.], [2., 4., 5.]
    mean_t, std_t = torch.tensor(mean)[:, None, None], torch.tensor(std)[:, None, None]
    normalized = (raw - mean_t) / std_t
    assert torch.allclose(darken_normalized_image(normalized, .25, mean, std),
                          (raw * .25 - mean_t) / std_t)


def test_projection_masks_and_replacement():
    corners = np.array([[0, 0, 1], [2, 0, 1], [0, 2, 1], [2, 2, 1],
                        [0, 0, 2], [2, 0, 2], [0, 2, 2], [2, 2, 2]], float)
    roi = projected_roi(corners, np.eye(4), (4, 4))
    assert roi == (0., 0., 2., 2.)
    mask = roi_cell_mask(roi, (4, 4), (2, 2))
    assert mask.tolist() == [[True, False], [False, False]]
    assert background_ring_mask(roi, [], (4, 4), (4, 4)).any()
    result = replace_local_feature(torch.zeros(2, 1, 2, 2), torch.ones(2, 1, 2, 2), mask, 1)
    assert result[0].sum() == 0 and result[1].sum() == 1


def test_separability_fractions_and_seed():
    feature = torch.tensor([[[1., 0.], [1., 0.]], [[0., 1.], [0., 1.]]])
    target = np.array([[True, False], [True, False]])
    result = centroid_separability(feature, target, ~target)
    assert math.isclose(result["cosine_distance"], 1., abs_tol=1e-6)
    assert math.isclose(recovery_fraction(.8, .2, .5), .5)
    assert math.isclose(destructive_fraction(.8, .2, .5), .5)
    assert bootstrap_median([1, 2, 3], 9, 50) == bootstrap_median([1, 2, 3], 9, 50)


def test_controls_are_unique_and_class_preferred():
    base = {"sample_token": "f", "gt_center_distance": 10,
            "alternative_view_count": 0, "max_projected_box_area_fraction": .1,
            "clean_s_pos": .8}
    lost = [dict(base, gt_token="l1", gt_class="car"),
            dict(base, gt_token="l2", gt_class="pedestrian")]
    retained = [dict(base, gt_token="r1", gt_class="pedestrian"),
                dict(base, gt_token="r2", gt_class="car")]
    pairs = match_retained_controls(lost, retained)
    assert len({p["retained"]["gt_token"] for p in pairs}) == 2
    assert all(p["lost"]["gt_class"] == p["retained"]["gt_class"] for p in pairs)
