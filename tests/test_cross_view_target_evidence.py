import numpy as np
import torch

from analysis.cross_view_target_evidence import (
    alternative_stratum,
    classify_mechanism,
    patch_camera_masks,
    same_area_background_mask,
)


def test_same_area_background_is_exact_and_excludes_gt_cells():
    target = np.zeros((5, 6), bool)
    target[2, 2:4] = True
    excluded = target.copy()
    excluded[1, 2] = True
    background = same_area_background_mask(target, excluded)
    assert background.sum() == target.sum()
    assert not np.any(background & excluded)


def test_patch_camera_masks_changes_only_selected_camera_cells():
    fault = torch.zeros(3, 2, 2, 2)
    clean = torch.ones_like(fault)
    mask = np.asarray([[True, False], [False, False]])
    output = patch_camera_masks(fault, clean, {1: mask})
    assert output[1, :, 0, 0].eq(1).all()
    assert output.sum() == 2


def _summary(tp, topk, fraction):
    return {"tp_recovery_rate": tp, "topk_recovery_rate": topk,
            "median_rescue_fraction": fraction}


def test_mechanism_classifier_distinguishes_underuse_and_primary_view():
    summaries = {
        "cam_back_target_clean": _summary(.7, .7, .8),
        "other_visible_target_clean": _summary(.1, .1, .1),
        "all_visible_target_clean": _summary(.75, .75, .85),
        "all_visible_background_clean": _summary(.1, .1, .1),
    }
    value = classify_mechanism(summaries, True, .4)
    assert value["mechanism"] == "alternative_evidence_underuse"
    value = classify_mechanism(summaries, False, .1)
    assert value["mechanism"] == "primary_view_dependence"


def test_alternative_strata_are_fixed():
    assert [alternative_stratum(value) for value in (0, 1, 2, 7)] == ["0", "1", "2+", "2+"]
