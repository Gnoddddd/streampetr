import copy

import numpy as np
import pytest
import torch

from datasets.corruption import ApplyPartialObservation
from datasets.distillation import (
    FinalizePairedCleanImages,
    MatchFilteredGTInstanceTokens,
)


def _images():
    base = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    return [base + index for index in range(6)]


def test_exclusive_sampler_is_deterministic_and_at_most_one_family():
    transform = ApplyPartialObservation(
        seed=2026,
        exclusive_uniform=True,
        corruption_probability=1.0,
    )
    first = transform(
        dict(img=_images(), sample_idx="same", scene_token="s", frame_idx=3)
    )
    second = transform(
        dict(img=_images(), sample_idx="same", scene_token="s", frame_idx=3)
    )
    assert all(np.array_equal(a, b) for a, b in zip(first["img"], second["img"]))
    changed = [not np.array_equal(a, b) for a, b in zip(first["img"], _images())]
    assert 1 <= sum(changed) <= 2  # Compound is the sole two-view family.


def test_clean_pair_is_a_deep_copy_before_corruption():
    transform = ApplyPartialObservation(
        seed=9,
        exclusive_uniform=True,
        corruption_probability=1.0,
        preserve_clean=True,
    )
    result = transform(
        dict(img=_images(), sample_idx="paired", scene_token="s", frame_idx=0)
    )
    assert all(np.array_equal(a, b) for a, b in zip(result["clean_img"], _images()))
    assert any(not np.array_equal(a, b) for a, b in zip(result["img"], result["clean_img"]))


def test_finalize_clean_images_matches_student_layout():
    transform = FinalizePairedCleanImages(
        mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False, size_divisor=4
    )
    result = transform(dict(clean_img=_images()))
    assert np.asarray(result["clean_img"]).shape == (6, 3, 8, 12)


class _Boxes:
    def __init__(self, tensor):
        self.tensor = torch.as_tensor(tensor, dtype=torch.float32)


def test_filtered_gt_tokens_follow_class_aware_box_matching():
    transform = MatchFilteredGTInstanceTokens()
    result = transform(
        {
            "_raw_token_boxes": np.asarray([[1, 2, 9], [4, 5, 8]], np.float32),
            "_raw_token_labels": np.asarray([2, 3]),
            "_raw_gt_instance_tokens": ["instance-a", "instance-b"],
            "gt_bboxes_3d": _Boxes([[4, 5, 0], [1, 2, 0]]),
            "gt_labels_3d": np.asarray([3, 2]),
        }
    )
    assert result["gt_instance_tokens"] == ["instance-b", "instance-a"]


@pytest.mark.integration
def test_disabled_detector_has_exact_b0_state_and_parameter_count():
    pytest.importorskip("mmdet3d")
    from mmcv import Config
    from mmdet3d.models import build_model

    baseline_cfg = Config.fromfile("configs/stage3/mini_observability_b0.py")
    r1_cfg = Config.fromfile("configs/stage3/mini_observability_r1.py")
    disabled_cfg = copy.deepcopy(r1_cfg.model)
    disabled_cfg.enable_observability_distillation = False
    disabled_cfg.pts_bbox_head.enable_observability_distillation = False
    baseline = build_model(
        baseline_cfg.model,
        train_cfg=baseline_cfg.get("train_cfg"),
        test_cfg=baseline_cfg.get("test_cfg"),
    )
    disabled = build_model(
        disabled_cfg,
        train_cfg=r1_cfg.get("train_cfg"),
        test_cfg=r1_cfg.get("test_cfg"),
    )
    baseline_state = baseline.state_dict()
    disabled_state = disabled.state_dict()
    assert tuple(baseline_state) == tuple(disabled_state)
    assert sum(p.numel() for p in baseline.parameters()) == sum(
        p.numel() for p in disabled.parameters()
    )
    assert all("_ema_teacher" not in key for key in disabled_state)


def test_ema_teacher_is_frozen_and_not_registered_in_deployment_state():
    from hooks.ema_teacher_hook import TrainingOnlyEMATeacherHook

    class TinyStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            self.enable_observability_distillation = True
            self.num_frame_losses = 1
            object.__setattr__(self, "_ema_teacher", None)

        def set_ema_teacher(self, teacher):
            object.__setattr__(self, "_ema_teacher", teacher)

    class Runner:
        model = TinyStudent()

    before = tuple(Runner.model.state_dict())
    TrainingOnlyEMATeacherHook(momentum=0.999).before_run(Runner)
    teacher = Runner.model._ema_teacher
    assert teacher is not None
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert tuple(Runner.model.state_dict()) == before
    assert all("teacher" not in key for key in Runner.model.state_dict())
