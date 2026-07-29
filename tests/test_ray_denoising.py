from types import SimpleNamespace

import torch

from models.ray_denoising import prepare_ray_denoising


class _Boxes:
    def __init__(self, tensor):
        self.tensor = tensor
        self.gravity_center = tensor[:, :3]


def _head():
    return SimpleNamespace(
        training=True,
        with_dn=True,
        scalar=2,
        bbox_noise_scale=1.0,
        bbox_noise_trans=0.0,
        split=0.75,
        pc_range=torch.tensor([-10.0, -10.0, -5.0, 10.0, 10.0, 15.0]),
        num_classes=3,
        num_query=4,
        num_propagated=2,
        memory_len=6,
    )


def _inputs(dtype=torch.float32):
    box_tensor = torch.tensor(
        [[1.0, 0.0, 5.0, 2.0, 2.0, 4.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=dtype,
    )
    meta = {
        "gt_bboxes_3d": _Boxes(box_tensor),
        "gt_labels_3d": torch.tensor([1]),
        "pad_shape": [(100, 100, 3)],
    }
    data = {"lidar2img": torch.eye(4, dtype=dtype).view(1, 1, 4, 4)}
    reference = torch.rand(4, 3, dtype=dtype)
    return reference, [meta], data


def test_raydn_constructs_one_positive_and_four_hard_negatives():
    torch.manual_seed(2026)
    head = _head()
    reference, metas, data = _inputs()
    padded, mask, mask_dict = prepare_ray_denoising(
        head, 1, reference, metas, data
    )

    # Two existing DN copies plus five ray depths for the single GT.
    assert mask_dict["pad_size"] == 7
    assert mask_dict["raydn_pad_size"] == 5
    assert padded.shape == (1, 11, 3)
    assert mask.shape == (13, 17)
    labels = mask_dict["known_lbs_bboxes"][0][-5:]
    assert (labels == 1).sum().item() == 1
    assert (labels == head.num_classes).sum().item() == 4
    assert mask[7:, :7].all()


def test_raydn_is_training_only_and_has_no_runtime_state():
    head = _head()
    head.training = False
    reference, metas, data = _inputs()
    padded, mask, mask_dict = prepare_ray_denoising(
        head, 1, reference, metas, data
    )
    assert torch.equal(padded[0], reference)
    assert mask is None
    assert mask_dict is None
    assert not hasattr(head, "raydn_sampler")


def test_raydn_cpu_fp16_is_safe():
    torch.manual_seed(2026)
    head = _head()
    head.pc_range = head.pc_range.half()
    reference, metas, data = _inputs(torch.float16)
    padded, mask, mask_dict = prepare_ray_denoising(
        head, 1, reference, metas, data
    )
    assert padded.dtype == torch.float16
    assert torch.isfinite(padded).all()
    assert mask.dtype == torch.bool
    assert mask_dict["known_lbs_bboxes"][1].dtype == torch.float16

