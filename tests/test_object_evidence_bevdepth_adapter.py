from types import SimpleNamespace

import torch
from torch import nn

from models.object_evidence.adapters.bevdepth import (
    BEVDepthAdapter, BEVFeatureCapture, camera_reliability, weighted_depth_loss,
)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.neck = nn.Identity()
        self.train_cfg = dict(
            point_cloud_range=[0, 0, -1, 4, 4, 1], voxel_size=[1, 1, 2], out_size_factor=1,
        )


def test_neck_hook_and_runtime_shape():
    head = Head()
    value = torch.randn(2, 256, 3, 5)
    with BEVFeatureCapture(head) as capture:
        head.neck(value)
    assert capture.tensor.shape == (2, 256, 3, 5)


def test_bilinear_gt_center_sampling_and_output_shape():
    head = Head()
    feature = torch.arange(16.0).reshape(1, 1, 4, 4).expand(1, 256, 4, 4).clone()
    result = BEVDepthAdapter(head).extract(feature, [torch.tensor([[1.0, 2.0, 0.0]])])
    assert result.tokens.shape == (1, 256)
    assert torch.allclose(result.tokens, torch.full((1, 256), 9.0))
    assert result.valid_mask.tolist() == [True]


def test_fault_depth_loss_changes_only_selected_camera():
    reliability = camera_reliability(6, 3, "dark", 0.6)
    assert torch.allclose(reliability, torch.tensor([1, 1, 1, 0.4, 1, 1.0]))
    prediction = torch.full((1, 6, 2, 1, 1), 0.5)
    target = torch.zeros(1, 6, 1, 1, 2)
    target[..., 0] = 1
    loss = weighted_depth_loss(prediction, target, reliability)
    assert torch.isfinite(loss) and loss > 0
