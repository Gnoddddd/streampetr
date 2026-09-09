from pathlib import Path
import runpy

from mmcv import Config
import numpy as np
import torch
from torch import nn

from datasets.corruption import apply_dark
from models.object_evidence.fault_sampler import FaultEpisode, FaultSpec
from models.object_evidence.integrations.bevdepth_lightning import (
    bevdepth_storage_episode, downsample_depth_labels, reshape_depth_prediction,
    torch_voxel_pooling_train,
)
from models.object_evidence.integrations.fault_images import fault_normalized_images
from models.object_evidence.paired_training import temporary_native_teacher


ROOT = Path(__file__).resolve().parents[1]


def test_sampled_episode_uses_existing_pixel_corruption_after_onset():
    mean, std = [10, 20, 30], [2, 4, 5]
    pixels = np.full((4, 5, 3), 100, dtype=np.float32)
    normalized = torch.from_numpy((pixels - np.asarray(mean)) / np.asarray(std))
    clean = normalized.permute(2, 0, 1).view(1, 1, 1, 3, 4, 5).expand(1, 2, 6, 3, 4, 5).clone()
    episode = FaultEpisode(FaultSpec(2, "dark", 0.6), (False, True))
    fault = fault_normalized_images(clean, episode, mean, std, ["scene"])
    assert torch.equal(fault[:, 0], clean[:, 0])
    assert torch.equal(fault[:, 1, [0, 1, 3, 4, 5]], clean[:, 1, [0, 1, 3, 4, 5]])
    expected_pixels = apply_dark(pixels, 0.6)
    expected = torch.from_numpy((expected_pixels - np.asarray(mean)) / np.asarray(std)).permute(2, 0, 1)
    assert torch.allclose(fault[0, 1, 2], expected)


def test_depth_prediction_runtime_reshape_has_explicit_camera_axis():
    value = torch.randn(12, 8, 3, 4)
    assert reshape_depth_prediction(value, 2, 6).shape == (2, 6, 8, 3, 4)


def test_bevdepth_maps_chronological_onset_to_current_first_storage():
    episode = FaultEpisode(FaultSpec(1, "crash", 1.0), (False, False, True))
    assert bevdepth_storage_episode(episode).active == (True, False, False)


def test_depth_target_downsampling_matches_runtime_shape():
    owner = type("Owner", (), dict(
        downsample_factor=2, dbound=(1.0, 5.0, 1.0), depth_channels=4,
    ))()
    labels = torch.zeros(1, 2, 4, 4)
    labels[:, :, 0, 0] = 2
    result = downsample_depth_labels(owner, labels)
    assert result.shape == (8, 4)
    assert torch.isfinite(result).all()


def test_reference_voxel_pooling_sums_cells_and_backpropagates():
    geometry = torch.tensor([[[0, 0, 0], [0, 0, 0], [1, 1, 0], [2, 0, 0]]])
    features = torch.tensor([[[1.0], [2.0], [4.0], [8.0]]], requires_grad=True)
    output = torch_voxel_pooling_train(geometry, features, torch.tensor([2, 2, 1]))
    assert output.shape == (1, 1, 2, 2)
    assert torch.equal(output.detach(), torch.tensor([[[[3.0, 0.0], [0.0, 4.0]]]]))
    output.sum().backward()
    assert torch.equal(features.grad, torch.tensor([[[1.0], [1.0], [1.0], [0.0]]]))


def test_native_teacher_stops_bn_updates_and_restores_training_flags():
    model = nn.Sequential(nn.BatchNorm1d(4), nn.Dropout())
    model.train()
    before = model[0].running_mean.clone()
    with temporary_native_teacher(model):
        model(torch.ones(2, 4))
    assert torch.equal(model[0].running_mean, before)
    assert model.training and model[0].training and model[1].training


def test_full_stream_configs_share_everything_except_oe_weight(monkeypatch):
    monkeypatch.setenv("OE_ADAPTATION_MAX_ITERS", "12")
    r0 = Config.fromfile(str(ROOT / "configs/object_evidence/streampetr_r0_full.py"))
    oe = Config.fromfile(str(ROOT / "configs/object_evidence/streampetr_oe_full.py"))
    assert r0.data.train.ann_file == "data/nuscenes/nuscenes2d_temporal_infos_train.pkl"
    assert r0.load_from == oe.load_from
    assert r0.optimizer == oe.optimizer and r0.runner == oe.runner
    left, right = dict(r0.model.object_evidence), dict(oe.model.object_evidence)
    assert left.pop("lambda_oe") == 0 and right.pop("lambda_oe") == 0.5
    assert left == right


def test_full_bevdepth_configs_share_everything_except_oe_weight(monkeypatch):
    monkeypatch.setenv("OE_ADAPTATION_MAX_EPOCHS", "3")
    r0 = runpy.run_path(str(ROOT / "configs/object_evidence/bevdepth_r0_full.py"))
    oe = runpy.run_path(str(ROOT / "configs/object_evidence/bevdepth_oe_full.py"))
    assert r0["train_info"] == "data/nuscenes/nuscenes_infos_train.pkl"
    assert r0["pretrained_checkpoint"] == oe["pretrained_checkpoint"]
    assert r0["adaptation_max_epochs"] == oe["adaptation_max_epochs"] == 3
    left, right = dict(r0["object_evidence"]), dict(oe["object_evidence"])
    assert left.pop("lambda_oe") == 0 and right.pop("lambda_oe") == 0.5
    assert left == right
