import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from models.adapters import PreviousPrediction, from_streampetr_result
from models.geocorr_recovery import GeometryCandidateSampler, sample_candidate_features

MMDET3D_ROOT = Path(__file__).resolve().parents[1] / "repos/StreamPETR/mmdetection3d"


def _upstream_points_cam2img(point, matrix):
    """Run the pinned upstream helper without polluting pytest's module cache."""
    program = """
import json
import torch
from mmdet3d.core.bbox.structures.utils import points_cam2img
payload = json.loads(input())
point = torch.tensor(payload['point'])
matrix = torch.tensor(payload['matrix'])
print(json.dumps(points_cam2img(point, matrix).tolist()))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(MMDET3D_ROOT)
    payload = json.dumps({"point": point.tolist(), "matrix": matrix.tolist()})
    output = subprocess.check_output(
        [sys.executable, "-c", program], input=payload, text=True, env=environment
    )
    return torch.tensor(json.loads(output))


OFFSETS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _prediction(center=None, velocity=None):
    return PreviousPrediction(
        center_3d=torch.tensor(center or [[1.0, 2.0, 3.0]], requires_grad=True),
        size_3d=torch.tensor([[2.0, 4.0, 1.5]], requires_grad=True),
        yaw=torch.tensor([0.25], requires_grad=True),
        velocity=None if velocity is None else torch.tensor(velocity, requires_grad=True),
        score=torch.tensor([0.9], requires_grad=True),
        label=torch.tensor([2]),
        timestamp=4.0,
    )


def test_streampetr_result_adapter_uses_decoded_predictions_and_detaches():
    boxes = torch.tensor([[1.0, 2.0, 3.0, 2.0, 4.0, 2.0, 0.5, 3.0, -1.0]], requires_grad=True)
    result = {
        "pts_bbox": {
            "boxes_3d": boxes,
            "scores_3d": torch.tensor([0.8], requires_grad=True),
            "labels_3d": torch.tensor([4]),
        }
    }
    output = from_streampetr_result(result, timestamp=10.0)
    assert torch.allclose(output.center_3d, torch.tensor([[1.0, 2.0, 4.0]]))
    assert torch.equal(output.size_3d, boxes[:, 3:6])
    assert torch.equal(output.velocity, boxes[:, 7:9])
    assert output.timestamp == 10.0 and output.coordinate_frame == "previous_lidar"
    assert not output.center_3d.requires_grad
    assert not output.score.requires_grad


def test_ego_motion_and_velocity_propagation():
    sampler = GeometryCandidateSampler(OFFSETS)
    current_from_previous = torch.eye(4)
    current_from_previous[:3, 3] = torch.tensor([10.0, -2.0, 1.0])
    propagated = sampler.propagate(
        _prediction(velocity=[[2.0, -4.0]]), current_from_previous, delta_t=0.5
    )
    assert torch.allclose(propagated, torch.tensor([[12.0, -2.0, 4.0]]))


def test_candidate_grid_is_3_by_3_and_keeps_height():
    sampler = GeometryCandidateSampler(OFFSETS)
    anchor = torch.tensor([[10.0, 20.0, 2.5]])
    candidates = sampler.generate_candidates(anchor)
    assert candidates.shape == (1, 9, 3)
    assert torch.equal(candidates[0, :, :2], torch.tensor(OFFSETS) + anchor[0, :2])
    assert torch.equal(candidates[0, :, 2], torch.full((9,), 2.5))


def test_camera_projection_matches_homogeneous_project_and_masks_invalid():
    sampler = GeometryCandidateSampler(OFFSETS)
    candidates = torch.tensor([[[2.0, 4.0, 2.0]]]).expand(1, 9, 3).clone()
    cameras = torch.eye(4).repeat(2, 1, 1)
    cameras[1, 2, 2] = -1.0  # same pixel algebra, but behind camera
    output = sampler.project(candidates, cameras, torch.tensor([[10, 10], [10, 10]]), (5, 5))
    # This is the projection helper shipped by the pinned StreamPETR copy.
    reference = _upstream_points_cam2img(
        torch.tensor([[2.0, 4.0, 2.0]]), cameras[0]
    )[0]
    assert torch.allclose(output["projected_points_2d"][0, 0, 0], reference)
    assert output["camera_valid_mask"][0, :, 0].all()
    assert not output["camera_valid_mask"][0, :, 1].any()


def test_raw_projection_applies_augmentation_matrix_once():
    sampler = GeometryCandidateSampler(OFFSETS)
    candidates = torch.tensor([[[1.0, 2.0, 1.0]]]).expand(1, 9, 3).clone()
    raw = torch.eye(4).unsqueeze(0)
    aug = torch.tensor([[2.0, 0.0, 3.0], [0.0, 2.0, -1.0], [0.0, 0.0, 1.0]])
    explicitly_augmented = raw.clone()
    explicitly_augmented[:, :3, :] = aug @ raw[:, :3, :]
    shape = torch.tensor([[20, 20]])
    from_raw = sampler.project(candidates, raw, shape, (10, 10), aug, False)
    from_aug = sampler.project(candidates, explicitly_augmented, shape, (10, 10))
    assert torch.allclose(from_raw["projected_points_2d"], from_aug["projected_points_2d"])
    assert torch.allclose(from_raw["grid_sample_coords"], from_aug["grid_sample_coords"])
    assert torch.allclose(from_raw["projected_points_2d"][0, 0, 0], torch.tensor([5.0, 3.0]))


def test_image_to_fpn_mapping_and_grid_sample_shape():
    sampler = GeometryCandidateSampler(OFFSETS, align_corners=False)
    # u=4.5, v=1.5 in a 10x4 padded image maps to feature (1.5, 0.5)
    candidates = torch.tensor([[[4.5, 1.5, 1.0]]]).expand(2, 9, 3).clone()
    matrices = torch.eye(4).repeat(6, 1, 1)
    output = sampler.project(candidates, matrices, torch.tensor([[4, 10]]).repeat(6, 1), (2, 4))
    assert torch.allclose(output["feature_coords"][0, 0, 0], torch.tensor([1.5, 0.5]))
    assert torch.allclose(output["grid_sample_coords"][0, 0, 0], torch.tensor([0.0, 0.0]))
    features = torch.randn(1, 6, 256, 2, 4)
    coords = output["grid_sample_coords"].unsqueeze(0)
    mask = output["camera_valid_mask"].unsqueeze(0)
    sampled = sample_candidate_features(features, coords, mask)
    assert sampled.shape == (1, 2, 9, 6, 256)


def test_padding_uses_valid_bounds_but_padded_fpn_scale():
    sampler = GeometryCandidateSampler(OFFSETS, align_corners=False)
    candidates = torch.tensor([[[4.5, 1.5, 1.0]]]).expand(1, 9, 3).clone()
    output = sampler.project(
        candidates,
        torch.eye(4).unsqueeze(0),
        torch.tensor([[4, 8]]),
        (2, 5),
        padded_image_shapes=torch.tensor([[4, 10]]),
    )
    assert output["camera_valid_mask"].all()
    assert torch.allclose(output["feature_coords"][0, 0, 0], torch.tensor([2.0, 0.5]))


def test_forward_contract_shapes_for_cpfpn():
    sampler = GeometryCandidateSampler(OFFSETS)
    prediction = PreviousPrediction(
        center_3d=torch.tensor([[5.0, 0.0, 10.0], [3.0, 1.0, 10.0]]),
        size_3d=torch.ones(2, 3),
        yaw=torch.zeros(2),
        velocity=None,
        score=torch.ones(2),
        label=torch.zeros(2, dtype=torch.long),
        timestamp=0.0,
    )
    output = sampler(
        prediction,
        torch.eye(4),
        0.5,
        torch.eye(4).repeat(6, 1, 1),
        torch.tensor([[256, 704]]).repeat(6, 1),
        (16, 44),
    )
    assert output["propagated_anchor"].shape == (2, 3)
    assert output["candidate_points_3d"].shape == (2, 9, 3)
    assert output["camera_valid_mask"].shape == (2, 9, 6)
    assert output["projected_points_2d"].shape == (2, 9, 6, 2)
    assert output["feature_coords"].shape == (2, 9, 6, 2)
