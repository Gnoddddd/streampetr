import torch

from models.object_evidence.observability import camera_support, observability_gap


def test_gap_range_and_unsupported_object():
    support = torch.tensor([[0.2, 0.8], [0.0, 0.0]])
    gap, valid = observability_gap(support, torch.tensor([0.3, 1.0]))
    assert torch.all((gap >= 0) & (gap <= 1))
    assert gap[1].item() == 0
    assert valid.tolist() == [True, False]


def test_fault_camera_without_gt_support_has_zero_gap():
    gap, valid = observability_gap(torch.tensor([[1.0, 0.0]]), torch.tensor([0.0, 1.0]))
    assert valid.item()
    assert torch.isclose(gap, torch.zeros_like(gap), atol=1e-6).all()


def test_only_supporting_camera_crash_has_unit_gap():
    gap, _ = observability_gap(torch.tensor([[0.0, 1.0]]), torch.tensor([0.0, 1.0]))
    assert torch.isclose(gap, torch.ones_like(gap), atol=2e-6).all()


def test_projection_counts_eight_corners_and_center():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 2.0, 4.0, 2.0, 0.0]])
    projection = torch.eye(4).unsqueeze(0)
    projection[0, 0, 0] = 10
    projection[0, 1, 1] = 10
    projection[0, 0, 2] = 10
    projection[0, 1, 2] = 10
    support = camera_support(boxes, projection, [(20, 20)])
    assert support.shape == (1, 1)
    assert support.item() == 1.0
