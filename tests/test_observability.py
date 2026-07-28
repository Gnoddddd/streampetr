import torch

from models.observability_head import GeometricObservabilityHead


def camera_matrix():
    matrix = torch.eye(4)
    matrix[0, 0] = 100.0
    matrix[1, 1] = 100.0
    matrix[0, 2] = 50.0
    matrix[1, 2] = 50.0
    return matrix


def test_geometry_and_camera_online_gate():
    head = GeometricObservabilityHead(num_cameras=6, boundary_softness=2.0)
    query = torch.tensor([[[0.0, 0.0, 5.0]]])
    matrices = camera_matrix().view(1, 1, 4, 4).repeat(1, 6, 1, 1)
    output = head(query, matrices, (100, 100))
    assert output["observability"].shape == (1, 1)
    assert output["observability"].item() > 0.95
    offline = head(
        query,
        matrices,
        (100, 100),
        camera_online_mask=torch.zeros(1, 6),
    )
    assert offline["observability"].item() < 1e-5


def test_fresh_ratio_and_effective_count():
    head = GeometricObservabilityHead(num_cameras=6, boundary_softness=2.0)
    query = torch.tensor([[[0.0, 0.0, 5.0]]])
    matrices = camera_matrix().view(1, 1, 4, 4).repeat(1, 6, 1, 1)
    fresh = torch.tensor([[1, 0, 0, 0, 0, 0]], dtype=torch.float32)
    output = head(query, matrices, (100, 100), camera_fresh_mask=fresh)
    assert 0.0 < output["fresh_ratio"].item() < 1.0
    assert 1.0 <= output["effective_count"].item() <= 6.0


def test_observability_supports_half_precision_correlation_math():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA is required for the float16 observability test")

    device = torch.device("cuda")
    head = GeometricObservabilityHead(num_cameras=2).to(device)

    xyz = torch.tensor(
        [[[0.0, 0.0, 5.0]]],
        dtype=torch.float16,
        device=device,
    )
    lidar2img = (
        torch.eye(4, dtype=torch.float16, device=device)
        .view(1, 1, 4, 4)
        .repeat(1, 2, 1, 1)
    )

    output = head(xyz, lidar2img, (100, 100))

    assert output["effective_count"].is_cuda
    assert torch.isfinite(output["effective_count"]).all()
    assert output["effective_count"].dtype == torch.float16
