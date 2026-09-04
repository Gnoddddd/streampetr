import torch

from models.care3d import (
    CARE3DCore,
    CARE3DStateEncoder,
    CounterfactualVulnerabilityHead,
    CounterfactualVulnerabilityLoss,
    SparseEvidenceRouter,
)


def test_state_encoder_accepts_detector_agnostic_inputs():
    encoder = CARE3DStateEncoder(
        object_dim=16,
        num_cameras=6,
        hidden_dim=32,
        state_dim=12,
        decision_dim=3,
        use_temporal=True,
    )
    object_features = torch.randn(2, 5, 16)
    camera_support = torch.rand(2, 5, 6)
    camera_quality = torch.rand(2, 6)
    temporal = torch.randn(2, 5, 16)
    decision = torch.randn(2, 5, 3)

    state = encoder(
        object_features,
        camera_support,
        camera_quality,
        temporal_features=temporal,
        decision_features=decision,
    )
    assert state.shape == (2, 5, 12)
    assert torch.isfinite(state).all()


def test_vulnerability_head_outputs_protocol_vector():
    head = CounterfactualVulnerabilityHead(
        state_dim=12,
        num_protocols=3,
        hidden_dim=16,
    )
    output = head(torch.randn(2, 7, 12))
    assert output["vulnerability"].shape == (2, 7, 3)
    assert output["boundary_crossing_logits"].shape == (2, 7, 3)
    assert (output["vulnerability"] >= 0).all()


def test_vulnerability_loss_masks_invalid_objects():
    head = CounterfactualVulnerabilityHead(8, 3, hidden_dim=16)
    prediction = head(torch.randn(1, 4, 8))
    target_drop = torch.rand(1, 4, 3)
    target_crossing = torch.randint(0, 2, (1, 4, 3)).float()
    valid = torch.tensor([[True, True, False, False]])
    criterion = CounterfactualVulnerabilityLoss()
    losses = criterion(prediction, target_drop, target_crossing, valid)
    assert set(losses) == {
        "loss_care3d_vulnerability",
        "loss_care3d_crossing",
        "loss_care3d",
    }
    assert torch.isfinite(losses["loss_care3d"]).all()


def test_sparse_router_is_exact_identity_at_initialization():
    router = SparseEvidenceRouter(
        object_dim=16,
        source_dim=10,
        num_protocols=3,
        topk_sources=2,
        hidden_dim=16,
    )
    objects = torch.randn(2, 5, 16)
    sources = torch.randn(2, 5, 4, 10)
    reliability = torch.rand(2, 5, 4)
    vulnerability = torch.rand(2, 5, 3)
    output = router(objects, sources, reliability, vulnerability)

    assert torch.equal(output["enhanced_features"], objects)
    assert output["route_weights"].shape == (2, 5, 4)
    assert output["route_indices"].shape == (2, 5, 2)
    assert torch.allclose(
        output["route_weights"].sum(-1),
        torch.ones_like(output["route_weights"].sum(-1)),
    )


def test_sparse_router_handles_no_reliable_source_without_nan():
    router = SparseEvidenceRouter(8, 8, 3, topk_sources=2)
    objects = torch.randn(1, 2, 8)
    sources = torch.randn(1, 2, 3, 8)
    reliability = torch.zeros(1, 2, 3)
    vulnerability = torch.rand(1, 2, 3)
    output = router(objects, sources, reliability, vulnerability)
    assert torch.equal(output["enhanced_features"], objects)
    assert torch.equal(output["route_weights"], torch.zeros_like(output["route_weights"]))
    assert torch.isfinite(output["route_residual"]).all()


def test_core_p0_does_not_change_detector_features():
    core = CARE3DCore(
        object_dim=16,
        num_cameras=6,
        num_protocols=3,
        hidden_dim=32,
        state_dim=12,
        decision_dim=2,
        use_temporal=False,
        enable_routing=False,
    )
    objects = torch.randn(1, 6, 16)
    support = torch.rand(1, 6, 6)
    quality = torch.rand(1, 6)
    decision = torch.randn(1, 6, 2)
    output = core(
        object_features=objects,
        camera_support=support,
        camera_quality=quality,
        decision_features=decision,
    )
    assert torch.equal(output["enhanced_features"], objects)
    assert output["vulnerability"].shape == (1, 6, 3)
