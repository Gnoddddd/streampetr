import torch

from models.adapters import PreviousPrediction
from models.geocorr_recovery import (
    GeoCorrStage3BOutput,
    GeoCorrStage3BRecovery,
    HistoricalEvidenceRetriever,
    ObjectCentricCorrelationField,
    RecoveryMLP,
    SparseBilinearResidualWriteback,
    masked_softmax,
    normalized_entropy_confidence,
    recovery_feature_loss,
    select_top_predictions,
    topk_prediction_indices,
)


def _retrieval_inputs():
    probabilities = torch.tensor(
        [[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]]
    ).reshape(1, 1, 2, 3, 1, 1)
    history = torch.tensor(
        [[1.0, 0.0], [3.0, 0.0], [9.0, 0.0]]
    ).reshape(1, 1, 3, 1, 1, 2)
    valid = torch.ones_like(probabilities, dtype=torch.bool)
    return probabilities, history, valid


def test_retrieval_shape_and_weighted_correctness():
    probabilities, history, valid = _retrieval_inputs()
    output = HistoricalEvidenceRetriever()(probabilities, history, valid)

    assert output["historical_retrieved_feature"].shape == (1, 1, 2, 2)
    assert torch.allclose(
        output["historical_retrieved_feature"][0, 0, :, 0],
        torch.tensor([5.6, 7.0]),
    )


def test_retrieval_masks_and_renormalizes_valid_support():
    probabilities, history, valid = _retrieval_inputs()
    valid[..., 2, :, :] = False
    output = HistoricalEvidenceRetriever()(probabilities, history, valid)

    expected = torch.tensor([2.2, 7.0 / 3.0])
    assert torch.allclose(
        output["historical_retrieved_feature"][0, 0, :, 0], expected
    )
    assert torch.allclose(
        output["normalized_probabilities"].sum(dim=(3, 4, 5)),
        torch.ones(1, 1, 2),
    )
    assert torch.count_nonzero(output["normalized_probabilities"][..., 2, :, :]) == 0


def test_all_invalid_retrieval_and_confidence_are_finite_zero():
    probabilities, history, valid = _retrieval_inputs()
    valid.zero_()
    output = HistoricalEvidenceRetriever()(probabilities, history, valid)

    assert torch.isfinite(output["historical_retrieved_feature"]).all()
    assert torch.count_nonzero(output["historical_retrieved_feature"]) == 0
    assert torch.count_nonzero(output["confidence"]) == 0
    assert not output["query_valid"].any()


def test_entropy_confidence_range_and_peaked_exceeds_uniform():
    probabilities = torch.tensor(
        [[1.0, 0.0], [0.5, 0.5]]
    ).reshape(1, 1, 2, 1, 1, 2)
    valid = torch.ones_like(probabilities, dtype=torch.bool)
    confidence, _, query_valid = normalized_entropy_confidence(probabilities, valid)

    assert query_valid.all()
    assert ((confidence >= 0) & (confidence <= 1)).all()
    assert confidence[0, 0, 0, 0] > confidence[0, 0, 1, 0]
    assert torch.allclose(confidence[0, 0, 0], torch.ones(1))
    assert torch.allclose(confidence[0, 0, 1], torch.zeros(1))


def test_singleton_support_confidence_is_stable_one():
    probabilities = torch.tensor([0.2, 0.8]).reshape(1, 1, 1, 1, 1, 2)
    valid = torch.tensor([False, True]).reshape(1, 1, 1, 1, 1, 2)
    confidence, normalized, query_valid = normalized_entropy_confidence(
        probabilities, valid
    )

    assert query_valid.item()
    assert confidence.item() == 1.0
    assert torch.equal(normalized, valid.to(normalized.dtype))


def test_recovery_shape_and_zero_init_identity():
    q_dirty = torch.randn(2, 3, 4, 8)
    history = torch.randn_like(q_dirty)
    confidence = torch.rand(2, 3, 4, 1)
    output = RecoveryMLP(8)(q_dirty, history, confidence)

    assert output["delta_q"].shape == q_dirty.shape
    assert torch.count_nonzero(output["delta_q"]) == 0
    assert torch.equal(output["q_recovered"], q_dirty)


def test_recovery_feature_loss_detaches_clean_teacher():
    q_recovered = torch.randn(1, 2, 3, 4, requires_grad=True)
    q_clean = torch.randn(1, 2, 3, 4, requires_grad=True)
    loss = recovery_feature_loss(
        q_recovered, q_clean, torch.ones(1, 2, 3, dtype=torch.bool)
    )
    loss.backward()

    assert q_recovered.grad is not None
    assert q_clean.grad is None


def test_recovery_feature_loss_value_and_masked_reduction():
    q_recovered = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    q_clean = torch.tensor([[[[0.0, 1.0], [0.0, -1.0]]]])
    mask = torch.tensor([[[True, False]]])

    loss = recovery_feature_loss(q_recovered, q_clean, mask)
    per_query = recovery_feature_loss(q_recovered, q_clean, mask, reduction="none")

    assert torch.allclose(loss, torch.tensor(1.0))
    assert torch.allclose(per_query, torch.tensor([[[1.0, 0.0]]]))


def test_recovery_feature_loss_all_invalid_is_safe_and_differentiable():
    q_recovered = torch.zeros(1, 1, 1, 4, requires_grad=True)
    q_clean = torch.zeros_like(q_recovered)
    loss = recovery_feature_loss(
        q_recovered, q_clean, torch.zeros(1, 1, 1, dtype=torch.bool)
    )
    loss.backward()

    assert loss.item() == 0.0
    assert torch.isfinite(q_recovered.grad).all()


def _writeback_inputs(objects=1):
    fpn = torch.zeros(1, 2, 3, 4, 5)
    residual = torch.ones(1, objects, 2, 3)
    coords = torch.tensor([1.0, 2.0]).expand(1, objects, 2, 2).clone()
    valid = torch.ones(1, objects, 2, dtype=torch.bool)
    confidence = torch.ones(1, objects, 2, 1)
    return fpn, residual, coords, valid, confidence


def test_sparse_writeback_shape_and_zero_residual_identity():
    fpn, residual, coords, valid, confidence = _writeback_inputs()
    output = SparseBilinearResidualWriteback()(
        fpn, residual.zero_(), coords, valid, confidence
    )

    assert output.shape == fpn.shape
    assert torch.equal(output, fpn)


def test_sparse_writeback_invalid_projection_does_not_write():
    fpn, residual, coords, valid, confidence = _writeback_inputs()
    valid.zero_()
    output = SparseBilinearResidualWriteback()(
        fpn, residual, coords, valid, confidence
    )

    assert torch.equal(output, fpn)


def test_sparse_writeback_multiple_queries_use_weighted_mean():
    fpn, residual, coords, valid, confidence = _writeback_inputs(objects=2)
    residual[:, 0] = 1.0
    residual[:, 1] = 3.0
    output = SparseBilinearResidualWriteback()(
        fpn, residual, coords, valid, confidence
    )

    assert torch.allclose(output[0, :, :, 2, 1], torch.full((2, 3), 2.0))
    assert torch.count_nonzero(output) == 6


def _prediction(count=60):
    scores = torch.linspace(0.0, 1.0, count)
    return PreviousPrediction(
        center_3d=torch.arange(count * 3, dtype=torch.float32).reshape(count, 3),
        size_3d=torch.ones(count, 3),
        yaw=torch.zeros(count),
        velocity=torch.zeros(count, 2),
        score=scores,
        label=torch.arange(count),
        timestamp=1.0,
    )


def test_top25_selection_is_default_and_score_sorted():
    prediction = _prediction()
    indices = topk_prediction_indices(prediction.score)
    selected = select_top_predictions(prediction)

    assert indices.numel() == 25
    assert selected.score.shape == (25,)
    assert torch.equal(selected.score, prediction.score[indices])
    assert torch.all(selected.score[:-1] >= selected.score[1:])


def test_top50_and_all_are_configurable():
    prediction = _prediction()

    assert select_top_predictions(prediction, 50).score.numel() == 50
    assert select_top_predictions(prediction, "all").score.numel() == 60


def test_structured_forward_contract_and_two_level_clean_identity():
    batch, objects, views, candidates, channels = 1, 2, 2, 3, 4
    q_dirty = torch.randn(batch, objects, views, channels)
    q_clean = torch.randn_like(q_dirty)
    history = torch.randn(batch, objects, candidates, 1, 1, channels)
    probabilities = torch.softmax(
        torch.randn(batch, objects, views, candidates, 1, 1), dim=3
    )
    valid = torch.ones_like(probabilities, dtype=torch.bool)
    fpn = torch.randn(batch, views, channels, 4, 5)
    coords = torch.tensor([1.0, 2.0]).expand(batch, objects, views, 2).clone()
    projected_valid = torch.ones(batch, objects, views, dtype=torch.bool)
    module = GeoCorrStage3BRecovery(feature_dim=channels)

    output = module(
        q_dirty=q_dirty,
        q_clean=q_clean,
        historical_features=history,
        p_student=probabilities,
        correspondence_valid_mask=valid,
        p_teacher=probabilities.clone(),
        previous_predictions=module.select_historical_anchors(_prediction(30)),
        current_clean_fpn=fpn.clone(),
        current_dirty_fpn=fpn,
        projected_center_coords=coords,
        projected_center_valid=projected_valid,
    )

    assert isinstance(output, GeoCorrStage3BOutput)
    assert output.historical_retrieved_feature.shape == q_dirty.shape
    assert output.confidence.shape == q_dirty.shape[:-1] + (1,)
    assert torch.equal(output.q_recovered, q_dirty)
    assert torch.equal(output.recovered_current_fpn, fpn)
    assert output.previous_predictions.score.numel() == 25


def test_existing_stage1_stage2_public_apis_remain_usable():
    offsets = [
        (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0),
        (0, 1), (1, -1), (1, 0), (1, 1),
    ]
    field = ObjectCentricCorrelationField(offsets, feature_dim=4)
    logits = torch.zeros(1, 1, 1, 9, 1, 1)
    probabilities, query_valid = masked_softmax(
        logits, torch.ones_like(logits, dtype=torch.bool), 0.1
    )

    assert field.center_candidate_index == 4
    assert probabilities.shape == logits.shape
    assert query_valid.item()
