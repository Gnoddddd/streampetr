import math

import torch

from models.adapters import PreviousPrediction
from models.geocorr_recovery import (
    GeometryCandidateSampler,
    ObjectCentricCorrelationField,
    correspondence_distillation,
    find_center_candidate_index,
    masked_softmax,
    propagate_previous_points,
    reverse_current_points,
    sample_history_candidate_features,
)


OFFSETS = [
    (1.0, 0.0), (-1.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 1.0),
    (0.0, 0.0), (1.0, -1.0), (0.0, -1.0), (-1.0, 1.0),
]


def _transform():
    angle = 0.4
    transform = torch.eye(4)
    transform[:2, :2] = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    transform[:3, 3] = torch.tensor([4.0, -2.0, 0.5])
    return transform


def _tokens(batch=1, objects=2, channels=8):
    torch.manual_seed(4)
    current = torch.randn(batch, objects, 9, 6, channels)
    history = torch.randn(batch, objects, 9, 1, 6, channels)
    current_valid = torch.ones(batch, objects, 9, 6, dtype=torch.bool)
    history_valid = torch.ones(batch, objects, 9, 1, 6, dtype=torch.bool)
    return current, history, current_valid, history_valid


def _loss(module, clean, dirty, history, current_valid, history_valid):
    output = module(clean, dirty, history, current_valid, history_valid)
    loss, diagnostics = correspondence_distillation(
        output["teacher_logits"], output["student_logits"], output["valid_mask"],
        temperature=0.1, center_candidate_index=module.center_candidate_index,
    )
    return output, loss, diagnostics


def test_geometry_round_trip_with_motion_and_identity():
    previous = torch.tensor(
        [[[1.0, 2.0, 3.0], [2.0, -1.0, 4.0]],
         [[0.0, 3.0, 2.0], [5.0, 2.0, 1.0]]]
    )
    velocity = torch.tensor([[2.0, -0.5], [-1.0, 3.0]])
    current = propagate_previous_points(previous, _transform(), velocity, 0.3)
    recovered = reverse_current_points(current, _transform(), velocity, 0.3)
    assert torch.allclose(recovered, previous, atol=1e-6)
    identity_current = propagate_previous_points(previous, torch.eye(4), None, 1.0)
    identity_recovered = reverse_current_points(identity_current, torch.eye(4), None, 1.0)
    assert torch.equal(identity_current, previous)
    assert torch.equal(identity_recovered, previous)


def test_history_geometry_and_feature_sampling_preserve_time_dimension():
    sampler = GeometryCandidateSampler(OFFSETS)
    prediction = PreviousPrediction(
        center_3d=torch.tensor([[0.0, 0.0, 10.0]]), size_3d=torch.ones(1, 3),
        yaw=torch.zeros(1), velocity=torch.tensor([[1.0, 0.0]]), score=torch.ones(1),
        label=torch.zeros(1, dtype=torch.long), timestamp=0.0,
    )
    projection = torch.eye(4).repeat(6, 1, 1)
    geometry = sampler.forward_with_history(
        prediction, torch.eye(4), 0.5, projection,
        torch.tensor([[32, 32]]).repeat(6, 1), (4, 4), projection,
        torch.tensor([[32, 32]]).repeat(6, 1), (4, 4),
    )
    assert geometry["previous_candidate_points_3d"].shape == (1, 9, 1, 3)
    assert geometry["current_feature_coords"].shape == (1, 9, 6, 2)
    assert geometry["previous_feature_coords"].shape == (1, 9, 1, 6, 2)
    assert geometry["previous_grid_coords"].shape == (1, 9, 1, 6, 2)
    assert geometry["previous_valid_mask"].shape == (1, 9, 1, 6)
    features = torch.randn(1, 1, 6, 8, 4, 4)
    tokens = sample_history_candidate_features(
        features, geometry["previous_grid_coords"].unsqueeze(0),
        geometry["previous_valid_mask"].unsqueeze(0),
    )
    assert tokens.shape == (1, 1, 9, 1, 6, 8)


def test_center_candidate_is_detected_from_offsets_not_position_four():
    assert find_center_candidate_index(OFFSETS) == 5
    assert ObjectCentricCorrelationField(OFFSETS, feature_dim=8).center_candidate_index == 5


def test_correlation_and_probability_shapes_keep_t_equal_one():
    clean, history, current_valid, history_valid = _tokens(2, 3, 256)
    module = ObjectCentricCorrelationField(OFFSETS, feature_dim=256)
    output, _, diagnostics = _loss(module, clean, clean.clone(), history, current_valid, history_valid)
    expected = (2, 3, 6, 9, 1, 6)
    assert output["teacher_logits"].shape == expected
    assert output["student_logits"].shape == expected
    assert diagnostics["teacher_probs"].shape == expected
    assert diagnostics["student_probs"].shape == expected
    assert output["valid_mask"].shape == expected


def test_identical_clean_dirty_has_identical_distribution_and_zero_kl():
    clean, history, current_valid, history_valid = _tokens()
    module = ObjectCentricCorrelationField(OFFSETS, feature_dim=8)
    output, loss, diagnostics = _loss(module, clean, clean.clone(), history, current_valid, history_valid)
    assert torch.allclose(output["teacher_logits"], output["student_logits"], atol=1e-6)
    assert torch.allclose(diagnostics["teacher_probs"], diagnostics["student_probs"], atol=1e-6)
    assert loss.abs() < 1e-6


def test_dirty_feature_perturbation_increases_kl():
    clean, history, current_valid, history_valid = _tokens()
    module = ObjectCentricCorrelationField(OFFSETS, feature_dim=8)
    _, identical, _ = _loss(module, clean, clean.clone(), history, current_valid, history_valid)
    dirty = clean.clone()
    dirty[:, :, module.center_candidate_index] *= -1
    _, perturbed, _ = _loss(module, clean, dirty, history, current_valid, history_valid)
    assert perturbed > identical + 1e-4


def test_invalid_keys_are_zero_and_valid_probabilities_normalize():
    logits = torch.randn(1, 1, 2, 9, 1, 6)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[..., 1::2] = False
    probabilities, query_valid = masked_softmax(logits, mask, temperature=0.1)
    assert query_valid.all()
    assert torch.count_nonzero(probabilities.masked_select(~mask)) == 0
    assert torch.allclose(probabilities.flatten(start_dim=3).sum(-1), torch.ones(1, 1, 2))


def test_invalid_current_query_and_object_are_excluded_by_field_mask():
    clean, history, current_valid, history_valid = _tokens(1, 2, 8)
    module = ObjectCentricCorrelationField(OFFSETS, feature_dim=8)
    current_valid[:, 0, module.center_candidate_index, 2] = False
    object_valid = torch.tensor([[True, False]])
    output = module(
        clean, clean.clone(), history, current_valid, history_valid, object_valid
    )
    assert not output["valid_mask"][0, 0, 2].any()
    assert not output["valid_mask"][0, 1].any()
    assert output["valid_mask"][0, 0, 0].all()


def test_all_invalid_query_is_finite_excluded_and_differentiable():
    teacher = torch.randn(1, 1, 2, 9, 1, 6)
    student = torch.randn(1, 1, 2, 9, 1, 6, requires_grad=True)
    mask = torch.zeros_like(teacher, dtype=torch.bool)
    loss, diagnostics = correspondence_distillation(teacher, student, mask, 0.1, 5)
    assert torch.isfinite(loss)
    assert diagnostics["num_valid_queries"].item() == 0
    assert torch.count_nonzero(diagnostics["teacher_probs"]) == 0
    assert torch.count_nonzero(diagnostics["student_probs"]) == 0
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_gradient_isolation_and_teacher_detach():
    clean, history, current_valid, history_valid = _tokens()
    clean.requires_grad_()
    dirty = clean.detach().clone().requires_grad_()
    history.requires_grad_()
    module = ObjectCentricCorrelationField(OFFSETS, feature_dim=8)
    _, loss, diagnostics = _loss(module, clean, dirty, history, current_valid, history_valid)
    assert not diagnostics["teacher_probs"].requires_grad
    loss.backward()
    assert clean.grad is None and dirty.grad is None and history.grad is None
    for parameter in module.adapter.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_synthetic_center_correspondence_has_high_center_mass():
    channels = 8
    center = find_center_candidate_index(OFFSETS)
    clean = torch.zeros(1, 1, 9, 1, channels)
    clean[:, :, center, :, 0] = 1.0
    history = torch.zeros(1, 1, 9, 1, 2, channels)
    history[..., 1] = 1.0
    history[:, :, center] = 0.0
    history[:, :, center, :, :, 0] = 1.0
    valid_current = torch.ones(1, 1, 9, 1, dtype=torch.bool)
    valid_history = torch.ones(1, 1, 9, 1, 2, dtype=torch.bool)
    module = ObjectCentricCorrelationField(OFFSETS, feature_dim=channels)
    _, _, diagnostics = _loss(module, clean, clean.clone(), history, valid_current, valid_history)
    assert diagnostics["teacher_center_candidate_mass"] > 0.9
    assert diagnostics["top1_agreement"] == 1
