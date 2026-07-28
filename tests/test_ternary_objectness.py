import torch

from models.ternary_objectness import (
    ABSENT,
    PRESENT,
    UNOBSERVED,
    ObservabilityConditionedTernaryLoss,
    build_ternary_targets,
    observability_conditioned_background_weights,
)


def test_soft_unmatched_target_matches_formula():
    observability = torch.tensor([0.9, 0.2, 0.0, 1.0])
    targets, weights = build_ternary_targets(
        4,
        torch.tensor([0]),
        torch.tensor([1, 2, 3]),
        observability,
        torch.float32,
        torch.device("cpu"),
    )
    assert targets[0, PRESENT] == 1
    assert torch.allclose(targets[1], torch.tensor([0.0, 0.2, 0.8]))
    assert targets[2, ABSENT] == 0
    assert targets[2, UNOBSERVED] == 1
    assert weights.sum() == 4


def test_background_weight_is_observability_conditioned():
    weights = torch.ones(4)
    output = observability_conditioned_background_weights(
        weights,
        torch.tensor([1, 2]),
        torch.tensor([1.0, 0.25, 0.0, 1.0]),
    )
    assert output[1] == 0.25
    assert output[2] == 0.0


def test_ternary_loss_is_finite():
    logits = torch.randn(4, 3, requires_grad=True)
    targets = torch.softmax(torch.randn(4, 3), dim=-1)
    loss = ObservabilityConditionedTernaryLoss()(logits, targets, torch.ones(4))
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
