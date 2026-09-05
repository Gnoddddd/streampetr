import torch

from analysis.ctep_objective import ctep_term, disabled_detection_loss


def test_ctep_active_gradient_increases_target_score():
    reference = torch.tensor(0.8, requires_grad=True)
    target = torch.tensor(0.0, requires_grad=True)
    loss = ctep_term(reference, target)
    grad = torch.autograd.grad(loss, target)[0]
    assert grad.item() < 0
    assert reference.grad is None


def test_ctep_inactive_is_exact_zero():
    reference = torch.tensor(0.2)
    target = torch.tensor(2.0, requires_grad=True)
    loss = ctep_term(reference, target)
    assert loss.item() == 0.0


def test_disabled_returns_identical_tensor():
    loss = torch.tensor(3.0, requires_grad=True)
    assert disabled_detection_loss(loss) is loss
