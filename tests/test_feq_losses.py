import torch
import inspect

from models.feq_losses import (
    ABSENT, PRESENT, UNOBSERVED, adjacent_survival_loss,
    deployment_topk_boundary, geometric_auxiliary_cost,
    greedy_auxiliary_assignment, ranking_loss, supervision_weights,
    topk_boundary_loss,
)


def test_present_weight_is_one():
    assert supervision_weights(torch.tensor([PRESENT]), torch.tensor([False])).item() == 1


def test_absent_weight_is_zero():
    assert supervision_weights(torch.tensor([ABSENT]), torch.tensor([True])).item() == 0


def test_unobserved_reliable_weight():
    assert torch.equal(supervision_weights(torch.tensor([UNOBSERVED]), torch.tensor([True])), torch.tensor([.35]))


def test_unobserved_without_history_is_zero():
    assert supervision_weights(torch.tensor([UNOBSERVED]), torch.tensor([False])).item() == 0


def test_auxiliary_assignment_caps_three():
    sets, _ = greedy_auxiliary_assignment(torch.arange(10.).view(5, 2), torch.tensor([-1, -1]), torch.tensor([1, 1]), 3)
    assert all(len(values) <= 3 for values in sets)


def test_auxiliary_query_is_unique():
    sets, _ = greedy_auxiliary_assignment(torch.zeros(6, 2), torch.tensor([-1, -1]), torch.tensor([1, 1]), 3)
    values = sum(sets, [])
    assert len(values) == len(set(values))


def test_main_query_is_never_auxiliary():
    sets, _ = greedy_auxiliary_assignment(torch.zeros(4, 1), torch.tensor([0]), torch.tensor([1]), 3)
    assert 0 not in sets[0]


def test_ineligible_gt_receives_no_auxiliary():
    sets, _ = greedy_auxiliary_assignment(torch.zeros(4, 1), torch.tensor([-1]), torch.tensor([0]), 3)
    assert sets == [[]]


def test_competition_is_reported():
    _, conflicts = greedy_auxiliary_assignment(torch.zeros(2, 2), torch.tensor([-1, -1]), torch.tensor([1, 1]), 1)
    assert conflicts > 0


def _ranking_inputs(pos=1.0, neg=0.0):
    logits = torch.tensor([[pos], [neg], [-2.0]], requires_grad=True)
    labels = torch.tensor([0]); centers = torch.tensor([[0., 0, 0], [.1, 0, 0], [5., 0, 0]])
    gt = torch.zeros(1, 3); weights = torch.ones(1)
    return logits, labels, centers, gt, [[0]], weights


def test_ranking_zero_when_margin_satisfied():
    loss, _ = ranking_loss(*_ranking_inputs(1.0, 0.0), margin=.2)
    assert loss.item() == 0


def test_ranking_margin_formula():
    loss, _ = ranking_loss(*_ranking_inputs(0.0, 0.0), margin=.2)
    assert torch.allclose(loss, torch.tensor(.2))


def test_ranking_negative_excludes_positive():
    loss, _ = ranking_loss(*_ranking_inputs(10.0, 0.0), margin=.2)
    assert loss.item() == 0


def test_ranking_backward_is_finite():
    args = _ranking_inputs(0.0, 0.0); loss, _ = ranking_loss(*args, margin=.2)
    loss.backward(); assert torch.isfinite(args[0].grad).all()


def test_survival_only_adjacent_layers():
    logits = [torch.tensor([[2.]], requires_grad=True), torch.tensor([[0.]], requires_grad=True), torch.tensor([[-2.]], requires_grad=True)]
    boxes = [torch.zeros(1, 3) for _ in logits]
    loss, comparisons = adjacent_survival_loss(logits, boxes, torch.tensor([0]), torch.zeros(1, 3), [[[0]], [[0]], [[0]]], torch.ones(1), .05)
    assert comparisons == 2 and loss.item() > 0


def test_survival_skips_missing_previous_candidate():
    logits = [torch.zeros(1, 1, requires_grad=True), torch.zeros(1, 1, requires_grad=True)]
    loss, comparisons = adjacent_survival_loss(logits, [torch.zeros(1, 3)] * 2, torch.tensor([0]), torch.zeros(1, 3), [[[]], [[0]]], torch.ones(1))
    assert comparisons == 0 and loss.item() == 0


def test_survival_no_cross_gt_pollution():
    logits = [torch.zeros(2, 2, requires_grad=True), torch.zeros(2, 2, requires_grad=True)]
    boxes = [torch.zeros(2, 3), torch.zeros(2, 3)]
    _, comparisons = adjacent_survival_loss(logits, boxes, torch.tensor([0, 1]), torch.zeros(2, 3), [[[0], []], [[0], [1]]], torch.ones(2))
    assert comparisons == 1


def test_survival_backward_is_finite():
    logits = [torch.tensor([[2.]], requires_grad=True), torch.tensor([[0.]], requires_grad=True)]
    loss, _ = adjacent_survival_loss(logits, [torch.zeros(1, 3)] * 2, torch.tensor([0]), torch.zeros(1, 3), [[[0]], [[0]]], torch.ones(1))
    loss.backward(); assert all(torch.isfinite(value.grad).all() for value in logits)


def test_fp16_losses_are_finite():
    logits, labels, centers, gt, positives, weights = _ranking_inputs(0.0, 0.0)
    loss, _ = ranking_loss(logits.half(), labels, centers.half(), gt.half(), positives, weights.half())
    assert torch.isfinite(loss)


def test_geometric_cost_does_not_accept_or_depend_on_logits():
    boxes = torch.tensor([[0., 0, 0] + [0.] * 7, [1., 0, 0] + [0.] * 7])
    gt = torch.zeros(1, 10)
    first = geometric_auxiliary_cost(boxes, gt, [-2, -2, -2, 2, 2, 2])
    # Classification changes have no input path into geometry-first selection.
    logits_a = torch.tensor([[100.], [-100.]])
    logits_b = -logits_a
    assert torch.equal(first, geometric_auxiliary_cost(boxes, gt, [-2, -2, -2, 2, 2, 2]))
    assert not torch.equal(logits_a, logits_b)


def test_geometric_assignment_still_caps_three_and_is_unique():
    boxes = torch.randn(10, 10); gt = torch.randn(2, 10)
    cost = geometric_auxiliary_cost(boxes, gt, [-10, -10, -5, 10, 10, 5])
    assigned, _ = greedy_auxiliary_assignment(cost, torch.tensor([0, 1]), torch.ones(2, dtype=torch.bool), 3)
    flat = sum(assigned, [])
    assert all(len(group) <= 3 for group in assigned) and len(flat) == len(set(flat))


def test_deployment_boundary_matches_flattened_topk():
    logits = torch.tensor([[0., 3.], [2., 1.], [-1., 4.]])
    boundary, indices = deployment_topk_boundary(logits, 3)
    expected = torch.topk(logits.sigmoid().flatten(), 3)
    assert torch.equal(indices, expected.indices)
    assert torch.equal(boundary, expected.values[-1])


def test_deployment_boundary_is_stop_gradient():
    logits = torch.tensor([[3.], [2.], [-2.]], requires_grad=True)
    loss, details = topk_boundary_loss(logits, torch.tensor([0]), [[2]], torch.ones(1), 1, .1)
    loss.backward()
    assert details[0]["s_k"].requires_grad is False
    assert logits.grad[0].item() == 0 and logits.grad[2].item() != 0


def test_boundary_positive_only_comes_from_gt_candidate_set():
    logits = torch.tensor([[10.], [1.], [2.]])
    _, details = topk_boundary_loss(logits, torch.tensor([0]), [[1, 2]], torch.ones(1), 1, .1)
    assert int(details[0]["positive_query"]) == 2
    assert torch.equal(details[0]["s_pos"], logits[2, 0].sigmoid())


def test_absent_has_no_boundary_term():
    logits = torch.zeros(2, 1, requires_grad=True)
    weights = supervision_weights(torch.tensor([ABSENT]), torch.tensor([True]))
    loss, details = topk_boundary_loss(logits, torch.tensor([0]), [[0]], weights, 1)
    assert loss.item() == 0 and details == []


def test_unobserved_boundary_weight_is_point_three_five():
    logits = torch.tensor([[-2.], [2.]])
    weights = supervision_weights(torch.tensor([UNOBSERVED]), torch.tensor([True]))
    loss, details = topk_boundary_loss(logits, torch.tensor([0]), [[0]], weights, 1, .1)
    expected = .35 * torch.relu(torch.tensor(.1) - logits[0, 0].sigmoid() + logits[1, 0].sigmoid())
    assert torch.allclose(loss, expected) and details[0]["weighted_loss"] > 0


def test_boundary_amp_loss_and_gradient_are_finite():
    logits = torch.tensor([[-2.], [2.]], dtype=torch.float16, requires_grad=True)
    loss, _ = topk_boundary_loss(logits, torch.tensor([0]), [[0]], torch.ones(1).half(), 1)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_head_disabled_path_returns_before_feq_objectives():
    from models.feq_head import FEQStreamPETRHead
    source = inspect.getsource(FEQStreamPETRHead.loss)
    guard = source.index("if not (self.training and self.enable_feq_core)")
    objective = source.index("topk_boundary_loss(")
    assert guard < objective


def test_head_inference_does_not_cache_feq_context():
    from models.feq_head import FEQStreamPETRHead
    source = inspect.getsource(FEQStreamPETRHead.forward)
    assert "if self.training and self.enable_feq_core" in source
    assert 'self._feq_context = None' in source


def test_legacy_ranking_is_not_called_by_head():
    from models.feq_head import FEQStreamPETRHead
    source = inspect.getsource(FEQStreamPETRHead.loss)
    assert "ranking_loss(" not in source
    assert "with torch.no_grad():" in source
