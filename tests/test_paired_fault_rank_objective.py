import torch

from models.paired_fault_rank import (
    paired_margin_preservation_loss,
    select_paired_margin_events,
)


def _case():
    clean = torch.full((4, 2), -6.0, requires_grad=True)
    fault = torch.full((4, 2), -6.0, requires_grad=True)
    with torch.no_grad():
        clean[0, 0] = 3.0       # Clean q+ is query 0.
        clean[1, 1] = 2.0       # K=2 boundary competitor.
        fault[2, 0] = -1.0      # Fault q+ is replacement query 2.
        fault[1, 1] = 2.0       # Detached K boundary.
        fault[3, 1] = 1.0
    clean_boxes = torch.tensor([[0., 0, 0], [8, 0, 0], [1., 0, 0], [9, 0, 0]])
    fault_boxes = torch.tensor([[8., 0, 0], [8, 0, 0], [0.5, 0, 0], [9, 0, 0]])
    centers = torch.tensor([[0., 0, 0]])
    labels = torch.tensor([0])
    return clean, fault, clean_boxes, fault_boxes, centers, labels


def test_selection_is_gt_pool_based_and_does_not_bind_query_id():
    clean, fault, clean_boxes, fault_boxes, centers, labels = _case()
    events, _ = select_paired_margin_events(
        clean, clean_boxes, fault, fault_boxes, centers, labels, topk=2
    )
    event = events[0]
    assert event["clean_query"] == 0
    assert event["fault_query"] == 2
    assert not event["same_query_id"]
    assert event["boundary_crossing"] and event["collapse_eligible"]


def test_loss_detaches_clean_reference_and_kth_boundaries():
    clean, fault, clean_boxes, fault_boxes, centers, labels = _case()
    events, _ = select_paired_margin_events(
        clean, clean_boxes, fault, fault_boxes, centers, labels, topk=2
    )
    loss, details = paired_margin_preservation_loss(fault, events, delta=0.10)
    loss.backward()
    assert details[0]["nonzero"]
    assert clean.grad is None
    assert fault.grad[2, 0].abs() > 0
    assert fault.grad[1, 1] == 0  # Actual Kth boundary is stop-gradient.


def test_generic_clean_hard_and_non_degrading_cases_are_not_eligible():
    clean, fault, clean_boxes, fault_boxes, centers, labels = _case()
    with torch.no_grad():
        clean[0, 0] = -2.0
        fault[2, 0] = 3.0
    events, _ = select_paired_margin_events(
        clean, clean_boxes, fault, fault_boxes, centers, labels, topk=2
    )
    assert not events[0]["collapse_eligible"]
    loss, details = paired_margin_preservation_loss(fault, events)
    assert loss == 0 and details == []


def test_disabled_path_is_zero_and_does_not_change_inputs():
    clean, fault, clean_boxes, fault_boxes, centers, labels = _case()
    before = fault.detach().clone()
    events, _ = select_paired_margin_events(
        clean, clean_boxes, fault, fault_boxes, centers, labels, topk=2
    )
    loss, details = paired_margin_preservation_loss(fault, events, enabled=False)
    assert loss == 0 and details == []
    assert torch.equal(fault.detach(), before)


def test_amp_objective_and_gradient_are_finite():
    clean, fault, clean_boxes, fault_boxes, centers, labels = _case()
    fault = fault.detach().half().requires_grad_(True)
    events, _ = select_paired_margin_events(
        clean.half(), clean_boxes.half(), fault, fault_boxes.half(),
        centers.half(), labels, topk=2,
    )
    loss, _ = paired_margin_preservation_loss(fault, events)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(fault.grad).all()
