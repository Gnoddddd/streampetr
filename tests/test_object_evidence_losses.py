import torch

from models.object_evidence.losses import object_evidence_loss, privileged_geometry_loss
from models.object_evidence.privileged_geometry.projector import PrivilegedGeometryProjector


def test_clean_is_detached_and_fault_receives_oe_gradient():
    torch.manual_seed(1)
    clean = torch.randn(3, 256, requires_grad=True)
    fault = torch.randn(3, 256, requires_grad=True)
    loss = object_evidence_loss(
        fault, clean, torch.ones(3), torch.ones(3, dtype=torch.bool)
    )
    loss.backward()
    assert clean.grad is None
    assert fault.grad is not None and fault.grad.abs().sum() > 0


def test_zero_gap_is_finite_strict_zero():
    fault = torch.randn(2, 256, requires_grad=True)
    loss = object_evidence_loss(
        fault, torch.randn_like(fault), torch.zeros(2), torch.ones(2, dtype=torch.bool)
    )
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(fault.grad, torch.zeros_like(fault))


def test_pg_routes_gradient_only_to_student_and_projector():
    detector_layer = torch.nn.Linear(12, 256)
    student = detector_layer(torch.randn(3, 12))
    teacher = torch.randn(3, 16, requires_grad=True)
    projector = PrivilegedGeometryProjector(16)
    loss = privileged_geometry_loss(
        projector(student), teacher, torch.ones(3), torch.ones(3, dtype=torch.bool),
        torch.tensor([5, 10, 20]),
    )
    loss.backward()
    assert detector_layer.weight.grad is not None and detector_layer.weight.grad.abs().sum() > 0
    assert projector[1].weight.grad is not None and projector[1].weight.grad.abs().sum() > 0
    assert teacher.grad is None
