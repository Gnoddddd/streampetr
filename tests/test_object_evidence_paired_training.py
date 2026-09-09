import torch
from torch import nn

from models.object_evidence.adapters.streampetr import StreamPETRAdapter
from models.object_evidence.paired_training import (
    ObjectEvidenceObjective, ObjectEvidenceTrainingWrapper, auxiliary_scale,
)


class TemporalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_branches = nn.ModuleList([nn.Identity()])
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_velo = None
        self.memory_timestamp = None
        self.memory_egopose = None


class TemporalDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.pts_bbox_head = TemporalHead()
        self.bn = nn.BatchNorm1d(256)


def test_clean_trajectory_is_detached_and_memory_restarts_identically():
    detector = TemporalDetector().train()
    adapter = StreamPETRAdapter(detector)
    starts = []

    def forward(frame, with_gradient):
        head = detector.pts_bbox_head
        starts.append(None if head.memory_embedding is None else float(head.memory_embedding.item()))
        head.memory_embedding = torch.tensor([[float(frame)]])
        tokens = detector.bn(torch.ones(2, 256) * frame).unsqueeze(0)
        return head.cls_branches[-1](tokens)

    clean_out, clean_tokens, fault_out, fault_tokens = adapter.paired_temporal_forward(
        [1, 2], [10, 20], forward
    )
    assert starts == [None, 1.0, None, 10.0]
    assert not clean_tokens.requires_grad
    assert fault_tokens.requires_grad
    assert detector.training
    assert detector.bn.num_batches_tracked.item() == 2


def test_warmup_and_pg_gradient_route():
    assert auxiliary_scale(0) == 0 and auxiliary_scale(500) == 0.5 and auxiliary_scale(1000) == 1
    fault = torch.randn(2, 256, requires_grad=True)
    clean = torch.randn(2, 256, requires_grad=True)
    teacher = torch.randn(2, 8, requires_grad=True)
    objective = ObjectEvidenceObjective(teacher_dim=8)
    result = objective(
        clean, fault, torch.ones(2), torch.ones(2, dtype=torch.bool), 1000,
        teacher, torch.tensor([10, 20]),
    )
    result["loss_object_evidence_total"].backward()
    assert clean.grad is None and teacher.grad is None
    assert fault.grad is not None and fault.grad.abs().sum() > 0


def test_disabled_equivalence_calls_baseline_once_and_preserves_state():
    class Detector(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.0))
            self.calls = 0

        def forward(self, value):
            self.calls += 1
            return value * self.weight

    detector = Detector()
    wrapper = ObjectEvidenceTrainingWrapper(detector, enabled=False).train()
    value = torch.tensor(3.0)
    output = wrapper(value)
    assert output.item() == 6 and detector.calls == 1
