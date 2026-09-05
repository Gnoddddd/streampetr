from pathlib import Path

import numpy as np
import torch

from models.lidar_privileged_target_evidence import (
    select_target_evidence,
    target_evidence_loss,
)


def _matrices(extra_visible=False):
    matrices = -torch.eye(4).repeat(6, 1, 1)
    matrices[3] = torch.eye(4)
    if extra_visible:
        matrices[0] = torch.eye(4)
    return matrices


def test_selector_hits_exact_s_pos_only_for_no_alternative_view_lidar_gt():
    probabilities = torch.tensor([[0.10, 0.20], [0.80, 0.30], [0.40, 0.90]])
    logits = torch.logit(probabilities)
    boxes = torch.tensor([[0.5, 0, 10, 1, 1, 1, 0],
                          [1.0, 0, 10, 1, 1, 1, 0],
                          [4.0, 0, 10, 1, 1, 1, 0]], dtype=torch.float32)
    gt_boxes = torch.tensor([[0, 0, 10, 2, 2, 2, 0]], dtype=torch.float32)
    selected, diagnostics = select_target_evidence(
        logits, boxes, gt_boxes, torch.tensor([0]), torch.tensor([True]),
        _matrices(), (256, 704),
    )
    assert len(selected) == 1
    assert selected[0]["positive_query"] == 1
    np.testing.assert_allclose(selected[0]["s_pos"], 0.8, rtol=1e-6)
    assert diagnostics[0]["alternative_view_count"] == 0


def test_selector_rejects_nonfault_no_lidar_and_alternative_view():
    logits = torch.zeros(1, 1)
    boxes = torch.tensor([[0, 0, 10, 1, 1, 1, 0]], dtype=torch.float32)
    gt = torch.tensor([[0, 0, 10, 2, 2, 2, 0]], dtype=torch.float32)
    common = (logits, boxes, gt, torch.tensor([0]))
    assert not select_target_evidence(*common, torch.tensor([False]),
                                      _matrices(), (256, 704))[0]
    assert not select_target_evidence(*common, torch.tensor([True]),
                                      _matrices(extra_visible=True), (256, 704))[0]
    assert not select_target_evidence(*common, torch.tensor([True]),
                                      _matrices(), (256, 704), fault_active=False)[0]


def test_positive_bce_gradient_touches_only_selected_gt_class_logit():
    logits = torch.tensor([[0.1, -0.2], [-1.0, 0.4]], requires_grad=True)
    selected = [{"positive_query": 1, "gt_class": 0, "gt": 0}]
    loss, details = target_evidence_loss(logits, selected)
    loss.backward()
    assert torch.isfinite(loss)
    assert details[0]["raw_loss"] > 0
    assert logits.grad[1, 0] < 0
    mask = torch.ones_like(logits.grad, dtype=torch.bool)
    mask[1, 0] = False
    assert torch.count_nonzero(logits.grad[mask]) == 0


def test_train_only_source_has_no_rank_or_competitor_objective():
    source = Path("models/lidar_privileged_target_evidence.py").read_text()
    assert "topk" not in source.lower()
    assert "competitor" not in source.lower()
    head = Path("models/lidar_privileged_target_evidence_head.py").read_text()
    assert "if self.training and self.enable_lidar_target_evidence" in head
    assert "self._lidar_target_context = None" in head
