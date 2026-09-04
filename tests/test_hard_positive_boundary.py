import ast
from pathlib import Path

import torch

from models.hard_positive_boundary import (
    hard_positive_boundary_loss,
    select_hard_positive_pairs,
)


def fixture():
    logits = torch.full((30, 3), -4.0)
    # Exactly 20 query-class pairs outrank the GT-class hard positive.
    logits.reshape(-1)[:20] = 3.0
    logits[25, 2] = 0.0
    boxes = torch.full((30, 10), 20.0)
    boxes[25, :3] = torch.tensor([0.5, 0.0, 0.0])
    return logits, boxes, torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([2])


def test_selection_targets_strict_geometry_near_rank_out_case():
    logits, boxes, centers, labels = fixture()
    selected, summary = select_hard_positive_pairs(
        logits, boxes, centers, labels, topk=10, geometry_threshold=2.0
    )
    assert summary["topk"] == 10
    assert len(selected) == 1
    assert selected[0]["positive_query"] == 25
    assert selected[0]["positive_rank"] > 10
    assert selected[0]["center_distance"] == 0.5
    assert selected[0]["negative_rank"] == 10


def test_easy_topk_positive_is_not_ranked():
    logits, boxes, centers, labels = fixture()
    logits[25, 2] = 10.0
    selected, _ = select_hard_positive_pairs(logits, boxes, centers, labels, topk=10)
    assert selected == []


def test_lower_ranked_near_duplicate_cannot_activate_easy_gt():
    logits, boxes, centers, labels = fixture()
    boxes[1, :3] = torch.tensor([0.2, 0.0, 0.0])
    logits[1, 2] = 9.0
    selected, summary = select_hard_positive_pairs(
        logits, boxes, centers, labels, topk=10
    )
    assert summary["per_gt"][0]["near_query_count"] == 2
    assert summary["per_gt"][0]["best_near_rank"] <= 10
    assert selected == []


def test_pairwise_loss_gradients_only_selected_scores():
    logits, boxes, centers, labels = fixture()
    logits.requires_grad_()
    selected, _ = select_hard_positive_pairs(logits, boxes, centers, labels, topk=10)
    loss, details = hard_positive_boundary_loss(logits, selected, margin=0.10)
    loss.backward()
    nonzero = torch.nonzero(logits.grad, as_tuple=False)
    assert len(nonzero) == 2
    assert details[0]["nonzero"]
    assert details[0]["negative_truly_outranks"]


def test_positive_query_is_unique_across_gts():
    logits, boxes, _, _ = fixture()
    centers = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    labels = torch.tensor([2, 2])
    selected, _ = select_hard_positive_pairs(logits, boxes, centers, labels, topk=10)
    queries = [item["positive_query"] for item in selected]
    assert len(queries) == len(set(queries))


def test_head_disabled_path_calls_stock_loss_before_early_return():
    source = Path("models/hard_positive_boundary_head.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "original = super().loss" in source
    assert "if not (self.training and self.enable_hard_positive_boundary)" in source
    assert "return original" in source
    calls = {
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "register_buffer" not in calls
    assert "register_parameter" not in calls
