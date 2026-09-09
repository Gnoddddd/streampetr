from types import SimpleNamespace

import torch
from torch import nn

from models.object_evidence.adapters.base import align_by_object_id
from models.object_evidence.adapters.streampetr import (
    FinalDecoderQueryCapture, StreamPETRAdapter, strip_dn_prefix,
)
from models.object_evidence.types import ObjectEvidenceBatch


class Assigner:
    def __init__(self, queries):
        self.queries = queries

    def assign(self, boxes, scores, gt_boxes, gt_labels, *args):
        assigned = torch.zeros(len(boxes), dtype=torch.long, device=boxes.device)
        for gt, query in enumerate(self.queries[:len(gt_boxes)]):
            assigned[query] = gt + 1
        return SimpleNamespace(gt_inds=assigned)


class Head(nn.Module):
    def __init__(self, queries=(1, 3)):
        super().__init__()
        self.cls_branches = nn.ModuleList([nn.Linear(256, 2)])
        self.assigner = Assigner(queries)
        self.match_costs = None
        self.match_with_velo = False


def test_dn_prefix_is_removed():
    values = torch.arange(7 * 256).reshape(1, 7, 256)
    assert torch.equal(strip_dn_prefix(values, 2, 5), values[:, 2:])


def test_passive_hook_captures_final_pre_cls_representation():
    head = Head()
    value = torch.randn(1, 5, 256)
    with FinalDecoderQueryCapture(head) as capture:
        head.cls_branches[-1](value)
    assert capture.tensor.data_ptr() == value.data_ptr()


def test_adapter_returns_one_256_token_per_matched_gt_after_dn():
    head = Head()
    adapter = StreamPETRAdapter(SimpleNamespace(pts_bbox_head=head))
    captured = torch.randn(1, 7, 256)
    outputs = {
        "all_cls_scores": torch.randn(1, 1, 5, 2),
        "all_bbox_preds": torch.randn(1, 1, 5, 10),
        "dn_mask_dict": {"pad_size": 2},
    }
    result = adapter.extract(
        captured, outputs, [torch.randn(2, 9)], [torch.tensor([0, 1])],
        object_ids=[["a", "b"]],
    )
    assert result.batch.tokens.shape == (2, 256)
    assert result.matched_queries.tolist() == [1, 3]
    assert torch.equal(result.batch.tokens, captured[0, torch.tensor([3, 5])])


def _evidence(tokens, ids):
    return ObjectEvidenceBatch(
        tokens=tokens,
        batch_indices=torch.zeros(len(ids), dtype=torch.long),
        object_ids=ids,
        valid_mask=torch.ones(len(ids), dtype=torch.bool),
    )


def test_clean_fault_different_queries_pair_by_gt_identity():
    clean = _evidence(torch.stack((torch.ones(256), torch.ones(256) * 2)), ["gt0", "gt1"])
    fault = _evidence(torch.stack((torch.ones(256) * 3, torch.ones(256) * 4)), ["gt1", "gt0"])
    aligned_clean, aligned_fault = align_by_object_id(clean, fault)
    assert aligned_clean.object_ids == ["gt0", "gt1"]
    assert torch.equal(aligned_fault.tokens[:, 0], torch.tensor([4.0, 3.0]))
