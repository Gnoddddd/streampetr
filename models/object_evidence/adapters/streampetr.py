"""Passive StreamPETR training adapter; no third-party source changes required."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch
from torch import Tensor

from .base import ObjectEvidenceAdapter
from ..paired_training import temporary_eval_no_grad
from ..types import ObjectEvidenceBatch


MEMORY_NAMES = (
    "memory_embedding", "memory_reference_point", "memory_velo",
    "memory_timestamp", "memory_egopose",
)


def dn_pad_size(outputs: Optional[Dict[str, Any]]) -> int:
    if not outputs:
        return 0
    mask = outputs.get("dn_mask_dict") or outputs.get("mask_dict")
    return int(mask.get("pad_size", 0)) if mask else 0


def strip_dn_prefix(tokens: Tensor, pad_size: int, expected_queries: Optional[int] = None) -> Tensor:
    """Strip only the DN prefix while retaining all normal detection queries."""
    if tokens.ndim != 3:
        raise ValueError("captured decoder tokens must have shape [B,Q,256]")
    if tokens.shape[-1] != 256 or pad_size < 0 or pad_size > tokens.shape[1]:
        raise ValueError("invalid decoder token shape or DN pad size")
    stripped = tokens[:, pad_size:]
    if expected_queries is not None and stripped.shape[1] != expected_queries:
        raise ValueError(
            f"DN stripping produced {stripped.shape[1]} queries; expected {expected_queries}"
        )
    return stripped


class FinalDecoderQueryCapture:
    """Capture the input to the final classification branch via a passive hook."""

    def __init__(self, head):
        self.head = head
        self.values: List[Tensor] = []
        self._handle = None

    def __enter__(self):
        def hook(_module, arguments):
            if not arguments or not torch.is_tensor(arguments[0]):
                raise RuntimeError("classification pre-hook did not receive a tensor")
            self.values.append(arguments[0])

        self._handle = self.head.cls_branches[-1].register_forward_pre_hook(hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._handle.remove()
        self._handle = None

    @property
    def tensor(self) -> Tensor:
        if not self.values:
            raise RuntimeError("the final decoder query hook captured nothing")
        value = self.values[-1]
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if value.ndim != 3 or value.shape[-1] != 256:
            raise RuntimeError(f"unexpected final decoder token shape {tuple(value.shape)}")
        return value


@dataclass
class StreamPETREvidence:
    batch: ObjectEvidenceBatch
    matched_queries: Tensor


def _tensor_boxes(boxes, device) -> Tensor:
    if hasattr(boxes, "gravity_center") and hasattr(boxes, "tensor"):
        return torch.cat((boxes.gravity_center, boxes.tensor[:, 3:]), dim=1).to(device)
    return torch.as_tensor(boxes, device=device)


class StreamPETRAdapter(ObjectEvidenceAdapter):
    def __init__(self, detector):
        self.detector = detector
        self.head = detector.pts_bbox_head

    def capture(self):
        return FinalDecoderQueryCapture(self.head)

    def extract(
        self,
        captured_tokens: Tensor,
        outputs: Dict[str, Tensor],
        gt_bboxes: Sequence[Any],
        gt_labels: Sequence[Tensor],
        object_ids: Optional[Sequence[Sequence[Any]]] = None,
    ) -> StreamPETREvidence:
        scores = outputs["all_cls_scores"][-1]
        boxes = outputs["all_bbox_preds"][-1]
        pad = dn_pad_size(outputs)
        if captured_tokens.ndim == 2:
            captured_tokens = captured_tokens.unsqueeze(0)
        if captured_tokens.shape[1] == scores.shape[1]:
            normal_tokens = captured_tokens
        else:
            normal_tokens = strip_dn_prefix(captured_tokens, pad, scores.shape[1])
        if normal_tokens.shape[0] != scores.shape[0] or normal_tokens.shape[-1] != 256:
            raise ValueError("captured token and prediction batch dimensions disagree")

        tokens, batches, identities, gt_indexes, labels, centers, queries = [], [], [], [], [], [], []
        for batch_index in range(scores.shape[0]):
            gt_box = _tensor_boxes(gt_bboxes[batch_index], boxes.device)
            gt_label = gt_labels[batch_index].to(boxes.device)
            assignment = self.head.assigner.assign(
                boxes[batch_index], scores[batch_index], gt_box, gt_label, None,
                getattr(self.head, "match_costs", None),
                getattr(self.head, "match_with_velo", False),
            )
            for query in torch.nonzero(assignment.gt_inds > 0, as_tuple=False).flatten():
                gt_index = int(assignment.gt_inds[query]) - 1
                tokens.append(normal_tokens[batch_index, query])
                batches.append(batch_index)
                gt_indexes.append(gt_index)
                labels.append(gt_label[gt_index])
                centers.append(gt_box[gt_index, :3])
                queries.append(query)
                identities.append(
                    object_ids[batch_index][gt_index]
                    if object_ids is not None else gt_index
                )
        if not tokens:
            empty = ObjectEvidenceBatch.empty(normal_tokens)
            return StreamPETREvidence(empty, torch.empty(0, dtype=torch.long, device=boxes.device))
        batch = ObjectEvidenceBatch(
            tokens=torch.stack(tokens),
            batch_indices=torch.tensor(batches, dtype=torch.long, device=boxes.device),
            object_ids=identities,
            valid_mask=torch.ones(len(tokens), dtype=torch.bool, device=boxes.device),
            gt_indices=torch.tensor(gt_indexes, dtype=torch.long, device=boxes.device),
            labels=torch.stack(labels).long(),
            centers=torch.stack(centers),
        )
        return StreamPETREvidence(batch, torch.stack(queries).long())

    def snapshot_memory(self) -> Dict[str, Optional[Tensor]]:
        return {
            name: None if getattr(self.head, name, None) is None
            else getattr(self.head, name).detach().clone()
            for name in MEMORY_NAMES
        }

    def restore_memory(self, state: Dict[str, Optional[Tensor]]) -> None:
        for name, value in state.items():
            setattr(self.head, name, None if value is None else value.detach().clone())

    def paired_temporal_forward(
        self,
        clean_frames: Sequence[Any],
        fault_frames: Sequence[Any],
        forward_frame: Callable[[Any, bool], Any],
    ):
        """Serial clean-teacher and fault-student trajectories from identical memory."""
        if not clean_frames or len(clean_frames) != len(fault_frames):
            raise ValueError("clean and fault clips must have the same positive length")
        initial = self.snapshot_memory()
        self.restore_memory(initial)
        with temporary_eval_no_grad(self.detector):
            for frame in clean_frames[:-1]:
                forward_frame(frame, False)
            with self.capture() as clean_capture:
                clean_output = forward_frame(clean_frames[-1], False)
                clean_tokens = clean_capture.tensor.detach()

        self.restore_memory(initial)
        with torch.no_grad():
            for frame in fault_frames[:-1]:
                forward_frame(frame, False)
        with self.capture() as fault_capture:
            fault_output = forward_frame(fault_frames[-1], True)
            fault_tokens = fault_capture.tensor
        return clean_output, clean_tokens, fault_output, fault_tokens
