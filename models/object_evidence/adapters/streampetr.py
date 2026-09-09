"""Passive StreamPETR training adapter; no third-party source changes required."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch
from torch import Tensor

from .base import ObjectEvidenceAdapter, align_by_object_id
from ..fault_sampler import PairedFaultSampler
from ..integrations.fault_images import fault_normalized_images
from ..observability import camera_support, observability_gap
from ..paired_training import (
    ObjectEvidenceObjective, temporary_eval_no_grad, temporary_native_teacher,
)
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


try:  # Registered only inside a StreamPETR/OpenMMLab runtime.
    from mmdet.models import DETECTORS
    from projects.mmdet3d_plugin.models.detectors.petr3d import Petr3D
except (ImportError, ModuleNotFoundError):  # pragma: no cover - dependency-light unit tests
    DETECTORS = None
    Petr3D = None


if Petr3D is not None:
    @DETECTORS.register_module()
    class OEStreamPETR(Petr3D):
        """Native StreamPETR training integration for R0 and OE modes."""

        def __init__(self, object_evidence=None, **kwargs):
            config = dict(object_evidence or {})
            self.object_evidence_enabled = bool(config.pop("enabled", False))
            self.object_evidence_config = config
            super().__init__(**kwargs)
            self._oep_sampler = PairedFaultSampler(
                seed=int(config.get("seed", 2026)),
                pair_probability=float(config.get("pair_probability", 0.5)),
            )
            self._oep_iteration = 0
            self._oep_force_paired = False
            self.object_evidence = ObjectEvidenceObjective(
                teacher_dim=None,
                lambda_oe=float(config.get("lambda_oe", 0.5)),
                lambda_pg=0.0,
                warmup_iters=int(config.get("auxiliary_warmup_iters", 1000)),
            ) if self.object_evidence_enabled else None

        def _run_and_capture(self, arguments):
            adapter = StreamPETRAdapter(self)
            outputs = []
            handle = self.pts_bbox_head.register_forward_hook(
                lambda _module, _inputs, value: outputs.append(value)
            )
            try:
                with adapter.capture() as capture:
                    losses = super().forward_train(**arguments)
            finally:
                handle.remove()
            if not outputs:
                raise RuntimeError("native StreamPETR head produced no output")
            return losses, capture.tensor, outputs[-1]

        @staticmethod
        def _scene_tokens(img_metas):
            current = img_metas[-1]
            return [str(meta.get("scene_token", meta.get("sample_idx", index)))
                    for index, meta in enumerate(current)]

        @staticmethod
        def _normalization(img_metas):
            config = img_metas[-1][0].get("img_norm_cfg", {})
            return (
                config.get("mean", [123.675, 116.28, 103.53]),
                config.get("std", [58.395, 57.12, 57.375]),
            )

        def forward_train(
            self, img_metas=None, gt_bboxes_3d=None, gt_labels_3d=None,
            gt_labels=None, gt_bboxes=None, gt_bboxes_ignore=None,
            depths=None, centers2d=None, **data,
        ):
            if not self.object_evidence_enabled:
                return super().forward_train(
                    img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d,
                    gt_labels_3d=gt_labels_3d, gt_labels=gt_labels,
                    gt_bboxes=gt_bboxes, gt_bboxes_ignore=gt_bboxes_ignore,
                    depths=depths, centers2d=centers2d, **data,
                )
            iteration = self._oep_iteration
            self._oep_iteration += 1
            paired = self._oep_force_paired or self._oep_sampler.paired_iteration()
            common = dict(
                img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d, gt_labels=gt_labels,
                gt_bboxes=gt_bboxes, gt_bboxes_ignore=gt_bboxes_ignore,
                depths=depths, centers2d=centers2d,
            )
            clean_arguments = {**common, **data}
            if not paired:
                return super().forward_train(**clean_arguments)

            clip_length = int(data["img"].shape[1])
            episode = self._oep_sampler.sample(clip_length)
            mean, std = self._normalization(img_metas)
            fault_image = fault_normalized_images(
                data["img"], episode, mean, std, self._scene_tokens(img_metas)
            )
            fault_arguments = {**common, **data, "img": fault_image}
            self._object_evidence_last_episode = episode
            if self.object_evidence.lambda_oe <= 0:
                return super().forward_train(**fault_arguments)

            adapter = StreamPETRAdapter(self)
            initial_memory = adapter.snapshot_memory()
            adapter.restore_memory(initial_memory)
            with temporary_native_teacher(self):
                _, clean_tokens, clean_outputs = self._run_and_capture(clean_arguments)
            adapter.restore_memory(initial_memory)
            fault_losses, fault_tokens, fault_outputs = self._run_and_capture(fault_arguments)

            current_boxes = gt_bboxes_3d[-1]
            current_labels = gt_labels_3d[-1]
            identities = [list(range(len(boxes))) for boxes in current_boxes]
            clean_evidence = adapter.extract(
                clean_tokens, clean_outputs, current_boxes, current_labels, identities
            ).batch
            fault_evidence = adapter.extract(
                fault_tokens, fault_outputs, current_boxes, current_labels, identities
            ).batch
            clean_evidence, fault_evidence = align_by_object_id(clean_evidence, fault_evidence)

            transforms = data["lidar2img"]
            if transforms.ndim == 5:
                transforms = transforms[:, -1]
            gaps, support_masks = [], []
            strengths = data["img"].new_zeros(data["img"].shape[0], data["img"].shape[2])
            strengths[:, episode.spec.camera] = episode.spec.strength
            for batch_index, boxes in enumerate(current_boxes):
                box_tensor = _tensor_boxes(boxes, transforms.device)
                support = camera_support(
                    box_tensor, transforms[batch_index], data["img"].shape[-2:]
                )
                gap, supported = observability_gap(support, strengths[batch_index])
                gaps.append(gap)
                support_masks.append(supported)
            selected_gap, selected_valid = [], []
            for batch_index, gt_index in zip(
                clean_evidence.batch_indices.tolist(), clean_evidence.gt_indices.tolist()
            ):
                selected_gap.append(gaps[batch_index][gt_index])
                selected_valid.append(support_masks[batch_index][gt_index])
            gap_tensor = (
                torch.stack(selected_gap) if selected_gap
                else fault_evidence.tokens.new_empty((0,))
            )
            valid_mask = (
                torch.stack(selected_valid).bool() if selected_valid
                else torch.empty(0, dtype=torch.bool, device=fault_evidence.tokens.device)
            )
            valid_mask = valid_mask & clean_evidence.valid_mask & fault_evidence.valid_mask
            auxiliary = self.object_evidence(
                clean_evidence.tokens, fault_evidence.tokens, gap_tensor,
                valid_mask, iteration,
            )
            fault_losses["loss_oe"] = auxiliary["loss_object_evidence_total"]
            fault_losses["oe_value"] = auxiliary["loss_object_evidence"].detach()
            fault_losses["oe_gap_mean"] = (
                gap_tensor.mean().detach() if len(gap_tensor) else gap_tensor.new_zeros(())
            )
            fault_losses["oe_gap_max"] = (
                gap_tensor.max().detach() if len(gap_tensor) else gap_tensor.new_zeros(())
            )
            fault_losses["oe_matched_objects"] = gap_tensor.new_tensor(len(gap_tensor))
            return fault_losses
