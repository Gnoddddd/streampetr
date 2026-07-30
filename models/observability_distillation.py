"""Training-only observability-guided temporal distillation for StreamPETR."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from mmdet.core import multi_apply, reduce_mean
from mmdet.models import DETECTORS, HEADS
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from projects.mmdet3d_plugin.models.dense_heads.streampetr_head import StreamPETRHead
from projects.mmdet3d_plugin.models.detectors.petr3d import Petr3D


@HEADS.register_module()
class ObservabilityDistillationStreamPETRHead(StreamPETRHead):
    """Parameter-identical B0 head with training-only target selection.

    With ``enable_observability_distillation=False`` every overridden method
    immediately dispatches to the stock implementation.
    """

    def __init__(
        self,
        *args,
        enable_observability_distillation: bool = False,
        distill_cls_weight: float = 1.0,
        distill_box_weight: float = 1.0,
        distill_query_weight: float = 0.1,
        **kwargs,
    ):
        self.enable_observability_distillation = bool(
            enable_observability_distillation
        )
        self.distill_cls_weight = float(distill_cls_weight)
        self.distill_box_weight = float(distill_box_weight)
        self.distill_query_weight = float(distill_query_weight)
        super().__init__(*args, **kwargs)
        self._distillation_context: Optional[Dict] = None
        self._teacher_targets: Optional[Dict] = None
        self._last_outputs: Optional[Dict] = None
        self._last_query_embeddings: Optional[torch.Tensor] = None
        self._transformer_handle = None
        if self.enable_observability_distillation:
            self._transformer_handle = self.transformer.register_forward_hook(
                self._capture_transformer_output
            )

    def _capture_transformer_output(self, _module, _inputs, output) -> None:
        self._last_query_embeddings = output[0]

    def set_distillation_context(
        self, context: Optional[Dict], teacher_targets: Optional[Dict]
    ) -> None:
        self._distillation_context = context
        self._teacher_targets = teacher_targets

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        if self.enable_observability_distillation:
            self._last_outputs = outputs
        return outputs

    def _query_observable(self, centers: torch.Tensor) -> Optional[torch.Tensor]:
        context = self._distillation_context
        if context is None:
            return None
        lidar2img = context["lidar2img"].to(device=centers.device, dtype=centers.dtype)
        online = context["camera_online_mask"].to(centers).clamp(0, 1)
        fresh = context["camera_fresh_mask"].to(centers).clamp(0, 1)
        ones = torch.ones_like(centers[..., :1])
        homogeneous = torch.cat((centers, ones), dim=-1)
        projected = torch.einsum("bnij,bqj->bnqi", lidar2img, homogeneous)
        depth = projected[..., 2]
        safe_depth = depth.clamp_min(1e-6)
        u = projected[..., 0] / safe_depth
        v = projected[..., 1] / safe_depth
        height, width = context["image_hw"]
        inside = (
            (depth > 0.1)
            & (u >= 0)
            & (v >= 0)
            & (u < float(width))
            & (v < float(height))
        )
        available = (online * fresh > 0).unsqueeze(-1)
        return (inside & available).any(dim=1)

    def loss_single(
        self,
        cls_scores,
        bbox_preds,
        gt_bboxes_list,
        gt_labels_list,
        gt_bboxes_ignore_list=None,
    ):
        if (
            not self.enable_observability_distillation
            or self._distillation_context is None
        ):
            return super().loss_single(
                cls_scores,
                bbox_preds,
                gt_bboxes_list,
                gt_labels_list,
                gt_bboxes_ignore_list,
            )

        num_imgs = cls_scores.size(0)
        cls_reg_targets = self.get_targets(
            [cls_scores[i] for i in range(num_imgs)],
            [bbox_preds[i] for i in range(num_imgs)],
            gt_bboxes_list,
            gt_labels_list,
            gt_bboxes_ignore_list,
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        observable = self._query_observable(bbox_preds[..., :3])
        if observable is not None:
            for index, labels in enumerate(labels_list):
                negative = labels == self.num_classes
                label_weights_list[index][negative] *= observable[index][negative].to(
                    label_weights_list[index]
                )

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        flat_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                flat_scores.new_tensor([cls_avg_factor])
            )
        loss_cls = self.loss_cls(
            flat_scores,
            labels,
            label_weights,
            avg_factor=max(cls_avg_factor, 1),
        )
        positive_factor = torch.clamp(
            reduce_mean(loss_cls.new_tensor([num_total_pos])), min=1
        ).item()
        flat_bbox = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_targets = normalize_bbox(bbox_targets, self.pc_range)
        finite = torch.isfinite(normalized_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
            flat_bbox[finite, :10],
            normalized_targets[finite, :10],
            bbox_weights[finite, :10],
            avg_factor=positive_factor,
        )
        return torch.nan_to_num(loss_cls), torch.nan_to_num(loss_bbox)

    def _positive_token_map(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        tokens: List[str],
    ) -> Dict[str, int]:
        assignment = self.assigner.assign(
            bbox_preds,
            cls_scores,
            gt_boxes,
            gt_labels,
            None,
            self.match_costs,
            self.match_with_velo,
        )
        mapping: Dict[str, int] = {}
        for query_index in torch.nonzero(assignment.gt_inds > 0).flatten().tolist():
            gt_index = int(assignment.gt_inds[query_index].item()) - 1
            mapping[str(tokens[gt_index])] = int(query_index)
        return mapping

    def _distillation_losses(
        self, gt_bboxes_list, gt_labels_list, student_outputs
    ) -> Dict[str, torch.Tensor]:
        teacher = self._teacher_targets
        context = self._distillation_context
        zero = student_outputs["all_cls_scores"].sum() * 0.0
        if teacher is None or context is None:
            return {
                "loss_distill_cls": zero,
                "loss_distill_bbox": zero,
                "loss_distill_query": zero,
                "distill_match_rate": zero.detach(),
                "query_consistency": zero.detach(),
            }

        student_cls = student_outputs["all_cls_scores"][-1]
        student_bbox = student_outputs["all_bbox_preds"][-1]
        student_embed = self._last_query_embeddings[-1]
        teacher_cls = teacher["all_cls_scores"][-1].to(student_cls)
        teacher_bbox = teacher["all_bbox_preds"][-1].to(student_bbox)
        teacher_embed = teacher["query_embeddings"][-1].to(student_embed)
        cls_terms, box_terms, query_terms, consistencies = [], [], [], []
        matched = 0
        possible = 0
        for batch_index in range(student_cls.size(0)):
            gt_boxes = torch.cat(
                (
                    gt_bboxes_list[batch_index].gravity_center,
                    gt_bboxes_list[batch_index].tensor[:, 3:],
                ),
                dim=1,
            ).to(student_bbox)
            gt_labels = gt_labels_list[batch_index].to(student_cls.device)
            tokens = context["gt_instance_tokens"][batch_index]
            if len(tokens) != len(gt_labels):
                raise RuntimeError("GT instance tokens do not align with filtered GT")
            student_map = self._positive_token_map(
                student_cls[batch_index],
                student_bbox[batch_index],
                gt_boxes,
                gt_labels,
                tokens,
            )
            teacher_map = self._positive_token_map(
                teacher_cls[batch_index],
                teacher_bbox[batch_index],
                gt_boxes,
                gt_labels,
                tokens,
            )
            common = sorted(set(student_map) & set(teacher_map))
            possible += len(set(student_map) | set(teacher_map))
            matched += len(common)
            if not common:
                continue
            student_indices = torch.as_tensor(
                [student_map[token] for token in common],
                device=student_cls.device,
                dtype=torch.long,
            )
            teacher_indices = torch.as_tensor(
                [teacher_map[token] for token in common],
                device=teacher_cls.device,
                dtype=torch.long,
            )
            cls_terms.append(
                F.mse_loss(
                    student_cls[batch_index, student_indices],
                    teacher_cls[batch_index, teacher_indices].detach(),
                )
            )
            student_norm = normalize_bbox(
                student_bbox[batch_index, student_indices], self.pc_range
            )
            teacher_norm = normalize_bbox(
                teacher_bbox[batch_index, teacher_indices], self.pc_range
            )
            box_terms.append(F.l1_loss(student_norm, teacher_norm.detach()))
            student_query = F.normalize(
                student_embed[batch_index, student_indices].float(), dim=-1
            )
            teacher_query = F.normalize(
                teacher_embed[batch_index, teacher_indices].float(), dim=-1
            )
            cosine = (student_query * teacher_query.detach()).sum(dim=-1)
            query_terms.append((1.0 - cosine).mean())
            consistencies.append(cosine.mean().detach())

        def mean_or_zero(values):
            return torch.stack(values).mean() if values else zero

        return {
            "loss_distill_cls": self.distill_cls_weight * mean_or_zero(cls_terms),
            "loss_distill_bbox": self.distill_box_weight * mean_or_zero(box_terms),
            "loss_distill_query": self.distill_query_weight
            * mean_or_zero(query_terms),
            "distill_match_rate": zero.detach()
            + float(matched) / max(float(possible), 1.0),
            "query_consistency": mean_or_zero(consistencies).detach(),
        }

    def loss(self, gt_bboxes_list, gt_labels_list, preds_dicts, gt_bboxes_ignore=None):
        losses = super().loss(
            gt_bboxes_list, gt_labels_list, preds_dicts, gt_bboxes_ignore
        )
        if self.enable_observability_distillation:
            losses.update(
                self._distillation_losses(
                    gt_bboxes_list, gt_labels_list, preds_dicts
                )
            )
        return losses


@DETECTORS.register_module()
class ObservabilityDistillationPetr3D(Petr3D):
    """B0 detector with a non-registered, training-only EMA teacher."""

    def __init__(self, *args, enable_observability_distillation=False, **kwargs):
        self.enable_observability_distillation = bool(
            enable_observability_distillation
        )
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_ema_teacher", None)

    def set_ema_teacher(self, teacher) -> None:
        object.__setattr__(self, "_ema_teacher", teacher)

    def forward_train(
        self,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        gt_bboxes_ignore=None,
        depths=None,
        centers2d=None,
        **data,
    ):
        if not self.enable_observability_distillation:
            return super().forward_train(
                img_metas,
                gt_bboxes_3d,
                gt_labels_3d,
                gt_labels,
                gt_bboxes,
                gt_bboxes_ignore,
                depths,
                centers2d,
                **data,
            )
        teacher = self._ema_teacher
        if teacher is None:
            raise RuntimeError("EMA teacher was not installed before training")
        clean_img = data.pop("clean_img", None)
        if clean_img is None:
            raise RuntimeError("R1 requires paired clean six-camera images")
        teacher_data = dict(data)
        teacher_data["img"] = clean_img
        teacher.eval()
        teacher.pts_bbox_head.set_distillation_context(None, None)
        with torch.no_grad():
            Petr3D.forward_train(
                teacher,
                img_metas,
                gt_bboxes_3d,
                gt_labels_3d,
                gt_labels,
                gt_bboxes,
                gt_bboxes_ignore,
                depths,
                centers2d,
                **teacher_data,
            )
        teacher.eval()
        teacher_targets = {
            "all_cls_scores": teacher.pts_bbox_head._last_outputs[
                "all_cls_scores"
            ].detach(),
            "all_bbox_preds": teacher.pts_bbox_head._last_outputs[
                "all_bbox_preds"
            ].detach(),
            "query_embeddings": teacher.pts_bbox_head._last_query_embeddings.detach(),
        }
        final_metas = img_metas[-1]
        context = {
            "lidar2img": data["lidar2img"][:, -1],
            "camera_online_mask": data["camera_online_mask"][:, -1],
            "camera_fresh_mask": data["camera_fresh_mask"][:, -1],
            "image_hw": final_metas[0]["pad_shape"][0][:2],
            "gt_instance_tokens": [
                list(meta["gt_instance_tokens"]) for meta in final_metas
            ],
        }
        self.pts_bbox_head.set_distillation_context(context, teacher_targets)
        return super().forward_train(
            img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            gt_labels,
            gt_bboxes,
            gt_bboxes_ignore,
            depths,
            centers2d,
            **data,
        )
