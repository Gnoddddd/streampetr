"""Stock StreamPETR head plus an optional train-only boundary objective."""

from __future__ import annotations

from typing import Dict, List

import torch
from mmcv.runner import force_fp32
from mmdet.models import HEADS
from projects.mmdet3d_plugin.models.dense_heads.streampetr_head import StreamPETRHead

from .hard_positive_boundary import (
    hard_positive_boundary_loss,
    select_hard_positive_pairs,
)


@HEADS.register_module()
class HardPositiveBoundaryStreamPETRHead(StreamPETRHead):
    """Inference-identical B0 with a strict fault-only training loss."""

    def __init__(self, *args, enable_hard_positive_boundary=False,
                 hard_positive_boundary_weight=0.0,
                 hard_positive_boundary_margin=0.10,
                 hard_positive_geometry_threshold=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_hard_positive_boundary = bool(enable_hard_positive_boundary)
        self.hard_positive_boundary_weight = float(hard_positive_boundary_weight)
        self.hard_positive_boundary_margin = float(hard_positive_boundary_margin)
        self.hard_positive_geometry_threshold = float(hard_positive_geometry_threshold)
        self._hard_boundary_context = None
        self._last_hard_boundary_diagnostics: List[Dict] = []
        self._last_hard_boundary_summary: Dict[str, float] = {}

    def forward(self, memory_center, img_metas, topk_indexes=None, **data):
        if self.training and self.enable_hard_positive_boundary:
            self._hard_boundary_context = {
                "img_metas": img_metas,
                "online": data.get("camera_online_mask"),
                "quality": data.get("camera_quality"),
                "fresh": data.get("camera_fresh_mask"),
                "severity": data.get("corruption_severity"),
            }
        else:
            self._hard_boundary_context = None
        return super().forward(memory_center, img_metas, topk_indexes, **data)

    def _fault(self, batch: int) -> bool:
        if not self._hard_boundary_context:
            return False
        for key in ("online", "quality", "fresh"):
            value = self._hard_boundary_context.get(key)
            if torch.is_tensor(value):
                row = value[batch] if value.ndim > 1 else value
                if bool(torch.any(row.detach().float() < 1.0 - 1e-6)):
                    return True
        severity = self._hard_boundary_context.get("severity")
        if torch.is_tensor(severity):
            row = severity[batch] if severity.ndim > 1 else severity
            if bool(torch.any(row.detach().float() > 1e-6)):
                return True
        return False

    @force_fp32(apply_to=("preds_dicts",))
    def loss(self, gt_bboxes_list, gt_labels_list, preds_dicts,
             gt_bboxes_ignore=None):
        original = super().loss(
            gt_bboxes_list, gt_labels_list, preds_dicts, gt_bboxes_ignore
        )
        if not (self.training and self.enable_hard_positive_boundary):
            self._last_hard_boundary_diagnostics = []
            self._last_hard_boundary_summary = {}
            return original

        logits = preds_dicts["all_cls_scores"][-1]
        boxes = preds_dicts["all_bbox_preds"][-1]
        converted = [torch.cat(
            (value.gravity_center, value.tensor[:, 3:]), dim=1
        ).to(gt_labels_list[index].device) for index, value in enumerate(gt_bboxes_list)]
        terms, diagnostics = [], []
        fault_gt = fault_pairs = clean_would_pairs = 0
        topk = int(getattr(self.bbox_coder, "max_num", 100))
        for batch, (gt, labels) in enumerate(zip(converted, gt_labels_list)):
            selected, summary = select_hard_positive_pairs(
                logits[batch], boxes[batch], gt[:, :3], labels,
                topk=topk,
                geometry_threshold=self.hard_positive_geometry_threshold,
            )
            is_fault = self._fault(batch)
            raw, details = hard_positive_boundary_loss(
                logits[batch], selected,
                margin=self.hard_positive_boundary_margin,
            )
            if is_fault:
                terms.append(raw); fault_gt += len(gt); fault_pairs += len(selected)
            else:
                # Preserve the exact fault-only objective while recording the
                # clean counterfactual selection rate for activation audit.
                terms.append(raw * 0.0); clean_would_pairs += len(selected)
            meta = self._hard_boundary_context["img_metas"][batch]
            selected_by_gt = {int(item["gt"]): item for item in details}
            per_gt = {int(item["gt"]): item for item in summary["per_gt"]}
            for target in range(len(gt)):
                item = selected_by_gt.get(target)
                base = per_gt[target]
                diagnostics.append({
                    "sample_token": str(meta.get("sample_idx", "")),
                    "scene_token": str(meta.get("scene_token", "")),
                    "frame_idx": int(meta.get("frame_idx", -1)),
                    "batch": batch,
                    "gt": target,
                    "gt_class": int(labels[target]),
                    "fault": is_fault,
                    "actual_loss_enabled": is_fault,
                    "near_query_count": base["near_query_count"],
                    "best_near_rank": base["best_near_rank"],
                    "strict_rank_out_candidates": base["strict_rank_out_candidates"],
                    "pair_selected": item is not None,
                    **(item or {
                        "positive_query": -1, "positive_rank": -1,
                        "rank_distance_from_k": -1, "center_distance": float("nan"),
                        "negative_query": -1, "negative_class": -1,
                        "negative_rank": topk, "boundary_score_detached": summary["boundary_score"],
                        "s_pos": float("nan"), "s_neg": float("nan"),
                        "score_gap": float("nan"), "loss": 0.0,
                        "nonzero": False, "negative_truly_outranks": False,
                    }),
                    "actual_weighted_loss": float(item["loss"] if item and is_fault else 0.0),
                })
        zero = logits.sum() * 0.0
        raw_loss = torch.stack(terms).mean() if terms else zero
        original["loss_hard_positive_boundary"] = (
            raw_loss * self.hard_positive_boundary_weight
        )
        original["hard_positive_boundary_raw"] = raw_loss.detach()
        self._last_hard_boundary_diagnostics = diagnostics
        self._last_hard_boundary_summary = {
            "fault_gt": fault_gt, "fault_pairs": fault_pairs,
            "clean_would_pairs": clean_would_pairs,
        }
        return original

