"""Inference-identical StreamPETR head with optional train-only privileged loss."""

from __future__ import annotations

from typing import Dict, List

import torch
from mmcv.runner import force_fp32
from mmdet.models import HEADS
from projects.mmdet3d_plugin.models.dense_heads.streampetr_head import StreamPETRHead

from .lidar_privileged_target_evidence import (
    select_target_evidence,
    target_evidence_loss,
)


@HEADS.register_module()
class LiDARPrivilegedTargetEvidenceStreamPETRHead(StreamPETRHead):
    """B0 forward plus a disabled-by-default training-only positive signal."""

    def __init__(self, *args, enable_lidar_target_evidence=False,
                 lidar_target_evidence_weight=0.0,
                 lidar_target_geometry_threshold=2.0,
                 lidar_target_fault_camera=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_lidar_target_evidence = bool(enable_lidar_target_evidence)
        self.lidar_target_evidence_weight = float(lidar_target_evidence_weight)
        self.lidar_target_geometry_threshold = float(lidar_target_geometry_threshold)
        self.lidar_target_fault_camera = int(lidar_target_fault_camera)
        self._lidar_target_context = None
        self._last_lidar_target_diagnostics: List[Dict] = []

    def forward(self, memory_center, img_metas, topk_indexes=None, **data):
        if self.training and self.enable_lidar_target_evidence:
            self._lidar_target_context = {
                "img_metas": img_metas,
                "lidar2img": data.get("lidar2img"),
                "online": data.get("camera_online_mask"),
                "quality": data.get("camera_quality"),
                "severity": data.get("corruption_severity"),
            }
        else:
            # Privileged data cannot survive into disabled or eval inference.
            self._lidar_target_context = None
        return super().forward(memory_center, img_metas, topk_indexes, **data)

    @staticmethod
    def _batch_value(value, batch: int):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            value = value[batch] if len(value) > batch else value[-1]
            while isinstance(value, (list, tuple)):
                value = value[-1]
            return value
        if torch.is_tensor(value) and value.ndim and value.shape[0] > batch:
            value = value[batch]
        return value

    def _camera_row(self, key: str, batch: int):
        value = self._batch_value(self._lidar_target_context.get(key), batch)
        if not torch.is_tensor(value):
            return None
        return value.detach().float().reshape(-1, 6)[-1]

    def _cam_back_fault(self, batch: int) -> bool:
        camera = self.lidar_target_fault_camera
        for key in ("online", "quality"):
            row = self._camera_row(key, batch)
            if row is not None and float(row[camera]) < 1.0 - 1e-6:
                return True
        row = self._camera_row("severity", batch)
        return bool(row is not None and float(row[camera]) > 1e-6)

    def _matrices(self, batch: int):
        value = self._batch_value(self._lidar_target_context.get("lidar2img"), batch)
        if not torch.is_tensor(value):
            return None
        return value.detach().float().reshape(-1, 6, 4, 4)[-1]

    def _point_support(self, batch: int, count: int, device):
        meta = self._lidar_target_context["img_metas"][batch]
        value = meta.get("gt_lidar_point_counts")
        if value is None:
            return torch.zeros(count, dtype=torch.bool, device=device)
        value = torch.as_tensor(value, device=device).reshape(-1)
        if len(value) != count:
            raise ValueError("gt_lidar_point_counts must align with current GT")
        return value > 0

    @force_fp32(apply_to=("preds_dicts",))
    def loss(self, gt_bboxes_list, gt_labels_list, preds_dicts,
             gt_bboxes_ignore=None):
        original = super().loss(
            gt_bboxes_list, gt_labels_list, preds_dicts, gt_bboxes_ignore
        )
        if not (self.training and self.enable_lidar_target_evidence):
            self._last_lidar_target_diagnostics = []
            return original
        logits = preds_dicts["all_cls_scores"][-1]
        boxes = preds_dicts["all_bbox_preds"][-1]
        converted = [torch.cat(
            (value.gravity_center, value.tensor[:, 3:]), dim=1
        ).to(gt_labels_list[index].device) for index, value in enumerate(gt_bboxes_list)]
        terms, all_diagnostics = [], []
        for batch, (gt, labels) in enumerate(zip(converted, gt_labels_list)):
            matrices = self._matrices(batch)
            fault = self._cam_back_fault(batch)
            if matrices is None:
                selected, diagnostics = [], []
            else:
                meta = self._lidar_target_context["img_metas"][batch]
                pad_shape = meta.get("pad_shape", [(256, 704, 3)])
                shape = pad_shape[0] if isinstance(pad_shape, (list, tuple)) and isinstance(pad_shape[0], (list, tuple)) else pad_shape
                selected, diagnostics = select_target_evidence(
                    logits[batch], boxes[batch], gt, labels,
                    self._point_support(batch, len(gt), gt.device), matrices,
                    (int(shape[0]), int(shape[1])),
                    fault_camera=self.lidar_target_fault_camera,
                    geometry_threshold=self.lidar_target_geometry_threshold,
                    fault_active=fault,
                )
            raw, details = target_evidence_loss(logits[batch], selected)
            terms.append(raw)
            detailed = {int(item["gt"]): item for item in details}
            for item in diagnostics:
                merged = {**item, **detailed.get(int(item["gt"]), {})}
                merged.update({"batch": batch, "fault": fault})
                all_diagnostics.append(merged)
        zero = logits.sum() * 0.0
        raw_loss = torch.stack(terms).mean() if terms else zero
        original["loss_lidar_target_evidence"] = raw_loss * self.lidar_target_evidence_weight
        original["lidar_target_evidence_raw"] = raw_loss.detach()
        self._last_lidar_target_diagnostics = all_diagnostics
        return original
