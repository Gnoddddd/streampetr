"""BEVDepth shared-BEV feature and faulted-camera depth-loss adapter."""

from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import ObjectEvidenceAdapter
from ..types import ObjectEvidenceBatch


class BEVFeatureCapture:
    def __init__(self, head):
        self.head = head
        self.values = []
        self._handle = None

    def __enter__(self):
        self._handle = self.head.neck.register_forward_hook(
            lambda _module, _arguments, output: self.values.append(output)
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._handle.remove()
        self._handle = None

    @property
    def tensor(self):
        if not self.values:
            raise RuntimeError("BEV neck hook captured nothing")
        value = self.values[-1]
        if isinstance(value, (tuple, list)):
            if len(value) != 1:
                raise RuntimeError("expected a single shared BEV feature level")
            value = value[0]
        if value.ndim != 4 or value.shape[1] != 256:
            raise RuntimeError(f"shared BEV feature must be [B,256,H,W], got {tuple(value.shape)}")
        return value


def camera_reliability(num_cameras: int, fault_camera: int, fault_type: str, severity: float) -> Tensor:
    if not 0 <= fault_camera < num_cameras or not 0 <= severity <= 1:
        raise ValueError("invalid camera fault")
    weights = torch.ones(num_cameras)
    weights[fault_camera] = 0.0 if fault_type == "crash" else 1.0 - float(severity)
    return weights


def weighted_depth_loss(
    prediction: Tensor,
    target: Tensor,
    camera_weights: Tensor,
    loss_weight: float = 3.0,
) -> Tensor:
    """Official BCE depth loss with only the faulted camera reliability changed."""
    if prediction.ndim != 5:
        raise ValueError("prediction must have shape [B,C,D,H,W]")
    if target.shape != prediction.permute(0, 1, 3, 4, 2).shape:
        raise ValueError("target must have shape [B,C,H,W,D]")
    if camera_weights.ndim == 1:
        camera_weights = camera_weights[None].expand(prediction.shape[0], -1)
    if camera_weights.shape != prediction.shape[:2]:
        raise ValueError("camera_weights must have shape [C] or [B,C]")
    probabilities = prediction.permute(0, 1, 3, 4, 2).float()
    target = target.float()
    foreground = target.amax(dim=-1) > 0
    per_pixel = F.binary_cross_entropy(probabilities, target, reduction="none").sum(dim=-1)
    weights = camera_weights.to(per_pixel)[:, :, None, None] * foreground.to(per_pixel)
    return float(loss_weight) * (per_pixel * weights).sum() / weights.sum().clamp_min(1)


class BEVDepthAdapter(ObjectEvidenceAdapter):
    def __init__(self, detector_or_head, x_bound=None, y_bound=None):
        self.detector = detector_or_head
        self.head = getattr(
            detector_or_head, "pts_bbox_head", getattr(detector_or_head, "head", detector_or_head)
        )
        self.x_bound = x_bound
        self.y_bound = y_bound

    def capture(self):
        return BEVFeatureCapture(self.head)

    def _coordinates(self, centers: Tensor, height: int, width: int) -> Tuple[Tensor, Tensor]:
        config = getattr(self.head, "train_cfg", None) or {}
        if self.x_bound is not None and self.y_bound is not None:
            x_min, x_step = float(self.x_bound[0]), (float(self.x_bound[1]) - float(self.x_bound[0])) / width
            y_min, y_step = float(self.y_bound[0]), (float(self.y_bound[1]) - float(self.y_bound[0])) / height
        elif all(key in config for key in ("point_cloud_range", "voxel_size", "out_size_factor")):
            x_min, y_min = float(config["point_cloud_range"][0]), float(config["point_cloud_range"][1])
            factor = float(config["out_size_factor"])
            x_step = float(config["voxel_size"][0]) * factor
            y_step = float(config["voxel_size"][1]) * factor
        else:
            raise ValueError("BEV bounds or train_cfg geometry is required")
        return (centers[:, 0] - x_min) / x_step, (centers[:, 1] - y_min) / y_step

    def extract(
        self,
        bev_feature: Tensor,
        centers: Sequence[Tensor],
        object_ids: Optional[Sequence[Sequence[Any]]] = None,
        labels: Optional[Sequence[Tensor]] = None,
    ) -> ObjectEvidenceBatch:
        if bev_feature.ndim != 4 or bev_feature.shape[1] != 256:
            raise ValueError("bev_feature must have shape [B,256,H,W]")
        batch_size, _, height, width = bev_feature.shape
        if len(centers) != batch_size:
            raise ValueError("centers must have one tensor per batch item")
        token_parts, batches, identities, gt_indexes, label_parts, center_parts, valid_parts = [], [], [], [], [], [], []
        for batch_index, item_centers in enumerate(centers):
            item_centers = torch.as_tensor(item_centers, device=bev_feature.device, dtype=bev_feature.dtype)
            if item_centers.ndim != 2 or item_centers.shape[1] < 2:
                raise ValueError("each centers tensor must have shape [N,>=2]")
            if not len(item_centers):
                continue
            x, y = self._coordinates(item_centers, height, width)
            valid = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
            norm_x = x.new_zeros(x.shape) if width == 1 else 2 * x / (width - 1) - 1
            norm_y = y.new_zeros(y.shape) if height == 1 else 2 * y / (height - 1) - 1
            grid = torch.stack((norm_x, norm_y), dim=-1).view(1, -1, 1, 2)
            sampled = F.grid_sample(
                bev_feature[batch_index:batch_index + 1], grid,
                mode="bilinear", padding_mode="zeros", align_corners=True,
            )[0, :, :, 0].transpose(0, 1)
            token_parts.append(sampled)
            count = len(item_centers)
            batches.extend([batch_index] * count)
            gt_indexes.extend(range(count))
            identities.extend(
                list(object_ids[batch_index]) if object_ids is not None else list(range(count))
            )
            center_parts.append(item_centers[:, :3] if item_centers.shape[1] >= 3 else F.pad(item_centers, (0, 1)))
            valid_parts.append(valid)
            if labels is not None:
                label_parts.append(labels[batch_index].to(bev_feature.device).long())
        if not token_parts:
            return ObjectEvidenceBatch.empty(bev_feature)
        return ObjectEvidenceBatch(
            tokens=torch.cat(token_parts),
            batch_indices=torch.tensor(batches, dtype=torch.long, device=bev_feature.device),
            object_ids=identities,
            valid_mask=torch.cat(valid_parts),
            gt_indices=torch.tensor(gt_indexes, dtype=torch.long, device=bev_feature.device),
            labels=torch.cat(label_parts) if labels is not None else None,
            centers=torch.cat(center_parts),
        )
