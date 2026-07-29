"""Training-only Ray Denoising adaptation for StreamPETR.

The implementation follows the official RayDN release at commit
``cdb8c2cf72b4b1f1a768f2e1371224436bcc4635`` while keeping all changes on the
Evidence3D side of the repository boundary.  Ray queries are ordinary
denoising queries: they never enter inference output or temporal memory.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor
from mmdet.models import HEADS

from projects.mmdet3d_plugin.models.dense_heads.streampetr_head import (
    StreamPETRHead,
)


def _unwrap(value):
    return value._data if hasattr(value, "_data") else value


def prepare_ray_denoising(
    head,
    batch_size: int,
    reference_points: Tensor,
    img_metas: Sequence[Dict],
    data: Dict[str, Tensor],
    *,
    raydn_group: int = 1,
    raydn_num: int = 5,
    raydn_alpha: float = 8.0,
    raydn_beta: float = 2.0,
    raydn_radius: float = 3.0,
) -> Tuple[Tensor, Tensor, Dict]:
    """Build standard DN and ray-aligned DN queries for a training batch."""

    if not head.training or not head.with_dn:
        return (
            reference_points.unsqueeze(0).repeat(batch_size, 1, 1),
            None,
            None,
        )
    if raydn_group != 1 or raydn_num != 5:
        raise ValueError("The screening protocol fixes raydn_group=1 and raydn_num=5")
    if raydn_alpha != 8.0 or raydn_beta != 2.0 or raydn_radius != 3.0:
        raise ValueError("The screening protocol fixes Beta(8,2) and radius=3")
    if "lidar2img" not in data:
        raise KeyError("RayDN training requires lidar2img")

    targets = [
        torch.cat(
            (
                _unwrap(meta["gt_bboxes_3d"]).gravity_center,
                _unwrap(meta["gt_bboxes_3d"]).tensor[:, 3:],
            ),
            dim=1,
        )
        for meta in img_metas
    ]
    labels_per_image = [
        _unwrap(meta["gt_labels_3d"]).to(reference_points.device)
        for meta in img_metas
    ]
    known_num = [target.size(0) for target in targets]
    if not known_num or max(known_num) == 0:
        return (
            reference_points.unsqueeze(0).repeat(batch_size, 1, 1),
            None,
            None,
        )

    labels = torch.cat(labels_per_image)
    boxes = torch.cat(targets).to(reference_points.device)
    compute_dtype = (
        torch.float32
        if reference_points.device.type == "cpu"
        and reference_points.dtype == torch.float16
        else boxes.dtype
    )
    compute_boxes = boxes.to(dtype=compute_dtype)
    pc_range = head.pc_range.to(
        device=reference_points.device, dtype=compute_dtype
    )
    batch_idx = torch.cat(
        [
            labels.new_full((target.size(0),), index)
            for index, target in enumerate(targets)
        ]
    )
    base_indices = torch.arange(labels.numel(), device=reference_points.device)
    total_ray_groups = raydn_group * raydn_num
    total_groups = int(head.scalar) + total_ray_groups
    known_indice = base_indices.repeat(total_groups)
    known_bid = batch_idx.repeat(total_groups)

    # Preserve StreamPETR's existing spatial DN construction exactly.
    known_labels = labels.repeat(int(head.scalar)).long()
    known_bboxs_standard = compute_boxes.repeat(int(head.scalar), 1)
    known_bbox_center = known_bboxs_standard[:, :3].clone()
    known_bbox_scale = known_bboxs_standard[:, 3:6].clone()
    if head.bbox_noise_scale > 0:
        diff = known_bbox_scale / 2 + head.bbox_noise_trans
        rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
        known_bbox_center += (
            rand_prob * diff * head.bbox_noise_scale
        )
        known_bbox_center = (
            known_bbox_center - pc_range[0:3]
        ) / (pc_range[3:6] - pc_range[0:3])
        known_bbox_center = known_bbox_center.clamp(0.0, 1.0)
        known_labels[
            torch.norm(rand_prob, p=2, dim=1) > head.split
        ] = head.num_classes

    lidar2img = data["lidar2img"].to(
        device=reference_points.device, dtype=compute_dtype
    )
    beta_distribution = torch.distributions.Beta(
        torch.tensor(raydn_alpha),
        torch.tensor(raydn_beta),
    )
    ray_centers = []
    ray_labels = []
    total_gt = compute_boxes.shape[0]
    for _ in range(raydn_group):
        group_labels = labels.repeat(raydn_num).long()
        centers = compute_boxes[:, :3].repeat(raydn_num, 1).view(
            raydn_num, total_gt, 3
        )
        original_centers = centers.clone()
        object_radius = compute_boxes[:, 3:6].mean(dim=-1) / 2.0
        offsets = (
            beta_distribution.sample((raydn_num, total_gt)).to(
                reference_points.device
            )
            * 2.0
            - 1.0
        ) * raydn_radius

        _, closest = offsets.abs().min(dim=0)
        columns = torch.arange(total_gt, device=reference_points.device)
        closest_offsets = offsets[closest, columns]
        reset = closest_offsets.abs() > head.split
        if reset.any():
            closest_offsets = closest_offsets.clone()
            closest_offsets[reset] = (
                torch.rand(
                    int(reset.sum().item()),
                    device=reference_points.device,
                    dtype=compute_dtype,
                )
                * 2.0
                - 1.0
            ) * head.split
            offsets[closest, columns] = closest_offsets

        is_background = torch.ones_like(offsets, dtype=torch.bool)
        is_background[closest, columns] = False
        group_labels[is_background.reshape(-1)] = head.num_classes

        start = 0
        for batch_index, count in enumerate(known_num):
            end = start + count
            if count == 0:
                start = end
                continue
            sample_centers = original_centers[:, start:end]
            homogeneous = torch.cat(
                (
                    sample_centers,
                    sample_centers.new_ones(raydn_num, count, 1),
                ),
                dim=-1,
            )
            for camera_index in range(lidar2img.shape[1]):
                matrix = lidar2img[batch_index, camera_index]
                projected = torch.matmul(
                    matrix, homogeneous.transpose(1, 2)
                ).transpose(1, 2)
                depth = projected[..., 2]
                uv = projected[..., :2] / depth.unsqueeze(-1).clamp_min(1e-6)
                shapes = img_metas[batch_index].get(
                    "pad_shape", img_metas[batch_index].get("img_shape")
                )
                shape = shapes[camera_index] if isinstance(shapes, list) else shapes
                pad_h, pad_w = float(shape[0]), float(shape[1])
                valid = (
                    (depth > 0)
                    & (uv[..., 0] >= 0)
                    & (uv[..., 0] < pad_w)
                    & (uv[..., 1] >= 0)
                    & (uv[..., 1] < pad_h)
                )
                shifted_depth = depth + (
                    object_radius[start:end].unsqueeze(0)
                    * offsets[:, start:end]
                )
                shifted = projected.clone()
                shifted[..., :2] = uv * shifted_depth.unsqueeze(-1)
                shifted[..., 2] = shifted_depth
                projected_back = torch.matmul(
                    torch.linalg.inv(matrix), shifted.transpose(1, 2)
                ).transpose(1, 2)[..., :3]
                centers[:, start:end] = torch.where(
                    valid.unsqueeze(-1),
                    projected_back,
                    centers[:, start:end],
                )
            start = end

        centers = centers.reshape(-1, 3)
        centers = (centers - pc_range[0:3]) / (
            pc_range[3:6] - pc_range[0:3]
        )
        ray_centers.append(centers.clamp(0.0, 1.0))
        ray_labels.append(group_labels)

    known_bbox_center = torch.cat([known_bbox_center] + ray_centers, dim=0)
    known_labels = torch.cat([known_labels] + ray_labels, dim=0)
    known_bboxs = boxes.repeat(total_groups, 1)

    single_pad = int(max(known_num))
    pad_size = single_pad * total_groups
    padding = reference_points.new_zeros(pad_size, 3)
    padded_reference_points = torch.cat(
        (padding, reference_points), dim=0
    ).unsqueeze(0).repeat(batch_size, 1, 1)
    local_indices = torch.cat(
        [
            torch.arange(count, device=reference_points.device)
            for count in known_num
        ]
    )
    map_known_indice = torch.cat(
        [local_indices + single_pad * group for group in range(total_groups)]
    ).long()
    padded_reference_points[(known_bid.long(), map_known_indice)] = (
        known_bbox_center.to(reference_points.dtype)
    )

    target_size = pad_size + head.num_query
    attn_mask = torch.zeros(
        target_size,
        target_size,
        dtype=torch.bool,
        device=reference_points.device,
    )
    # Matching queries cannot read any denoising query.
    attn_mask[pad_size:, :pad_size] = True
    # Standard DN copies are isolated from every other DN group.
    for group in range(int(head.scalar)):
        row = slice(single_pad * group, single_pad * (group + 1))
        attn_mask[row, : single_pad * group] = True
        attn_mask[row, single_pad * (group + 1) : pad_size] = True
    # All five ray depths for one RayDN group interact, but not with other groups.
    ray_start = single_pad * int(head.scalar)
    ray_end = ray_start + single_pad * raydn_num
    attn_mask[ray_start:ray_end, :ray_start] = True
    attn_mask[ray_start:ray_end, ray_end:pad_size] = True

    query_size = pad_size + head.num_query + head.num_propagated
    temporal_target_size = pad_size + head.num_query + head.memory_len
    temporal_mask = torch.zeros(
        query_size,
        temporal_target_size,
        dtype=torch.bool,
        device=reference_points.device,
    )
    temporal_mask[:target_size, :target_size] = attn_mask
    temporal_mask[pad_size:, :pad_size] = True

    mask_dict = {
        "known_indice": known_indice.long(),
        "batch_idx": batch_idx.long(),
        "map_known_indice": map_known_indice.long(),
        "known_lbs_bboxes": (known_labels, known_bboxs),
        "know_idx": [
            torch.ones_like(label, device=reference_points.device)
            for label in labels_per_image
        ],
        "pad_size": pad_size,
        "raydn_pad_size": single_pad * total_ray_groups,
        "raydn_enabled": True,
    }
    return padded_reference_points, temporal_mask, mask_dict


@HEADS.register_module()
class RayDNStreamPETRHead(StreamPETRHead):
    """Official StreamPETR head with project-side, training-only RayDN."""

    def __init__(
        self,
        *args,
        enable_ray_denoising: bool = False,
        raydn_group: int = 1,
        raydn_num: int = 5,
        raydn_alpha: float = 8.0,
        raydn_beta: float = 2.0,
        raydn_radius: float = 3.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.enable_ray_denoising = bool(enable_ray_denoising)
        self.raydn_group = int(raydn_group)
        self.raydn_num = int(raydn_num)
        self.raydn_alpha = float(raydn_alpha)
        self.raydn_beta = float(raydn_beta)
        self.raydn_radius = float(raydn_radius)
        self._raydn_forward_data: Optional[Dict[str, Tensor]] = None

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        if not (self.training and self.enable_ray_denoising):
            return super().prepare_for_dn(
                batch_size, reference_points, img_metas
            )
        if self._raydn_forward_data is None:
            raise RuntimeError("RayDN training data is unavailable")
        return prepare_ray_denoising(
            self,
            batch_size,
            reference_points,
            img_metas,
            self._raydn_forward_data,
            raydn_group=self.raydn_group,
            raydn_num=self.raydn_num,
            raydn_alpha=self.raydn_alpha,
            raydn_beta=self.raydn_beta,
            raydn_radius=self.raydn_radius,
        )

    def forward(self, memory_center, img_metas, topk_indexes=None, **data):
        if not (self.training and self.enable_ray_denoising):
            return super().forward(
                memory_center,
                img_metas,
                topk_indexes=topk_indexes,
                **data,
            )
        self._raydn_forward_data = data
        try:
            return super().forward(
                memory_center,
                img_metas,
                topk_indexes=topk_indexes,
                **data,
            )
        finally:
            self._raydn_forward_data = None
