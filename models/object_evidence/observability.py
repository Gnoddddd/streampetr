"""Architecture-neutral object observability-gap geometry."""

from typing import Sequence, Tuple, Union

import torch
from torch import Tensor


def box_points(boxes: Tensor) -> Tensor:
    """Return the eight oriented corners plus center for [N,>=7] LiDAR boxes."""
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError("boxes must have shape [N,>=7]")
    signs = boxes.new_tensor([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ])
    # nuScenes box columns are center, width, length, height, yaw.
    local = signs[None] * boxes[:, None, [4, 3, 5]].abs() / 2
    cosine, sine = boxes[:, 6].cos()[:, None], boxes[:, 6].sin()[:, None]
    corners = torch.stack(
        (local[..., 0] * cosine - local[..., 1] * sine,
         local[..., 0] * sine + local[..., 1] * cosine,
         local[..., 2]),
        dim=-1,
    ) + boxes[:, None, :3]
    return torch.cat((corners, boxes[:, None, :3]), dim=1)


def camera_support(
    boxes: Tensor,
    lidar_to_image: Tensor,
    image_shapes: Union[Tensor, Sequence[Tuple[int, int]]],
) -> Tensor:
    """Fraction of nine box points in front of and inside each camera image."""
    if lidar_to_image.ndim != 3 or lidar_to_image.shape[-2:] != (4, 4):
        raise ValueError("lidar_to_image must have shape [C,4,4]")
    points = box_points(boxes)
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    projected = torch.einsum("cij,npj->ncpi", lidar_to_image.to(boxes), homogeneous)
    depth = projected[..., 2]
    denominator = depth.clamp_min(torch.finfo(depth.dtype).eps)
    x, y = projected[..., 0] / denominator, projected[..., 1] / denominator
    shapes = torch.as_tensor(image_shapes, device=boxes.device, dtype=boxes.dtype)
    if shapes.ndim == 1:
        shapes = shapes[None].expand(lidar_to_image.shape[0], -1)
    if shapes.shape != (lidar_to_image.shape[0], 2):
        raise ValueError("image_shapes must be [C,2] in (height,width) order")
    height, width = shapes[:, 0][None, :, None], shapes[:, 1][None, :, None]
    inside = (depth > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return inside.to(boxes.dtype).mean(dim=-1)


def compound_fault_strength(strengths: Tensor, dim: int = -1) -> Tensor:
    """Compose independent fault strengths with 1 - product(1-a)."""
    if torch.any((strengths < 0) | (strengths > 1)):
        raise ValueError("fault strengths must be in [0,1]")
    return 1 - (1 - strengths).prod(dim=dim)


def observability_gap(support: Tensor, fault_strength: Tensor, eps: float = 1e-6):
    """Return per-object gap and an eligibility mask for objects with camera support."""
    if support.ndim != 2:
        raise ValueError("support must have shape [N,C]")
    if fault_strength.ndim == 1:
        fault_strength = fault_strength[None, :]
    if fault_strength.shape[-1] != support.shape[-1] or fault_strength.shape[0] not in (1, len(support)):
        raise ValueError("fault_strength must broadcast to [N,C]")
    if torch.any((support < 0) | (support > 1)) or torch.any(
        (fault_strength < 0) | (fault_strength > 1)
    ):
        raise ValueError("support and fault_strength must be in [0,1]")
    total = support.sum(dim=1)
    valid = total > 0
    gap = (support * fault_strength.to(support)).sum(dim=1) / (total + eps)
    gap = torch.where(valid, gap.clamp(0, 1), torch.zeros_like(gap))
    return gap, valid
