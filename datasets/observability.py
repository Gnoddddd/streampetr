"""NumPy observability helpers for diagnostics and protocol analysis."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def project_points(points_xyz: np.ndarray, lidar2img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_xyz, dtype=np.float32)
    matrices = np.asarray(lidar2img, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must be [P,3]")
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError("lidar2img must be [N,4,4]")
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    projected = np.einsum("nij,pj->npi", matrices, homogeneous)
    depth = projected[..., 2]
    uv = projected[..., :2] / np.maximum(depth[..., None], 1e-6)
    return uv, depth


def compute_point_observability(
    points_xyz: np.ndarray,
    lidar2img: np.ndarray,
    image_hw: Sequence[Sequence[int]],
    camera_online_mask: Optional[np.ndarray] = None,
    camera_quality: Optional[np.ndarray] = None,
) -> np.ndarray:
    uv, depth = project_points(points_xyz, lidar2img)
    image_hw = np.asarray(image_hw, dtype=np.float32)
    online = np.ones(len(lidar2img), dtype=np.float32) if camera_online_mask is None else np.asarray(camera_online_mask, dtype=np.float32)
    quality = np.ones(len(lidar2img), dtype=np.float32) if camera_quality is None else np.asarray(camera_quality, dtype=np.float32)
    inside = (
        (depth > 0.1)
        & (uv[..., 0] >= 0)
        & (uv[..., 1] >= 0)
        & (uv[..., 0] < image_hw[:, None, 1])
        & (uv[..., 1] < image_hw[:, None, 0])
    )
    support = inside.astype(np.float32) * online[:, None] * quality[:, None]
    return 1.0 - np.prod(1.0 - support, axis=0)
