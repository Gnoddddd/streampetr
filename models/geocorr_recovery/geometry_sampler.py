"""Detector-independent propagation, projection, and local FPN sampling."""

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.adapters import PreviousPrediction


def _transform_points(transform: Tensor, points: Tensor) -> Tensor:
    homogeneous = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    return torch.matmul(transform, homogeneous.unsqueeze(-1)).squeeze(-1)[..., :3]


def _motion_displacement(points: Tensor, velocity: Optional[Tensor], delta_t: float) -> Tensor:
    """Broadcast previous-lidar xy velocity over ``[N,...,3]`` points."""
    if velocity is None:
        return torch.zeros_like(points)
    if velocity.ndim != 2 or velocity.shape != (points.shape[0], 2):
        raise ValueError("velocity must have shape [N_obj,2]")
    motion = torch.zeros(velocity.shape[0], 3, device=points.device, dtype=points.dtype)
    motion[:, :2] = velocity.to(points) * float(delta_t)
    return motion.reshape(points.shape[0], *([1] * (points.ndim - 2)), 3)


def propagate_previous_points(
    previous_points: Tensor,
    current_from_previous: Tensor,
    velocity_previous: Optional[Tensor],
    delta_t: float,
) -> Tensor:
    """Map ``[N,...,3]`` previous-lidar points into current lidar.

    Object velocity is expressed in the previous lidar frame and is added in
    that frame before applying ``current_from_previous``. Candidate offsets in
    ``previous_points`` therefore also belong to the previous lidar frame.
    """
    transform = current_from_previous.to(previous_points)
    if tuple(transform.shape) != (4, 4):
        raise ValueError("current_from_previous must have shape [4,4]")
    moved = previous_points + _motion_displacement(previous_points, velocity_previous, delta_t)
    return _transform_points(transform, moved)


def reverse_current_points(
    current_points: Tensor,
    current_from_previous: Tensor,
    velocity_previous: Optional[Tensor],
    delta_t: float,
) -> Tensor:
    """Invert :func:`propagate_previous_points` for ``[N,...,3]`` points.

    ``current_points`` and their offsets are in current lidar. The inverse ego
    transform first maps them to the motion-advanced previous lidar frame;
    ``velocity_previous * delta_t`` is then subtracted in previous lidar.
    """
    transform = current_from_previous.to(current_points)
    if tuple(transform.shape) != (4, 4):
        raise ValueError("current_from_previous must have shape [4,4]")
    moved_previous = _transform_points(torch.linalg.inv(transform), current_points)
    return moved_previous - _motion_displacement(moved_previous, velocity_previous, delta_t)


class GeometryCandidateSampler(nn.Module):
    """Build a fixed BEV grid and project it into augmented camera images.

    ``PreviousPrediction`` is in the previous lidar/ego-like Cartesian frame
    (x forward, y left, z up). ``current_from_previous`` is an explicit rigid
    4x4 transform into the current lidar frame.  Velocity is advanced in the
    previous frame before this transform. ``lidar2img`` maps current lidar
    homogeneous coordinates to image homogeneous coordinates.

    If ``lidar2img_is_augmented`` is true, matrices must be StreamPETR's
    post-IDA matrices (the pipeline already folds resize/crop/flip into the
    intrinsics). Otherwise ``image_aug_matrix`` is left-multiplied exactly once.
    """

    def __init__(
        self,
        candidate_offsets: Sequence[Sequence[float]],
        min_depth: float = 1e-5,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        offsets = torch.as_tensor(candidate_offsets, dtype=torch.float32)
        if tuple(offsets.shape) != (9, 2):
            raise ValueError("candidate_offsets must define a 3x3 grid with shape [9,2]")
        self.register_buffer("candidate_offsets", offsets)
        self.min_depth = float(min_depth)
        self.align_corners = bool(align_corners)

    def propagate(
        self,
        prediction: PreviousPrediction,
        current_from_previous: Tensor,
        delta_t: float,
    ) -> Tensor:
        """Return propagated gravity centers ``[N_obj,3]`` in current lidar."""
        return propagate_previous_points(
            prediction.center_3d, current_from_previous, prediction.velocity, delta_t
        )

    def reverse_candidates(
        self,
        current_candidates: Tensor,
        prediction: PreviousPrediction,
        current_from_previous: Tensor,
        delta_t: float,
    ) -> Tensor:
        """Return ``[N_obj,9,3]`` candidates in the previous lidar frame."""
        return reverse_current_points(
            current_candidates, current_from_previous, prediction.velocity, delta_t
        )

    def generate_candidates(self, propagated_anchor: Tensor) -> Tensor:
        """Return fixed-height 3x3 candidates ``[N_obj,9,3]``."""
        offsets = self.candidate_offsets.to(propagated_anchor)
        candidates = propagated_anchor[:, None, :].expand(-1, 9, -1).clone()
        candidates[..., :2] += offsets
        return candidates

    def project(
        self,
        candidates: Tensor,
        lidar2img: Tensor,
        image_shapes: Tensor,
        feature_shape: Tuple[int, int],
        image_aug_matrix: Optional[Tensor] = None,
        lidar2img_is_augmented: bool = True,
        padded_image_shapes: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Project candidates and return camera-first validity and coordinates.

        Args:
            candidates: Current-lidar points ``[N_obj,9,3]``.
            lidar2img: Camera matrices ``[N_cam,4,4]``.
            image_shapes: Valid augmented image ``[N_cam,2]`` as (H,W), before
                optional bottom/right padding.
            feature_shape: FPN (H,W), e.g. ``(16,44)``.
            padded_image_shapes: Optional padded canvas ``[N_cam,2]`` used for
                the image-to-FPN scale. Defaults to ``image_shapes``.
        """
        if lidar2img.ndim != 3 or tuple(lidar2img.shape[-2:]) != (4, 4):
            raise ValueError("lidar2img must have shape [N_cam,4,4]")
        num_cams = lidar2img.shape[0]
        if tuple(image_shapes.shape) != (num_cams, 2):
            raise ValueError("image_shapes must have shape [N_cam,2]")
        if padded_image_shapes is None:
            padded_image_shapes = image_shapes
        if tuple(padded_image_shapes.shape) != (num_cams, 2):
            raise ValueError("padded_image_shapes must have shape [N_cam,2]")
        matrices = lidar2img.to(candidates)
        if not lidar2img_is_augmented:
            if image_aug_matrix is None:
                raise ValueError("raw lidar2img requires image_aug_matrix")
            aug = image_aug_matrix.to(candidates)
            if aug.ndim == 2:
                aug = aug.expand(num_cams, -1, -1)
            augmented = matrices.clone()
            augmented[:, :3, :] = torch.matmul(aug, matrices[:, :3, :])
            matrices = augmented
        homogeneous = torch.cat(
            [candidates, torch.ones_like(candidates[..., :1])], dim=-1
        )
        projected = torch.einsum("cij,okj->ocki", matrices, homogeneous)
        depth = projected[..., 2]
        pixels = projected[..., :2] / depth.clamp_min(self.min_depth).unsqueeze(-1)
        heights = image_shapes[:, 0].to(candidates)[None, :, None]
        widths = image_shapes[:, 1].to(candidates)[None, :, None]
        padded_heights = padded_image_shapes[:, 0].to(candidates)[None, :, None]
        padded_widths = padded_image_shapes[:, 1].to(candidates)[None, :, None]
        finite = torch.isfinite(pixels).all(dim=-1) & torch.isfinite(depth)
        valid = (
            (depth > self.min_depth)
            & finite
            & (pixels[..., 0] >= 0)
            & (pixels[..., 0] < widths)
            & (pixels[..., 1] >= 0)
            & (pixels[..., 1] < heights)
        )
        feature_h, feature_w = feature_shape
        feature_coords = torch.empty_like(pixels)
        feature_coords[..., 0] = (
            (pixels[..., 0] + 0.5) * feature_w / padded_widths - 0.5
        )
        feature_coords[..., 1] = (
            (pixels[..., 1] + 0.5) * feature_h / padded_heights - 0.5
        )
        if self.align_corners:
            grid = torch.empty_like(feature_coords)
            grid[..., 0] = 2 * feature_coords[..., 0] / max(feature_w - 1, 1) - 1
            grid[..., 1] = 2 * feature_coords[..., 1] / max(feature_h - 1, 1) - 1
        else:
            grid = torch.empty_like(feature_coords)
            grid[..., 0] = 2 * (feature_coords[..., 0] + 0.5) / feature_w - 1
            grid[..., 1] = 2 * (feature_coords[..., 1] + 0.5) / feature_h - 1
        return {
            "camera_valid_mask": valid.permute(0, 2, 1),  # [O,J,Cam]
            "projected_points_2d": pixels.permute(0, 2, 1, 3),
            "feature_coords": feature_coords.permute(0, 2, 1, 3),
            "grid_sample_coords": grid.permute(0, 2, 1, 3),
            "depth": depth.permute(0, 2, 1),
        }

    def forward(
        self,
        prediction: PreviousPrediction,
        current_from_previous: Tensor,
        delta_t: float,
        lidar2img: Tensor,
        image_shapes: Tensor,
        feature_shape: Tuple[int, int],
        image_aug_matrix: Optional[Tensor] = None,
        lidar2img_is_augmented: bool = True,
        padded_image_shapes: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        anchor = self.propagate(prediction, current_from_previous, delta_t)
        candidates = self.generate_candidates(anchor)
        output = self.project(
            candidates,
            lidar2img,
            image_shapes,
            feature_shape,
            image_aug_matrix,
            lidar2img_is_augmented,
            padded_image_shapes,
        )
        output.update(propagated_anchor=anchor, candidate_points_3d=candidates)
        return output

    def forward_with_history(
        self,
        prediction: PreviousPrediction,
        current_from_previous: Tensor,
        delta_t: float,
        current_lidar2img: Tensor,
        current_image_shapes: Tensor,
        current_feature_shape: Tuple[int, int],
        previous_lidar2img: Tensor,
        previous_image_shapes: Tensor,
        previous_feature_shape: Tuple[int, int],
        current_padded_image_shapes: Optional[Tensor] = None,
        previous_padded_image_shapes: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Build current and t-1 geometry while preserving history dimension.

        Projection matrices must already include frame-specific image
        augmentation. Previous/history tensors have explicit ``T=1`` at
        dimension 2: ``[N_obj,J,T,V,...]``.
        """
        anchor = self.propagate(prediction, current_from_previous, delta_t)
        current_candidates = self.generate_candidates(anchor)
        previous_candidates = self.reverse_candidates(
            current_candidates, prediction, current_from_previous, delta_t
        )
        current = self.project(
            current_candidates,
            current_lidar2img,
            current_image_shapes,
            current_feature_shape,
            padded_image_shapes=current_padded_image_shapes,
        )
        previous = self.project(
            previous_candidates,
            previous_lidar2img,
            previous_image_shapes,
            previous_feature_shape,
            padded_image_shapes=previous_padded_image_shapes,
        )
        return {
            "propagated_anchor": anchor,
            "current_candidate_points_3d": current_candidates,
            "previous_candidate_points_3d": previous_candidates.unsqueeze(2),
            "current_projected_points": current["projected_points_2d"],
            "previous_projected_points": previous["projected_points_2d"].unsqueeze(2),
            "current_feature_coords": current["feature_coords"],
            "previous_feature_coords": previous["feature_coords"].unsqueeze(2),
            "current_grid_coords": current["grid_sample_coords"],
            "previous_grid_coords": previous["grid_sample_coords"].unsqueeze(2),
            "current_valid_mask": current["camera_valid_mask"],
            "previous_valid_mask": previous["camera_valid_mask"].unsqueeze(2),
        }


def sample_candidate_features(
    features: Tensor,
    grid_sample_coords: Tensor,
    valid_mask: Optional[Tensor] = None,
    align_corners: bool = False,
) -> Tensor:
    """Sample ``[B,Cam,C,H,W]`` into ``[B,N_obj,9,Cam,C]``."""
    if features.ndim != 5 or grid_sample_coords.ndim != 5:
        raise ValueError("features and coordinates must be 5D tensors")
    batch, cameras, channels, height, width = features.shape
    if grid_sample_coords.shape[0] != batch or grid_sample_coords.shape[3] != cameras:
        raise ValueError("batch/camera dimensions do not match")
    objects, candidates = grid_sample_coords.shape[1:3]
    grid = grid_sample_coords.permute(0, 3, 1, 2, 4).reshape(
        batch * cameras, objects, candidates, 2
    )
    flat_features = features.reshape(batch * cameras, channels, height, width)
    sampled = F.grid_sample(
        flat_features,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners,
    )
    sampled = sampled.reshape(batch, cameras, channels, objects, candidates)
    sampled = sampled.permute(0, 3, 4, 1, 2).contiguous()
    if valid_mask is not None:
        sampled = sampled * valid_mask[..., None].to(sampled.dtype)
    return sampled


__all__ = [
    "GeometryCandidateSampler",
    "propagate_previous_points",
    "reverse_current_points",
    "sample_candidate_features",
]
