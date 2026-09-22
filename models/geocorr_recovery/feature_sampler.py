"""Geometry-aligned current and historical FPN token sampling."""

from typing import Optional

from torch import Tensor

from .geometry_sampler import sample_candidate_features


def sample_history_candidate_features(
    features: Tensor,
    grid_sample_coords: Tensor,
    valid_mask: Optional[Tensor] = None,
    align_corners: bool = False,
) -> Tensor:
    """Sample ``[B,T,V,C,H,W]`` into ``[B,N,J,T,V,C]`` without squeezing T."""
    if features.ndim != 6 or grid_sample_coords.ndim != 6:
        raise ValueError("history features and coordinates must be 6D")
    batch, time, views, channels, height, width = features.shape
    if time != 1:
        raise ValueError("GeoCorr Stage 2 V1 supports exactly one history frame")
    if grid_sample_coords.shape[0] != batch or grid_sample_coords.shape[3] != time:
        raise ValueError("batch/time dimensions do not match")
    if grid_sample_coords.shape[4] != views:
        raise ValueError("history view dimensions do not match")
    objects, candidates = grid_sample_coords.shape[1:3]
    flat_features = features.reshape(batch * time, views, channels, height, width)
    flat_grid = grid_sample_coords.permute(0, 3, 1, 2, 4, 5).reshape(
        batch * time, objects, candidates, views, 2
    )
    flat_mask = None
    if valid_mask is not None:
        flat_mask = valid_mask.permute(0, 3, 1, 2, 4).reshape(
            batch * time, objects, candidates, views
        )
    sampled = sample_candidate_features(
        flat_features, flat_grid, flat_mask, align_corners=align_corners
    )
    return sampled.reshape(batch, time, objects, candidates, views, channels).permute(
        0, 2, 3, 1, 4, 5
    ).contiguous()


__all__ = ["sample_history_candidate_features"]
