"""Sparse bilinear residual add-back into the current dirty FPN."""

import torch
from torch import Tensor, nn


class SparseBilinearResidualWriteback(nn.Module):
    """Scatter query residuals with confidence-weighted mean aggregation.

    ``feature_coords`` are continuous FPN pixel coordinates ``(x,y)``. The
    query view dimension must equal the FPN view dimension. The input residual
    is normally ``q_recovered - q_dirty`` and is added to, never substituted
    for, the original dirty FPN.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        current_dirty_fpn: Tensor,
        query_residual: Tensor,
        feature_coords: Tensor,
        valid_mask: Tensor,
        confidence: Tensor,
    ) -> Tensor:
        if current_dirty_fpn.ndim != 5:
            raise ValueError("current_dirty_fpn must have shape [B,V,C,H,W]")
        if query_residual.ndim != 4:
            raise ValueError("query_residual must have shape [B,N,V,C]")
        batch, views, channels, height, width = current_dirty_fpn.shape
        if query_residual.shape[0] != batch or query_residual.shape[2:] != (views, channels):
            raise ValueError("query residual batch/view/channel dimensions do not match FPN")
        expected_coords = query_residual.shape[:3] + (2,)
        if feature_coords.shape != expected_coords:
            raise ValueError("feature_coords must have shape [B,N,V,2]")
        if valid_mask.shape == query_residual.shape[:3] + (1,):
            valid_mask = valid_mask.squeeze(-1)
        if valid_mask.shape != query_residual.shape[:3]:
            raise ValueError("valid_mask must have shape [B,N,V]")
        if confidence.shape != query_residual.shape[:3] + (1,):
            raise ValueError("confidence must have shape [B,N,V,1]")

        finite = torch.isfinite(feature_coords).all(dim=-1)
        x = torch.where(finite, feature_coords[..., 0], torch.zeros_like(feature_coords[..., 0]))
        y = torch.where(finite, feature_coords[..., 1], torch.zeros_like(feature_coords[..., 1]))
        x0 = torch.floor(x)
        y0 = torch.floor(y)
        x1 = x0 + 1
        y1 = y0 + 1
        accumulator = current_dirty_fpn.new_zeros(batch, views, channels, height * width)
        normalizer = current_dirty_fpn.new_zeros(batch, views, 1, height * width)
        base_valid = valid_mask.bool() & finite & (confidence.squeeze(-1) > 0)
        residual = query_residual.permute(0, 2, 3, 1)
        confidence_weight = confidence.squeeze(-1).permute(0, 2, 1)

        neighbors = (
            (x0, y0, (x1 - x) * (y1 - y)),
            (x1, y0, (x - x0) * (y1 - y)),
            (x0, y1, (x1 - x) * (y - y0)),
            (x1, y1, (x - x0) * (y - y0)),
        )
        for neighbor_x, neighbor_y, bilinear_weight in neighbors:
            inside = (
                base_valid
                & (neighbor_x >= 0)
                & (neighbor_x < width)
                & (neighbor_y >= 0)
                & (neighbor_y < height)
            )
            weight = (
                bilinear_weight * inside.to(bilinear_weight.dtype)
            ).permute(0, 2, 1) * confidence_weight
            flat_index = (
                neighbor_y.clamp(0, height - 1).long() * width
                + neighbor_x.clamp(0, width - 1).long()
            ).permute(0, 2, 1)
            expanded_index = flat_index[:, :, None].expand(-1, -1, channels, -1)
            accumulator.scatter_add_(
                -1, expanded_index, residual * weight[:, :, None]
            )
            normalizer.scatter_add_(
                -1, flat_index[:, :, None], weight[:, :, None]
            )

        averaged = torch.where(
            normalizer > self.eps,
            accumulator / normalizer.clamp_min(self.eps),
            torch.zeros_like(accumulator),
        ).reshape(batch, views, channels, height, width)
        return current_dirty_fpn + averaged


__all__ = ["SparseBilinearResidualWriteback"]
