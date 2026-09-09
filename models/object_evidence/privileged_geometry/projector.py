"""Training-only camera-to-LiDAR token projector."""

from torch import nn


class PrivilegedGeometryProjector(nn.Sequential):
    def __init__(self, teacher_dim: int):
        if teacher_dim <= 0:
            raise ValueError("teacher_dim must be positive")
        super().__init__(nn.LayerNorm(256), nn.Linear(256, teacher_dim, bias=False))
