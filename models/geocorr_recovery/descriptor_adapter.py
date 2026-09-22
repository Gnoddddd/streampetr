"""Minimal trainable descriptor adapter for contaminated current features."""

import torch.nn.functional as F
from torch import Tensor, nn


class ResidualDescriptorAdapter(nn.Module):
    """``Linear(C,C) -> GELU -> Linear(C,C)`` residual descriptor."""

    def __init__(self, feature_dim: int = 256, eps: float = 1e-6) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.eps = float(eps)
        self.layers = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, descriptor: Tensor) -> Tensor:
        if descriptor.shape[-1] != self.feature_dim:
            raise ValueError("descriptor feature dimension does not match adapter")
        return F.normalize(descriptor + self.layers(descriptor), dim=-1, eps=self.eps)


__all__ = ["ResidualDescriptorAdapter"]
