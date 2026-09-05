"""Minimal Counterfactual Temporal Evidence Preservation objective."""

from __future__ import annotations

import torch


def ctep_term(reference_score: torch.Tensor, target_logit: torch.Tensor) -> torch.Tensor:
    """Unit-weight one-sided preservation with a detached reference."""
    return torch.relu(reference_score.detach() - target_logit.sigmoid())


def disabled_detection_loss(detection_loss: torch.Tensor) -> torch.Tensor:
    """The disabled path is the original tensor, without arithmetic."""
    return detection_loss

