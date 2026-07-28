"""Keep/Recover/Defer decisions derived directly from evidence state."""

from __future__ import annotations

from enum import IntEnum
from typing import Dict

import torch
from torch import Tensor, nn


class Action(IntEnum):
    KEEP = 0
    RECOVER = 1
    DEFER = 2


class KeepRecoverDeferPolicy(nn.Module):
    def __init__(
        self,
        keep_observability: float = 0.45,
        keep_presence: float = 0.55,
        keep_max_uncertainty: float = 0.55,
        recover_presence: float = 0.55,
        recover_max_uncertainty: float = 0.80,
        recover_max_age: int = 3,
        recover_min_prior_strength: float = 0.5,
        strong_negative: float = 0.60,
        recover_score_scale: float = 0.75,
        defer_score_scale: float = 0.20,
    ) -> None:
        super().__init__()
        self.keep_observability = float(keep_observability)
        self.keep_presence = float(keep_presence)
        self.keep_max_uncertainty = float(keep_max_uncertainty)
        self.recover_presence = float(recover_presence)
        self.recover_max_uncertainty = float(recover_max_uncertainty)
        self.recover_max_age = int(recover_max_age)
        self.recover_min_prior_strength = float(recover_min_prior_strength)
        self.strong_negative = float(strong_negative)
        self.recover_score_scale = float(recover_score_scale)
        self.defer_score_scale = float(defer_score_scale)

    def forward(
        self,
        observability: Tensor,
        existence_probability: Tensor,
        uncertainty: Tensor,
        age_since_observation: Tensor,
        negative_probability: Tensor,
        prior_strength: Tensor,
        use_strong_negative: bool = True,
    ) -> Dict[str, Tensor]:
        keep = (
            (observability >= self.keep_observability)
            & (existence_probability >= self.keep_presence)
            & (uncertainty <= self.keep_max_uncertainty)
        )
        negative_ok = (
            negative_probability < self.strong_negative
            if use_strong_negative
            else torch.ones_like(
                negative_probability,
                dtype=torch.bool,
            )
        )

        recover = (
            (~keep)
            & (observability < self.keep_observability)
            & (prior_strength >= self.recover_min_prior_strength)
            & (existence_probability >= self.recover_presence)
            & (uncertainty <= self.recover_max_uncertainty)
            & (age_since_observation <= self.recover_max_age)
            & negative_ok
        )
        action = torch.full_like(
            existence_probability, int(Action.DEFER), dtype=torch.long
        )
        action = torch.where(recover, torch.full_like(action, int(Action.RECOVER)), action)
        action = torch.where(keep, torch.full_like(action, int(Action.KEEP)), action)

        score_scale = torch.full_like(
            existence_probability, self.defer_score_scale
        )
        score_scale = torch.where(
            recover,
            torch.full_like(score_scale, self.recover_score_scale),
            score_scale,
        )
        score_scale = torch.where(keep, torch.ones_like(score_scale), score_scale)
        write_mask = action != int(Action.DEFER)
        return {
            "action": action,
            "score_scale": score_scale,
            "write_mask": write_mask,
        }
