"""Canonical, architecture-neutral object evidence containers."""

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from torch import Tensor


@dataclass
class ObjectEvidenceBatch:
    """Object tokens indexed by stable object identity, never detector slot identity."""

    tokens: Tensor
    batch_indices: Tensor
    object_ids: Sequence[Any]
    valid_mask: Tensor
    gt_indices: Optional[Tensor] = None
    labels: Optional[Tensor] = None
    centers: Optional[Tensor] = None

    def __post_init__(self) -> None:
        if self.tokens.ndim != 2 or self.tokens.shape[1] != 256:
            raise ValueError("tokens must have shape [N,256]")
        count = self.tokens.shape[0]
        if self.batch_indices.shape != (count,) or self.batch_indices.dtype != torch.long:
            raise ValueError("batch_indices must be LongTensor[N]")
        if self.valid_mask.shape != (count,) or self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be BoolTensor[N]")
        if len(self.object_ids) != count:
            raise ValueError("object_ids must have one entry per token")
        for name in ("gt_indices", "labels", "centers"):
            value = getattr(self, name)
            if value is not None and value.shape[0] != count:
                raise ValueError(f"{name} must have N entries")

    def select(self, indices: Tensor) -> "ObjectEvidenceBatch":
        positions = indices.long().to(self.tokens.device)
        object_ids = [self.object_ids[int(index)] for index in positions.cpu()]
        return ObjectEvidenceBatch(
            tokens=self.tokens[positions],
            batch_indices=self.batch_indices[positions],
            object_ids=object_ids,
            valid_mask=self.valid_mask[positions],
            gt_indices=None if self.gt_indices is None else self.gt_indices[positions],
            labels=None if self.labels is None else self.labels[positions],
            centers=None if self.centers is None else self.centers[positions],
        )

    @classmethod
    def empty(cls, like: Tensor) -> "ObjectEvidenceBatch":
        return cls(
            tokens=like.new_empty((0, 256)),
            batch_indices=torch.empty(0, dtype=torch.long, device=like.device),
            object_ids=[],
            valid_mask=torch.empty(0, dtype=torch.bool, device=like.device),
        )
