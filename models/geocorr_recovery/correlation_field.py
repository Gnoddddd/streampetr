"""Object-centric current-center to historical-candidate correlation."""

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .descriptor_adapter import ResidualDescriptorAdapter


def find_center_candidate_index(
    candidate_offsets: Sequence[Sequence[float]], atol: float = 1e-7
) -> int:
    """Find the unique ``(0,0)`` offset without assuming its list index."""
    offsets = torch.as_tensor(candidate_offsets, dtype=torch.float32)
    if offsets.ndim != 2 or offsets.shape[1] != 2:
        raise ValueError("candidate_offsets must have shape [J,2]")
    matches = torch.isclose(offsets, torch.zeros_like(offsets), atol=atol).all(dim=1)
    indices = matches.nonzero(as_tuple=False).flatten()
    if indices.numel() != 1:
        raise ValueError("candidate_offsets must contain exactly one (0,0) offset")
    return int(indices.item())


class ObjectCentricCorrelationField(nn.Module):
    """Build cosine logits with shape ``[B,N,Vq,J,T,Vh]``.

    Only the current ``(0,0)`` candidate supplies queries. Every historical
    position, time and view supplies keys. There is no image-wide or
    cross-object all-pairs matching.
    """

    def __init__(
        self,
        candidate_offsets: Sequence[Sequence[float]],
        feature_dim: int = 256,
        geometry_prior_beta: float = 0.0,
        detach_teacher: bool = True,
        detach_history: bool = True,
        detach_dirty_base: bool = True,
    ) -> None:
        super().__init__()
        offsets = torch.as_tensor(candidate_offsets, dtype=torch.float32)
        self.register_buffer("candidate_offsets", offsets)
        self.center_candidate_index = find_center_candidate_index(candidate_offsets)
        self.adapter = ResidualDescriptorAdapter(feature_dim)
        self.geometry_prior_beta = float(geometry_prior_beta)
        self.detach_teacher = bool(detach_teacher)
        self.detach_history = bool(detach_history)
        self.detach_dirty_base = bool(detach_dirty_base)

    def forward(
        self,
        current_clean_tokens: Tensor,
        current_dirty_tokens: Tensor,
        history_clean_tokens: Tensor,
        current_valid_mask: Tensor,
        history_valid_mask: Tensor,
        object_valid_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if current_clean_tokens.shape != current_dirty_tokens.shape:
            raise ValueError("clean and dirty current token shapes must match")
        if current_clean_tokens.ndim != 5 or history_clean_tokens.ndim != 6:
            raise ValueError("current/history tokens must be 5D/6D")
        if history_clean_tokens.shape[3] != 1:
            raise ValueError("GeoCorr Stage 2 V1 supports exactly one history frame")
        if current_valid_mask.shape != current_clean_tokens.shape[:-1]:
            raise ValueError("current_valid_mask shape does not match tokens")
        if history_valid_mask.shape != history_clean_tokens.shape[:-1]:
            raise ValueError("history_valid_mask shape does not match tokens")
        if current_clean_tokens.shape[:3] != history_clean_tokens.shape[:3]:
            raise ValueError("current/history B,N,J dimensions must match")
        center = self.center_candidate_index
        clean_query = current_clean_tokens[:, :, center]
        dirty_query = current_dirty_tokens[:, :, center]
        history_key = history_clean_tokens
        if self.detach_teacher:
            clean_query = clean_query.detach()
        if self.detach_dirty_base:
            dirty_query = dirty_query.detach()
        if self.detach_history:
            history_key = history_key.detach()
        clean_descriptor = F.normalize(clean_query, dim=-1)
        dirty_descriptor = self.adapter(dirty_query)
        history_descriptor = F.normalize(history_key, dim=-1)
        teacher_logits = torch.einsum(
            "bnvc,bnjtwc->bnvjtw", clean_descriptor, history_descriptor
        )
        student_logits = torch.einsum(
            "bnvc,bnjtwc->bnvjtw", dirty_descriptor, history_descriptor
        )
        if self.geometry_prior_beta:
            prior = -self.geometry_prior_beta * self.candidate_offsets.square().sum(-1)
            prior = prior.to(teacher_logits).reshape(1, 1, 1, -1, 1, 1)
            teacher_logits = teacher_logits + prior
            student_logits = student_logits + prior
        query_valid = current_valid_mask[:, :, center]
        if object_valid_mask is not None:
            if object_valid_mask.shape != query_valid.shape[:2]:
                raise ValueError("object_valid_mask must have shape [B,N]")
            query_valid = query_valid & object_valid_mask[:, :, None]
        valid_mask = query_valid[:, :, :, None, None, None] & history_valid_mask[:, :, None]
        return {
            "teacher_logits": teacher_logits,
            "student_logits": student_logits,
            "valid_mask": valid_mask,
            "query_valid": query_valid,
            "clean_descriptor": clean_descriptor,
            "dirty_descriptor": dirty_descriptor,
            "history_descriptor": history_descriptor,
        }


__all__ = ["ObjectCentricCorrelationField", "find_center_candidate_index"]
