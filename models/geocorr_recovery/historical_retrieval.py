"""Masked historical evidence retrieval and fixed entropy confidence."""

from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from models.adapters import PreviousPrediction


def renormalize_correspondence(
    probabilities: Tensor, valid_mask: Tensor
) -> Tuple[Tensor, Tensor]:
    """Mask and renormalize ``P`` over ``J,T,Vh`` without all-invalid NaNs."""
    if probabilities.ndim != 6 or probabilities.shape != valid_mask.shape:
        raise ValueError(
            "probabilities and valid_mask must have shape [B,N,Vq,J,T,Vh]"
        )
    mask = valid_mask.bool()
    masked = probabilities * mask.to(probabilities.dtype)
    denominator = masked.sum(dim=(3, 4, 5), keepdim=True)
    query_valid = mask.reshape(*mask.shape[:3], -1).any(dim=-1)
    eps = torch.finfo(probabilities.dtype).eps
    normalized = torch.where(
        query_valid[..., None, None, None],
        masked / denominator.clamp_min(eps),
        torch.zeros_like(masked),
    )
    return normalized, query_valid


def normalized_entropy_confidence(
    probabilities: Tensor, valid_mask: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return fixed ``1 - normalized_entropy`` confidence ``[B,N,Vq,1]``.

    A valid singleton support has confidence one. An all-invalid query has
    confidence zero. Only valid support contributes to either entropy or its
    logarithmic normalization count.
    """
    normalized, query_valid = renormalize_correspondence(probabilities, valid_mask)
    eps = torch.finfo(normalized.dtype).eps
    point_entropy = torch.where(
        normalized > 0,
        -normalized * normalized.clamp_min(eps).log(),
        torch.zeros_like(normalized),
    )
    entropy = point_entropy.sum(dim=(3, 4, 5))
    support = valid_mask.bool().sum(dim=(3, 4, 5)).to(normalized.dtype)
    entropy_normalizer = support.clamp_min(1).log()
    normalized_entropy = torch.where(
        support > 1,
        entropy / entropy_normalizer.clamp_min(eps),
        torch.zeros_like(entropy),
    )
    confidence = torch.where(
        query_valid,
        1.0 - normalized_entropy,
        torch.zeros_like(normalized_entropy),
    ).clamp(0.0, 1.0)
    return confidence.unsqueeze(-1), normalized, query_valid


class HistoricalEvidenceRetriever(nn.Module):
    """Retrieve one descriptor per current query from historical candidates."""

    def forward(
        self,
        student_probabilities: Tensor,
        historical_features: Tensor,
        valid_mask: Tensor,
    ) -> Dict[str, Tensor]:
        if historical_features.ndim != 6:
            raise ValueError(
                "historical_features must have shape [B,N,J,T,Vh,C]"
            )
        if (
            student_probabilities.shape[:2] != historical_features.shape[:2]
            or student_probabilities.shape[3:] != historical_features.shape[2:-1]
        ):
            raise ValueError("probability and historical feature dimensions do not match")
        confidence, normalized, query_valid = normalized_entropy_confidence(
            student_probabilities, valid_mask
        )
        retrieved = torch.einsum(
            "bnvjtw,bnjtwc->bnvc", normalized, historical_features
        )
        retrieved = retrieved * query_valid[..., None].to(retrieved.dtype)
        return {
            "historical_retrieved_feature": retrieved,
            "confidence": confidence,
            "normalized_probabilities": normalized,
            "query_valid": query_valid,
        }


def topk_prediction_indices(scores: Tensor, top_k: Union[int, str, None] = 25) -> Tensor:
    """Select prediction indices by descending detector confidence."""
    if scores.ndim != 1:
        raise ValueError("prediction scores must have shape [N]")
    if top_k is None or top_k == "all":
        count = scores.numel()
    elif isinstance(top_k, int) and not isinstance(top_k, bool) and top_k > 0:
        count = min(top_k, scores.numel())
    else:
        raise ValueError("top_k must be a positive integer, 'all', or None")
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    return torch.topk(scores, k=count, largest=True, sorted=True).indices


def select_top_predictions(
    prediction: PreviousPrediction, top_k: Union[int, str, None] = 25
) -> PreviousPrediction:
    """Return the Top-K previous decoded predictions without score thresholding."""
    indices = topk_prediction_indices(prediction.score, top_k)
    velocity: Optional[Tensor] = prediction.velocity
    return PreviousPrediction(
        center_3d=prediction.center_3d[indices],
        size_3d=prediction.size_3d[indices],
        yaw=prediction.yaw[indices],
        velocity=None if velocity is None else velocity[indices],
        score=prediction.score[indices],
        label=prediction.label[indices],
        timestamp=prediction.timestamp,
        coordinate_frame=prediction.coordinate_frame,
    )


__all__ = [
    "HistoricalEvidenceRetriever",
    "normalized_entropy_confidence",
    "renormalize_correspondence",
    "select_top_predictions",
    "topk_prediction_indices",
]
