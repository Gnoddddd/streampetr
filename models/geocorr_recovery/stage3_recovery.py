"""Structured Stage 3-B recovery contract over precomputed detector features."""

from dataclasses import dataclass
from typing import Optional, Union

from torch import Tensor, nn

from models.adapters import PreviousPrediction

from .historical_retrieval import HistoricalEvidenceRetriever, select_top_predictions
from .residual_recovery import RecoveryMLP
from .sparse_feature_writeback import SparseBilinearResidualWriteback


@dataclass(frozen=True)
class GeoCorrStage3BOutput:
    previous_predictions: Optional[PreviousPrediction]
    previous_historical_features: Tensor
    current_clean_fpn: Optional[Tensor]
    current_dirty_fpn: Optional[Tensor]
    q_clean: Tensor
    q_dirty: Tensor
    p_teacher: Optional[Tensor]
    p_student: Tensor
    normalized_student_probabilities: Tensor
    historical_retrieved_feature: Tensor
    confidence: Tensor
    query_valid: Tensor
    delta_q: Tensor
    q_recovered: Tensor
    recovered_current_fpn: Optional[Tensor]


class GeoCorrStage3BRecovery(nn.Module):
    """Apply retrieval, fixed confidence, residual recovery, and write-back.

    Detector/FPN extraction and Stage-2 correspondence stay outside this
    module. Their products are accepted explicitly so Stage 3-C can connect
    the three image branches without duplicating the established mechanisms.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        top_k: Union[int, str, None] = 25,
        recovery_zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.retriever = HistoricalEvidenceRetriever()
        self.recovery = RecoveryMLP(feature_dim, zero_init=recovery_zero_init)
        self.writeback = SparseBilinearResidualWriteback()

    def select_historical_anchors(
        self, prediction: PreviousPrediction
    ) -> PreviousPrediction:
        return select_top_predictions(prediction, self.top_k)

    def forward(
        self,
        q_dirty: Tensor,
        q_clean: Tensor,
        historical_features: Tensor,
        p_student: Tensor,
        correspondence_valid_mask: Tensor,
        p_teacher: Optional[Tensor] = None,
        previous_predictions: Optional[PreviousPrediction] = None,
        current_clean_fpn: Optional[Tensor] = None,
        current_dirty_fpn: Optional[Tensor] = None,
        projected_center_coords: Optional[Tensor] = None,
        projected_center_valid: Optional[Tensor] = None,
    ) -> GeoCorrStage3BOutput:
        if q_clean.shape != q_dirty.shape:
            raise ValueError("q_clean and q_dirty shapes must match")
        retrieval = self.retriever(
            p_student, historical_features, correspondence_valid_mask
        )
        recovery = self.recovery(
            q_dirty,
            retrieval["historical_retrieved_feature"],
            retrieval["confidence"],
        )
        recovered_fpn = current_dirty_fpn
        writeback_values = (
            current_dirty_fpn,
            projected_center_coords,
            projected_center_valid,
        )
        provided = tuple(value is not None for value in writeback_values)
        if any(provided) and not all(provided):
            raise ValueError(
                "dirty FPN, projected center coordinates, and validity are required together"
            )
        if all(provided):
            recovered_fpn = self.writeback(
                current_dirty_fpn,
                recovery["q_recovered"] - q_dirty,
                projected_center_coords,
                projected_center_valid,
                retrieval["confidence"],
            )
        return GeoCorrStage3BOutput(
            previous_predictions=previous_predictions,
            previous_historical_features=historical_features,
            current_clean_fpn=current_clean_fpn,
            current_dirty_fpn=current_dirty_fpn,
            q_clean=q_clean,
            q_dirty=q_dirty,
            p_teacher=p_teacher,
            p_student=p_student,
            normalized_student_probabilities=retrieval["normalized_probabilities"],
            historical_retrieved_feature=retrieval["historical_retrieved_feature"],
            confidence=retrieval["confidence"],
            query_valid=retrieval["query_valid"],
            delta_q=recovery["delta_q"],
            q_recovered=recovery["q_recovered"],
            recovered_current_fpn=recovered_fpn,
        )


__all__ = ["GeoCorrStage3BOutput", "GeoCorrStage3BRecovery"]
