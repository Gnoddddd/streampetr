"""GeoCorr geometry and correspondence-only research modules."""

from .correlation_field import ObjectCentricCorrelationField, find_center_candidate_index
from .correspondence_distillation import correspondence_distillation, masked_softmax
from .descriptor_adapter import ResidualDescriptorAdapter
from .feature_sampler import sample_history_candidate_features
from .geometry_sampler import (
    GeometryCandidateSampler,
    propagate_previous_points,
    reverse_current_points,
    sample_candidate_features,
)
from .historical_retrieval import (
    HistoricalEvidenceRetriever,
    normalized_entropy_confidence,
    renormalize_correspondence,
    select_top_predictions,
    topk_prediction_indices,
)
from .residual_recovery import RecoveryMLP, recovery_feature_loss
from .sparse_feature_writeback import SparseBilinearResidualWriteback
from .stage3_recovery import GeoCorrStage3BOutput, GeoCorrStage3BRecovery

__all__ = [
    "GeometryCandidateSampler",
    "GeoCorrStage3BOutput",
    "GeoCorrStage3BRecovery",
    "HistoricalEvidenceRetriever",
    "ObjectCentricCorrelationField",
    "RecoveryMLP",
    "ResidualDescriptorAdapter",
    "SparseBilinearResidualWriteback",
    "correspondence_distillation",
    "find_center_candidate_index",
    "masked_softmax",
    "normalized_entropy_confidence",
    "propagate_previous_points",
    "reverse_current_points",
    "recovery_feature_loss",
    "renormalize_correspondence",
    "sample_candidate_features",
    "sample_history_candidate_features",
    "select_top_predictions",
    "topk_prediction_indices",
]
