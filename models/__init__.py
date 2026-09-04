"""Core Evidence3D modules.

The StreamPETR adapter is intentionally not imported here so that the core
modules remain unit-testable without the legacy OpenMMLab stack installed.
"""

from .observability_head import GeometricObservabilityHead
from .ternary_objectness import (
    ABSENT,
    PRESENT,
    UNOBSERVED,
    ObservabilityConditionedTernaryLoss,
    TernaryObjectnessHead,
)
from .temporal_update import EvidenceConservingTemporalUpdate
from .keep_recover_defer import Action, KeepRecoverDeferPolicy
from .evidence_ledger import EvidenceLedger
from .innovation import (
    ReliabilityCalibratedInnovation,
    normalized_js_divergence,
    wrapped_angle_difference,
)
from .care3d import (
    CARE3DCore,
    CARE3DStateEncoder,
    CounterfactualVulnerabilityHead,
    CounterfactualVulnerabilityLoss,
    SparseEvidenceRouter,
)

__all__ = [
    "GeometricObservabilityHead",
    "TernaryObjectnessHead",
    "ObservabilityConditionedTernaryLoss",
    "EvidenceConservingTemporalUpdate",
    "KeepRecoverDeferPolicy",
    "EvidenceLedger",
    "ReliabilityCalibratedInnovation",
    "normalized_js_divergence",
    "wrapped_angle_difference",
    "CARE3DCore",
    "CARE3DStateEncoder",
    "CounterfactualVulnerabilityHead",
    "CounterfactualVulnerabilityLoss",
    "SparseEvidenceRouter",
    "Action",
    "PRESENT",
    "ABSENT",
    "UNOBSERVED",
]
from .feq_losses import (  # noqa: F401
    adjacent_survival_loss,
    greedy_auxiliary_assignment,
    ranking_loss,
    supervision_weights,
)
