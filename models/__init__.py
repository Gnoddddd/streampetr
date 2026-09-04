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
    "Action",
    "PRESENT",
    "ABSENT",
    "UNOBSERVED",
]
