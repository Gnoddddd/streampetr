"""Registry bridge loaded by StreamPETR's ``custom_imports``.

Importing the official plugin first guarantees that the base detector, dataset,
heads, coders, samplers, and pipelines are registered before our subclasses.
"""

import projects.mmdet3d_plugin  # noqa: F401

from datasets.corruption import ApplyPartialObservation  # noqa: F401
from datasets.distillation import (  # noqa: F401
    FinalizePairedCleanImages,
    MatchFilteredGTInstanceTokens,
)
from datasets.nuscenes_wrapper import EvidenceNuScenesDataset  # noqa: F401
from hooks.ema_teacher_hook import TrainingOnlyEMATeacherHook  # noqa: F401
from hooks.evidence_trace_hook import EvidenceTraceHook  # noqa: F401
from models.observability_distillation import (  # noqa: F401
    ObservabilityDistillationPetr3D,
    ObservabilityDistillationStreamPETRHead,
)
from models.streampetr_adapter import EvidenceConservingStreamPETRHead  # noqa: F401

__all__ = [
    "ApplyPartialObservation",
    "FinalizePairedCleanImages",
    "MatchFilteredGTInstanceTokens",
    "EvidenceNuScenesDataset",
    "TrainingOnlyEMATeacherHook",
    "EvidenceTraceHook",
    "ObservabilityDistillationPetr3D",
    "ObservabilityDistillationStreamPETRHead",
    "EvidenceConservingStreamPETRHead",
]
