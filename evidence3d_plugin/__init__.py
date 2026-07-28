"""Registry bridge loaded by StreamPETR's ``custom_imports``.

Importing the official plugin first guarantees that the base detector, dataset,
heads, coders, samplers, and pipelines are registered before our subclasses.
"""

import projects.mmdet3d_plugin  # noqa: F401

from datasets.corruption import ApplyPartialObservation  # noqa: F401
from datasets.nuscenes_wrapper import EvidenceNuScenesDataset  # noqa: F401
from hooks.evidence_trace_hook import EvidenceTraceHook  # noqa: F401
from hooks.reacquisition_curriculum_hook import (  # noqa: F401
    ReacquisitionCurriculumHook,
)
from models.streampetr_adapter import EvidenceConservingStreamPETRHead  # noqa: F401

__all__ = [
    "ApplyPartialObservation",
    "EvidenceNuScenesDataset",
    "EvidenceTraceHook",
    "ReacquisitionCurriculumHook",
    "EvidenceConservingStreamPETRHead",
]
