"""Registry bridge loaded by StreamPETR's ``custom_imports``.

Importing the official plugin first guarantees that the base detector, dataset,
heads, coders, samplers, and pipelines are registered before our subclasses.
"""

import projects.mmdet3d_plugin  # noqa: F401

from datasets.corruption import ApplyPartialObservation  # noqa: F401
from datasets.fixed_camera_fault import ApplyFixedCameraFault  # noqa: F401
from datasets.lidar_privileged_target import (  # noqa: F401
    AttachLiDARPointCounts,
    LiDARPrivilegedNuScenesDataset,
)
from datasets.nuscenes_wrapper import EvidenceNuScenesDataset  # noqa: F401
from hooks.evidence_trace_hook import EvidenceTraceHook  # noqa: F401
from models.streampetr_adapter import EvidenceConservingStreamPETRHead  # noqa: F401
from models.feq_head import FEQStreamPETRHead  # noqa: F401
from models.hard_positive_boundary_head import (  # noqa: F401
    HardPositiveBoundaryStreamPETRHead,
)
from models.lidar_privileged_target_evidence_head import (  # noqa: F401
    LiDARPrivilegedTargetEvidenceStreamPETRHead,
)

__all__ = [
    "ApplyPartialObservation",
    "ApplyFixedCameraFault",
    "EvidenceNuScenesDataset",
    "EvidenceTraceHook",
    "EvidenceConservingStreamPETRHead",
    "FEQStreamPETRHead",
    "HardPositiveBoundaryStreamPETRHead",
    "LiDARPrivilegedTargetEvidenceStreamPETRHead",
    "LiDARPrivilegedNuScenesDataset",
    "AttachLiDARPointCounts",
]
