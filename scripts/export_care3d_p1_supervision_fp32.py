#!/usr/bin/env python3
"""Run the frozen CARE-3D P1 supervision exporter with FP32 storage.

The original exporter intentionally used float16 for large cached tensors.  P1
training replays the frozen StreamPETR classifier from the cached final decoder
query, so float16 round-trip quantization can exceed the frozen 5e-4 replay
check on rare rows.  This wrapper changes only cache storage precision for
arrays that the exporter would otherwise create as float16.  Detector/P0
weights, labels, splits, source bank, losses and all decision gates are
unchanged.

Old scene markers are treated as stale unless they carry STORAGE_POLICY.  This
forces engineering smoke and formal train/val supervision to be regenerated
before P1 training restarts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as _np

import scripts.export_care3d_p1_supervision as _exporter


STORAGE_POLICY = "fp32_router_supervision_v1"


class _NumpyFP32Proxy:
    """Delegate NumPy except that exporter requests for float16 become float32."""

    float16 = _np.float32

    def __getattr__(self, name):
        return getattr(_np, name)


def _marker_valid(path: Path, validation: dict) -> bool:
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    return bool(
        value.get("complete")
        and value.get("schema_version") == _exporter.SCHEMA
        and value.get("scene_manifest_sha256") == validation["scene_manifest_sha256"]
        and value.get("query_collision_policy") == _exporter.QUERY_COLLISION_POLICY
        and value.get("storage_precision_policy") == STORAGE_POLICY
    )


def _atomic_json_with_storage_policy(path: Path, value: object) -> None:
    if isinstance(value, dict) and path.name.endswith(".complete.json"):
        value = dict(value)
        value["storage_precision_policy"] = STORAGE_POLICY
    _ORIGINAL_ATOMIC_JSON(path, value)


_ORIGINAL_ATOMIC_JSON = _exporter.atomic_json
_exporter.np = _NumpyFP32Proxy()
_exporter.marker_valid = _marker_valid
_exporter.atomic_json = _atomic_json_with_storage_policy


if __name__ == "__main__":
    _exporter.main()
