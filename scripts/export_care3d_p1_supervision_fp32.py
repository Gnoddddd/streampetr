#!/usr/bin/env python3
"""Run the frozen CARE-3D P1 supervision exporter with FP32 query storage.

The original exporter intentionally used float16 for large cached tensors. P1
training replays the frozen StreamPETR classifier from the cached final decoder
query, so float16 round-trip quantization can exceed the frozen 5e-4 replay
check on rare rows. This wrapper promotes only ``clean_query`` and
``fault_query`` storage to float32 while preserving the original float16 source
feature/reliability cache. Detector/P0 weights, labels, splits, source bank,
losses and all decision gates are unchanged.

Old scene markers are treated as stale unless they carry STORAGE_POLICY. This
forces engineering smoke and formal train/val supervision to be regenerated
before P1 training restarts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as _np

import scripts.export_care3d_p1_supervision as _exporter


STORAGE_POLICY = "fp32_router_supervision_v1"


class _NumpyFP32Proxy:
    """Promote exporter query buffers to FP32 without changing source caches.

    The exporter references ``np.float16`` both when allocating buffers and in
    ``astype`` calls. Exposing float32 here prevents query quantization. The two
    large source-cache shapes are explicitly allocated as real float16 so their
    persisted precision remains identical to the original exporter.
    """

    float16 = _np.float32

    def zeros(self, shape, dtype=float, *args, **kwargs):
        dims = tuple(int(value) for value in shape)
        effective_dtype = dtype
        if dtype is self.float16:
            is_source_features = len(dims) == 4 and dims[-2:] == (3, 256)
            is_source_reliability = len(dims) == 3 and dims[-1] == 3
            if is_source_features or is_source_reliability:
                effective_dtype = _np.float16
        return _np.zeros(shape, dtype=effective_dtype, *args, **kwargs)

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
