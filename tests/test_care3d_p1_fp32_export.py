import json

import numpy as np

from scripts.export_care3d_p1_supervision_fp32 import (
    STORAGE_POLICY,
    _NumpyFP32Proxy,
    _marker_valid,
    _exporter,
)


def test_fp32_proxy_promotes_queries_but_preserves_source_cache_precision():
    proxy = _NumpyFP32Proxy()

    clean_query = proxy.zeros((2, 256), dtype=proxy.float16)
    fault_query = proxy.zeros((2, 3, 256), dtype=proxy.float16)
    source_features = proxy.zeros((2, 3, 3, 256), dtype=proxy.float16)
    source_reliability = proxy.zeros((2, 3, 3), dtype=proxy.float16)

    assert clean_query.dtype == np.float32
    assert fault_query.dtype == np.float32
    assert source_features.dtype == np.float16
    assert source_reliability.dtype == np.float16
    assert proxy.asarray([1, 2, 3]).dtype == np.int64


def test_fp32_marker_policy_invalidates_old_supervision(tmp_path):
    path = tmp_path / "scene.complete.json"
    validation = {"scene_manifest_sha256": "manifest"}
    base = {
        "complete": True,
        "schema_version": _exporter.SCHEMA,
        "scene_manifest_sha256": "manifest",
        "query_collision_policy": _exporter.QUERY_COLLISION_POLICY,
    }
    path.write_text(json.dumps(base))
    assert _marker_valid(path, validation) is False
    base["storage_precision_policy"] = STORAGE_POLICY
    path.write_text(json.dumps(base))
    assert _marker_valid(path, validation) is True
