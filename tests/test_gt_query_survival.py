import ast
from pathlib import Path

import numpy as np

from analysis.gt_query_survival import (
    flat_class_rank,
    geometry_statistics,
    projected_feature_support,
    wrap_angle,
)


def test_flat_class_rank_uses_deployment_query_class_pairs():
    logits = np.asarray([[1.0, 4.0], [3.0, 2.0]])
    assert flat_class_rank(logits, 0, 1) == 1
    assert flat_class_rank(logits, 1, 0) == 2
    assert flat_class_rank(logits, 0, 0) == 4


def test_geometry_statistics_counts_near_queries_and_best_cost():
    boxes = np.asarray([
        [0.5, 0, 0, 2, 4, 1.5, 0],
        [1.5, 0, 0, 2, 4, 1.5, 0],
        [3.0, 0, 0, 2, 4, 1.5, 0],
    ])
    result = geometry_statistics(boxes, [0, 0, 0], [2, 4, 1.5], 0)
    assert result["near_count"] == 2
    assert result["best_query"] == 0
    assert result["center_distance"] == 0.5


def test_projection_support_uses_camera_and_spatial_token_source():
    matrices = np.repeat(np.eye(4)[None], 2, axis=0)
    # Project [2,2,1] to pixel (2,2) in a 4x4 image/feature map.
    result = projected_feature_support(
        [2, 2, 1], matrices, [10, 16 + 15], [3.0, 9.0], [4, 4], [4, 4], 0.1
    )
    assert result["visible_cameras"] == 2
    assert result["supported_cameras"] == 1
    assert result["best_feature_norm"] == 3.0


def test_yaw_wrap_is_periodic():
    assert abs(wrap_angle(2 * np.pi + 0.2) - 0.2) < 1e-8


def test_trace_is_read_only_and_returns_original_result():
    source = Path("analysis/gt_query_survival_trace.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    attributes = {
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "register_buffer" not in attributes
    assert "register_parameter" not in attributes
    assert "return result" in source
