import numpy as np

from analysis.camera_reliability import (
    aggregate_camera_attention,
    deployed_query_indices,
    rank_correlation,
    safe_correlation,
)


def test_camera_attention_aggregation_conserves_query_mass():
    weights = np.asarray([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    source = np.asarray([0, 0, 1, 2])
    result = aggregate_camera_attention(weights, source, camera_count=3)
    np.testing.assert_allclose(result.sum(1), weights.sum(1), atol=1e-7, rtol=0)
    np.testing.assert_allclose(result[0], [0.3, 0.3, 0.4], atol=1e-7)


def test_deployed_query_indices_use_class_flat_topk_and_range():
    logits = np.asarray([[0.0, 4.0], [3.0, -1.0], [2.0, 1.0]])
    boxes = np.asarray([[0, 0, 0], [99, 0, 0], [1, 1, 1]])
    result = deployed_query_indices(logits, boxes, 3, [-10, -10, -10, 10, 10, 10])
    # Query 1 has the second-highest class score but is filtered by range.
    np.testing.assert_array_equal(result, [0, 2])


def test_correlations_are_finite_for_varying_data_and_nan_for_constants():
    assert safe_correlation([1, 2, 3], [2, 4, 6]) == 1.0
    assert rank_correlation([1, 3, 2], [10, 30, 20]) == 1.0
    assert np.isnan(safe_correlation([1, 1, 1], [1, 2, 3]))


def test_trace_import_is_noop_without_opt_in(monkeypatch):
    monkeypatch.delenv("CAMERA_ATTENTION_TRACE_DIR", raising=False)
    import analysis.camera_attention_trace as trace

    assert callable(trace._install_trace)
