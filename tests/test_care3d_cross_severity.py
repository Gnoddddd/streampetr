import numpy as np
import pytest

from analysis.care3d_cross_severity import (
    MAIN_PROTOCOL_INDEX,
    TRANSFER_PROTOCOLS,
    assert_exact_sample_alignment,
    assert_predictor_inputs_exact,
    transfer_gate_flags,
)


def _inputs():
    return {
        "object_features": np.zeros((2, 256), dtype=np.float32),
        "temporal_features": np.zeros((2, 256), dtype=np.float32),
        "decision_features": np.zeros((2, 21), dtype=np.float32),
        "camera_support": np.zeros((2, 6), dtype=np.float32),
        "camera_quality": np.ones((2, 6), dtype=np.float32),
    }


def test_transfer_protocols_map_to_frozen_main_heads():
    assert TRANSFER_PROTOCOLS == ("blur_s03", "dark_s03")
    assert MAIN_PROTOCOL_INDEX == {"blur_s03": 0, "dark_s03": 2}


def test_exact_sample_alignment_rejects_reordering():
    assert_exact_sample_alignment(["a", "b"], ["a", "b"])
    with pytest.raises(RuntimeError):
        assert_exact_sample_alignment(["a", "b"], ["b", "a"])


def test_predictor_input_identity_is_bitwise():
    main = _inputs()
    transfer = {key: value.copy() for key, value in main.items()}
    assert_predictor_inputs_exact(main, transfer)
    transfer["object_features"][0, 0] = np.float32(1e-7)
    with pytest.raises(RuntimeError):
        assert_predictor_inputs_exact(main, transfer)


def test_transfer_gate_requires_all_stability_terms():
    point = {"spearman": 0.2, "decile_drop_delta": 0.03, "auroc": 0.8}
    scene = {
        "spearman_ci_low": 0.1,
        "decile_drop_delta_ci_low": 0.01,
        "auroc_ci_low": 0.7,
        "auprc_minus_base_rate_ci_low": 0.02,
    }
    instance = dict(scene)
    flags = transfer_gate_flags(
        point,
        scene,
        instance,
        min_boundary_auroc=0.65,
        min_boundary_auroc_ci_low=0.50,
        min_auprc_excess_ci_low=0.0,
    )
    assert flags["seed_protocol_pass"] is True

    failed_scene = dict(scene)
    failed_scene["spearman_ci_low"] = -0.001
    flags = transfer_gate_flags(
        point,
        failed_scene,
        instance,
        min_boundary_auroc=0.65,
        min_boundary_auroc_ci_low=0.50,
        min_auprc_excess_ci_low=0.0,
    )
    assert flags["rank_pass"] is False
    assert flags["seed_protocol_pass"] is False
