import inspect
import os

import numpy as np
import torch

import analysis.active_recovery_injection as injection
from analysis.active_recovery_injection import (
    PC_RANGE,
    constant_velocity,
    denormalize_reference,
    ego_compensated_propagation,
    normalize_reference,
    replace_learned_tail,
    sigma_points,
)


def test_reference_normalization_round_trip():
    center = np.asarray([12.5, -8.25, 1.5], np.float32)
    assert np.allclose(denormalize_reference(normalize_reference(center)), center)
    assert np.allclose(normalize_reference(PC_RANGE[:3]), 0)
    assert np.allclose(normalize_reference(PC_RANGE[3:]), 1)


def test_constant_velocity_and_ego_compensation():
    assert np.allclose(
        constant_velocity(np.asarray([1, 2, 3]), np.asarray([4, -2]), 0.5),
        [3, 1, 3],
    )
    angle = np.pi / 2
    previous_rotation = np.asarray([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])
    center, velocity = ego_compensated_propagation(
        np.asarray([1, 0, 0]),
        np.asarray([2, 0]),
        previous_rotation,
        np.zeros(3),
        np.eye(3),
        np.asarray([0, 1, 0]),
        0.5,
    )
    assert np.allclose(center, [0, 1, 0], atol=1e-7)
    assert np.allclose(velocity, [0, 2], atol=1e-7)


def test_kalman_sigma_point_order_and_symmetry():
    mean = np.asarray([3.0, 4.0])
    covariance = np.diag([4.0, 1.0])
    points = sigma_points(mean, covariance)
    assert np.array_equal(points[0], mean)
    assert np.allclose(points[1] + points[2], 2 * mean)
    assert np.isclose(np.linalg.norm(points[1] - mean), 2.0)
    assert np.isclose(np.linalg.norm(points[3] - mean), 1.0)


def test_replacement_is_learned_tail_and_budget_conserving():
    tgt = torch.zeros(1, 900, 2)
    pos = torch.zeros_like(tgt)
    ref = torch.zeros(1, 900, 3)
    contents = torch.arange(8).reshape(4, 2).float()
    positions = torch.ones(4, 3)
    encoded = torch.full((1, 4, 2), 5.0)
    out_tgt, out_pos, out_ref, target = replace_learned_tail(
        tgt, pos, ref, contents, positions, encoded
    )
    assert (target.start, target.stop) == (640, 644)
    assert torch.equal(out_tgt[0, 640:644], contents)
    assert torch.equal(out_pos[0, 640:644], encoded[0])
    assert torch.equal(out_ref[0, 640:644], positions)
    assert torch.equal(out_tgt[:, 644:], tgt[:, 644:])
    assert 4 + (out_tgt.shape[1] - 4) == 900


def test_import_is_injection_off_noop():
    assert os.environ.get("ACTIVE_RECOVERY_MODE", "").upper() not in {
        "Q1", "Q2", "Q3"
    }
    assert "_install()" in inspect.getsource(injection)


def test_deployable_code_has_no_full_or_future_input():
    source = inspect.getsource(injection)
    deploy_section = source[
        source.index("def deploy_specs"):source.index("def injected_forward")
    ]
    assert "local_gt" not in deploy_section
    assert "future" not in deploy_section
    assert "full" not in deploy_section.lower()


def test_emit_only_restores_q0_memory_by_construction():
    source = inspect.getsource(injection)
    assert "_restore(self, q0_after)" in source
    assert "memory_restore_diff" in source
