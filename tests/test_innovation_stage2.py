"""S2.3 reliability-calibrated temporal innovation contracts."""

from __future__ import annotations

import io
import math

import pytest
import torch
from torch import nn

from models.evidence_ledger import EvidenceLedger
from models.innovation import (
    ReliabilityCalibratedInnovation,
    normalized_js_divergence,
    wrapped_angle_difference,
)
from models.temporal_update import EvidenceConservingTemporalUpdate


def _inputs(dtype=torch.float32):
    return dict(
        current_source=torch.tensor([[[1.0, 0.0]]], dtype=dtype),
        previous_source=torch.tensor([[[1.0, 0.0]]], dtype=dtype),
        current_feature=torch.tensor([[[1.0, 0.0]]], dtype=dtype),
        previous_feature=torch.tensor([[[1.0, 0.0]]], dtype=dtype),
        current_geometry=torch.zeros(1, 1, 6, dtype=dtype),
        previous_geometry=torch.zeros(1, 1, 6, dtype=dtype),
        current_class_probability=torch.tensor(
            [[[0.9, 0.1]]], dtype=dtype
        ),
        previous_class_probability=torch.tensor(
            [[[0.9, 0.1]]], dtype=dtype
        ),
        current_ternary_probability=torch.tensor(
            [[[0.9, 0.05, 0.05]]], dtype=dtype
        ),
        previous_ternary_probability=torch.tensor(
            [[[0.9, 0.05, 0.05]]], dtype=dtype
        ),
        previous_age=torch.zeros(1, 1, dtype=dtype),
        previous_strength=torch.ones(1, 1, dtype=dtype),
        previous_presence=torch.full((1, 1), 0.9, dtype=dtype),
        has_prior=torch.ones(1, 1, dtype=torch.bool),
        valid_feature_pair=torch.ones(1, 1, dtype=torch.bool),
        valid_geometry=torch.ones(1, 1, dtype=torch.bool),
        observability=torch.ones(1, 1, dtype=dtype),
        source_quality=torch.ones(1, 1, dtype=dtype),
        reliable_observation=torch.ones(1, 1, dtype=torch.bool),
        camera_coverage=torch.ones(1, 1, dtype=dtype),
    )


def _compute(**overrides):
    inputs = _inputs(overrides.pop("dtype", torch.float32))
    inputs.update(overrides)
    return ReliabilityCalibratedInnovation(mode="track")(**inputs)


def test_source_innovation_identical_changed_no_history_and_no_source():
    same = _compute()
    assert same["source_innovation"].item() == pytest.approx(0.0, abs=1e-6)

    changed = _compute(
        current_source=torch.tensor([[[0.0, 1.0]]])
    )
    assert changed["source_innovation"].item() > 0.9

    no_history = _compute(previous_source=torch.zeros(1, 1, 2))
    assert no_history["source_innovation"].item() == 1.0

    no_source = _compute(current_source=torch.zeros(1, 1, 2))
    assert no_source["source_innovation"].item() == 0.0
    assert torch.isfinite(no_source["source_innovation"]).all()


def test_feature_innovation_and_detach_contract():
    same = _compute()
    orthogonal_feature = torch.tensor(
        [[[0.0, 1.0]]], requires_grad=True
    )
    changed = _compute(current_feature=orthogonal_feature)
    assert same["feature_innovation"].item() == pytest.approx(0.0, abs=1e-6)
    assert changed["feature_innovation"].item() == pytest.approx(0.5)
    assert not changed["feature_innovation"].requires_grad


def test_feature_query_misalignment_fails_explicitly():
    inputs = _inputs()
    inputs["previous_feature"] = torch.zeros(1, 2, 2)
    with pytest.raises(ValueError, match="query-aligned"):
        ReliabilityCalibratedInnovation(mode="track")(**inputs)


def test_invalid_feature_pair_is_excluded_not_cross_matched():
    result = _compute(
        current_feature=torch.tensor([[[0.0, 1.0]]]),
        valid_feature_pair=torch.zeros(1, 1, dtype=torch.bool),
    )
    assert result["feature_innovation"].item() == 0.0


def test_geometry_innovation_jump_velocity_and_yaw_wrap():
    same = _compute()
    assert same["geometry_innovation"].item() == 0.0

    moving = _inputs()
    moving["current_geometry"][..., :3] = torch.tensor([0.2, 0.0, 0.0])
    moving["current_geometry"][..., 4:6] = torch.tensor([5.0, 0.0])
    moving["previous_geometry"][..., 4:6] = torch.tensor([5.0, 0.0])
    normal = ReliabilityCalibratedInnovation(mode="track")(**moving)
    assert normal["geometry_conflict"].item() < 0.1

    jump = _inputs()
    jump["current_geometry"][..., 0] = 30.0
    anomalous = ReliabilityCalibratedInnovation(mode="track")(**jump)
    assert anomalous["geometry_innovation"].item() > 0.9

    left = torch.tensor([math.pi - 0.01])
    right = torch.tensor([-math.pi + 0.01])
    assert wrapped_angle_difference(left, right).item() == pytest.approx(
        0.02, abs=1e-5
    )


def test_geometry_reference_ego_motion_round_trip():
    ledger = EvidenceLedger(
        memory_len=1, feature_dim=2, class_dim=2
    )
    ledger.pre_update(torch.zeros(1))
    ledger.reference_geometry[0, 0] = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    )
    ledger.reference_valid[0, 0] = True
    angle = math.pi / 2
    transform = torch.tensor(
        [[
            [math.cos(angle), -math.sin(angle), 0.0, 2.0],
            [math.sin(angle), math.cos(angle), 0.0, 3.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]]
    )
    ledger.transform_reference_geometry(transform)
    assert torch.allclose(
        ledger.reference_geometry[0, 0, :2],
        torch.tensor([2.0, 4.0]),
        atol=1e-5,
    )
    assert ledger.reference_geometry[0, 0, 3].item() == pytest.approx(
        angle, abs=1e-5
    )
    ledger.transform_reference_geometry(torch.linalg.inv(transform))
    assert torch.allclose(
        ledger.reference_geometry[0, 0],
        torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        atol=1e-5,
    )


def test_semantic_js_identity_difference_normalization_and_finiteness():
    same = torch.tensor([[[0.9, 0.1]]])
    opposite = torch.tensor([[[0.1, 0.9]]])
    assert normalized_js_divergence(same, same).item() == pytest.approx(
        0.0, abs=1e-6
    )
    assert normalized_js_divergence(same, opposite).item() > 0.5
    unnormalized = normalized_js_divergence(same * 10.0, opposite * 3.0)
    assert torch.isfinite(unnormalized).all()
    assert 0.0 <= unnormalized.item() <= 1.0


def test_temporal_reacquisition_monotonic_and_recovers_next_frame():
    zero = _compute(previous_age=torch.zeros(1, 1))
    one = _compute(previous_age=torch.ones(1, 1))
    five = _compute(previous_age=torch.full((1, 1), 5.0))
    assert zero["temporal_reacquisition"].item() == 0.0
    assert (
        zero["temporal_reacquisition"].item()
        < one["temporal_reacquisition"].item()
        < five["temporal_reacquisition"].item()
    )
    assert five["is_reacquired_query"].item()
    assert zero["is_continuous_observation"].item()


def test_reliability_high_then_falls_with_entropy_quality_or_observation():
    high = _compute()
    uncertain = _compute(
        current_class_probability=torch.tensor([[[0.5, 0.5]]]),
        current_ternary_probability=torch.tensor(
            [[[1.0 / 3, 1.0 / 3, 1.0 / 3]]]
        ),
    )
    low_quality = _compute(source_quality=torch.full((1, 1), 0.1))
    absent_observation = _compute(
        reliable_observation=torch.zeros(1, 1, dtype=torch.bool)
    )
    assert high["combined_reliability"] > uncertain["combined_reliability"]
    assert high["combined_reliability"] > low_quality["combined_reliability"]
    assert absent_observation["combined_reliability"].item() == 0.0


def test_disabling_extra_reliability_keeps_observability_evidence_factor():
    inputs = _inputs()
    inputs["observability"] = torch.full((1, 1), 0.4)
    result = ReliabilityCalibratedInnovation(
        mode="track", enable_reliability=False
    )(**inputs)
    assert result["positive_reliability"].item() == pytest.approx(
        0.4 * 0.95, abs=1e-6
    )


def test_conflict_present_absent_and_low_reliability_noise():
    compatible = _compute()
    conflict = _compute(
        current_ternary_probability=torch.tensor([[[0.01, 0.98, 0.01]]])
    )
    noisy = _compute(
        current_ternary_probability=torch.tensor([[[0.01, 0.98, 0.01]]]),
        source_quality=torch.zeros(1, 1),
    )
    assert conflict["conflict"] > compatible["conflict"]
    assert noisy["conflict"].item() == 0.0


def test_positive_negative_asymmetry_blocks_unobserved_and_camera_crash():
    unobserved = _compute(
        current_ternary_probability=torch.tensor([[[0.0, 0.0, 1.0]]]),
        reliable_observation=torch.zeros(1, 1, dtype=torch.bool),
    )
    crash = _compute(
        source_quality=torch.zeros(1, 1),
        camera_coverage=torch.zeros(1, 1),
    )
    visible_absence = _compute(
        current_ternary_probability=torch.tensor([[[0.01, 0.98, 0.01]]])
    )
    assert unobserved["negative_reliability"].item() == 0.0
    assert crash["negative_reliability"].item() == 0.0
    assert visible_absence["negative_reliability"].item() > 0.0


def _ledger(mode):
    return EvidenceLedger(
        memory_len=2,
        num_cameras=2,
        feature_dim=2,
        class_dim=2,
        temporal_update=EvidenceConservingTemporalUpdate(
            gamma=0.9,
            evidence_scale=2.0,
            enable_conservation=True,
        ),
        enable_source_ledger=True,
        innovation_cfg=dict(mode=mode),
    )


def _ledger_update(ledger, obs=1.0):
    ledger.pre_update(torch.zeros(1), scene_tokens=["scene-a"])
    return ledger.update_queries(
        torch.tensor([[[0.9, 0.05, 0.05]]]),
        torch.tensor([[obs]]),
        torch.tensor([[[1.0, 0.0]]]),
        torch.ones(1, 1),
        torch.ones(1, 1),
        0,
        1,
        raw_source_vector=torch.tensor([[[1.0, 0.0]]]),
        current_feature=torch.tensor([[[1.0, 0.0]]]),
        current_geometry=torch.zeros(1, 1, 6),
        current_class_probability=torch.tensor([[[0.9, 0.1]]]),
        source_quality=torch.ones(1, 1),
        camera_coverage=torch.ones(1, 1),
    )


def test_off_and_track_are_tensor_exact_for_s22_outputs():
    off = _ledger_update(_ledger("off"))
    track = _ledger_update(_ledger("track"))
    for key in (
        "alpha",
        "beta",
        "action",
        "score_scale",
        "write_mask",
        "novelty",
        "actual_added_positive_evidence",
        "actual_added_negative_evidence",
    ):
        assert torch.equal(off[key], track[key])


def test_active_changes_only_actual_evidence_path_and_conserves():
    track = _ledger_update(_ledger("track"))
    active = _ledger_update(_ledger("active"))
    assert not torch.equal(track["alpha"], active["alpha"])
    assert torch.equal(track["novelty"], active["novelty"])
    assert active["conservation_residual"].abs().max() < 1e-5
    assert not torch.any(active["unsupported_growth"])


def test_topk_reorders_all_s23_reference_state_together():
    ledger = _ledger("track")
    ledger.pre_update(torch.zeros(1))
    query_count = 3
    state = ledger.update_queries(
        torch.tensor([[[0.9, 0.05, 0.05]]]).expand(1, query_count, 3),
        torch.ones(1, query_count),
        torch.tensor([[[1.0, 0.0]]]).expand(1, query_count, 2),
        torch.ones(1, query_count),
        torch.ones(1, query_count),
        query_count,
        0,
        current_feature=torch.tensor(
            [[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]]
        ),
        current_geometry=torch.arange(18.0).reshape(1, 3, 6),
        current_class_probability=torch.tensor(
            [[[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]]
        ),
    )
    indexes = torch.tensor([[[2], [0]]])
    ledger.commit_topk(state, indexes, torch.ones(1, 3, dtype=torch.bool))
    assert torch.equal(
        ledger.reference_feature[:, :2],
        state["reference_feature"][:, [2, 0]],
    )
    assert torch.equal(
        ledger.reference_geometry[:, :2],
        state["reference_geometry"][:, [2, 0]],
    )
    assert torch.equal(
        ledger.reference_ternary_distribution[:, :2],
        state["reference_ternary_distribution"][:, [2, 0]],
    )


def test_defer_does_not_commit_feature_reference():
    ledger = _ledger("track")
    state = _ledger_update(ledger, obs=0.0)
    ledger.commit_topk(
        state, torch.tensor([[[0]]]), state["write_mask"]
    )
    assert not ledger.reference_valid[0, 0]
    assert torch.count_nonzero(ledger.reference_feature[0, 0]) == 0


def test_runtime_round_trip_and_checkpoints_cover_new_references():
    ledger = _ledger("track")
    state = _ledger_update(ledger)
    ledger.commit_topk(state, torch.tensor([[[0]]]), state["write_mask"])
    restored = _ledger("track")
    restored.load_runtime_state(ledger.export_runtime_state())
    for name in ledger._STATE_NAMES:
        assert torch.equal(getattr(ledger, name), getattr(restored, name))

    model = nn.Module()
    model.add_module("ledger", ledger)
    keys = tuple(model.state_dict())
    assert not any(name in key for key in keys for name in ledger._STATE_NAMES)
    payload = io.BytesIO()
    torch.save({"state_dict": model.state_dict()}, payload)
    payload.seek(0)
    assert not any(
        name in key
        for key in torch.load(payload)["state_dict"]
        for name in ledger._STATE_NAMES
    )


def test_scene_and_batch_changes_clear_s23_references():
    ledger = _ledger("track")
    state = _ledger_update(ledger)
    ledger.commit_topk(state, torch.tensor([[[0]]]), state["write_mask"])
    ledger.pre_update(torch.ones(1), scene_tokens=["scene-b"])
    assert not torch.any(ledger.reference_valid)
    ledger.pre_update(
        torch.ones(2), scene_tokens=["scene-b", "scene-c"]
    )
    assert ledger.reference_feature.shape == (2, 2, 2)
    assert not torch.any(ledger.reference_valid)


def test_cpu_fp16_track_is_finite():
    inputs = _inputs(torch.float16)
    result = ReliabilityCalibratedInnovation(mode="track")(**inputs)
    for value in result.values():
        if torch.is_tensor(value) and value.is_floating_point():
            assert torch.isfinite(value).all()
