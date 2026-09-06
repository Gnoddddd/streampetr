from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from analysis.care3d_counterfactual import (
    assert_disjoint_splits,
    assert_prospective_payload,
    assert_unique_sample_ids,
    clone_counterfactual_states,
    freeze_module,
    same_query_counterfactual,
    states_exact,
)
from models.care3d import CARE3DCore, CounterfactualVulnerabilityLoss


def clean_state():
    return {
        "memory_embedding": torch.arange(12, dtype=torch.float32).reshape(1, 3, 4),
        "memory_reference_point": torch.ones(1, 3, 3),
        "memory_timestamp": torch.zeros(1, 3, 1),
        "memory_egopose": torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 3, 1, 1),
        "memory_velo": torch.zeros(1, 3, 2),
    }


def test_four_counterfactual_branches_start_from_identical_clean_state():
    base = clean_state()
    branches = clone_counterfactual_states(base, 4)
    assert len(branches) == 4
    assert all(states_exact(base, branch) for branch in branches)
    assert len({id(branch["memory_embedding"]) for branch in branches}) == 4


def test_fault_branch_cannot_pollute_clean_continuation_state():
    base = clean_state()
    clean_branch, blur, crash, dark = clone_counterfactual_states(base, 4)
    blur["memory_embedding"].add_(1000)
    crash["memory_timestamp"].add_(5)
    dark["memory_velo"].sub_(7)
    assert states_exact(base, clean_branch)
    assert not states_exact(base, blur)
    assert not states_exact(base, crash)
    assert not states_exact(base, dark)


def test_predictor_payload_rejects_future_or_fault_inputs():
    payload = {
        "object_features": np.zeros(256),
        "temporal_features": np.zeros(256),
        "decision_features": np.zeros(21),
        "camera_support": np.zeros(6),
        "camera_quality": np.ones(6),
    }
    assert_prospective_payload(payload)
    with pytest.raises(RuntimeError):
        assert_prospective_payload({**payload, "fault_score": 0.3})
    with pytest.raises(RuntimeError):
        assert_prospective_payload({**payload, "future_representation": np.zeros(4)})


def test_evidence_drop_uses_same_clean_matched_query_not_fault_best_query():
    clean = torch.full((3, 2), -6.0)
    fault = torch.full((3, 2), -6.0)
    clean[1, 0] = 4.0  # clean-selected q=1,c=0
    fault[1, 0] = -2.0  # same query collapses
    fault[2, 0] = 8.0  # a different query becomes strongest under the fault
    result = same_query_counterfactual(clean, fault, query=1, label=0, k=2)
    assert result["evidence_drop"] > 0.7
    assert result["clean_score"] > result["fault_score"]
    assert result["fault_flat_rank"] > 1  # proves the replacement query was not substituted


def test_invalid_protocol_mask_is_respected_by_existing_loss():
    criterion = CounterfactualVulnerabilityLoss()
    prediction = {
        "vulnerability": torch.tensor([[[0.2, 0.4, 100.0]]]),
        "boundary_crossing_logits": torch.tensor([[[0.0, 0.0, 100.0]]]),
    }
    drop = torch.tensor([[[0.1, 0.3, 0.0]]])
    crossing = torch.tensor([[[0.0, 1.0, 0.0]]])
    mask = torch.tensor([[[1, 1, 0]]], dtype=torch.bool)
    losses = criterion(prediction, drop, crossing, mask)
    assert torch.isfinite(losses["loss_care3d"])
    prediction2 = {key: value.clone() for key, value in prediction.items()}
    prediction2["vulnerability"][0, 0, 2] = 1e6
    prediction2["boundary_crossing_logits"][0, 0, 2] = -1e6
    losses2 = criterion(prediction2, drop, crossing, mask)
    assert torch.equal(losses["loss_care3d"], losses2["loss_care3d"])


def test_detector_freeze_sets_every_parameter_requires_grad_false():
    detector = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    freeze_module(detector)
    assert not detector.training
    assert all(not parameter.requires_grad for parameter in detector.parameters())


def test_care_p0_disabled_routing_is_exact_detector_identity():
    core = CARE3DCore(
        object_dim=8, num_cameras=6, num_protocols=3,
        hidden_dim=16, state_dim=8, decision_dim=2,
        enable_routing=False,
    )
    object_features = torch.randn(2, 5, 8)
    output = core(
        object_features=object_features,
        camera_support=torch.ones(2, 5, 6),
        camera_quality=torch.ones(2, 6),
        temporal_features=torch.randn(2, 5, 8),
        decision_features=torch.randn(2, 5, 2),
    )
    assert torch.equal(output["enhanced_features"], object_features)
    assert core.router is None


def test_scene_split_rejects_cross_split_leakage():
    assert_disjoint_splits([
        {"scene_token": "a", "split": "probe_train"},
        {"scene_token": "b", "split": "probe_val"},
        {"scene_token": "c", "split": "probe_test"},
    ])
    with pytest.raises(RuntimeError):
        assert_disjoint_splits([
            {"scene_token": "a", "split": "probe_train"},
            {"scene_token": "a", "split": "probe_test"},
        ])


def test_resume_sample_ids_must_be_unique():
    assert_unique_sample_ids(["scene:a:2:3", "scene:b:3:4"])
    with pytest.raises(RuntimeError):
        assert_unique_sample_ids(["scene:a:2:3", "scene:a:2:3"])
