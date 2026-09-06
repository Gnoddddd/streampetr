import numpy as np
import pandas as pd
import pytest
import torch

from analysis.care3d_p1 import (
    QUERY_COLLISION_POLICY,
    SOURCE_CAMERA_INDICES,
    SOURCE_NAMES,
    assert_exact_sample_alignment,
    assert_source_contract,
    assert_unique_queries,
    build_source_bank,
    filter_aligned_rows,
    p1_gate_flags,
    query_collision_eligibility,
    sample_projected_camera_tokens,
)
from models.care3d_p1 import CARE3DP1ScoreRouter, p1_score_routing_loss


def _router():
    return CARE3DP1ScoreRouter(
        object_dim=256,
        source_dim=256,
        vulnerability_dim=3,
        hidden_dim=64,
        top_k=2,
    )


def test_p1_zero_init_fault_path_is_exact_identity():
    model = _router()
    query = torch.randn(4, 256)
    sources = torch.randn(4, 3, 256)
    reliability = torch.ones(4, 3)
    vulnerability = torch.rand(4, 3)
    logits = torch.randn(4, 3)
    protocol = torch.tensor([0, 1, 2, 0])
    routed, aux = model(
        query, sources, reliability, vulnerability, logits, protocol,
        fault_active=True,
    )
    assert torch.equal(routed, query)
    assert aux["topk_indices"].shape == (4, 2)


def test_p1_clean_bypass_stays_exact_after_router_changes():
    model = _router()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.5, 0.5)
    query = torch.randn(3, 256)
    sources = torch.randn(3, 3, 256)
    reliability = torch.ones(3, 3)
    vulnerability = torch.rand(3, 3)
    logits = torch.randn(3, 3)
    protocol = torch.tensor([0, 1, 2])
    routed, aux = model(
        query, sources, reliability, vulnerability, logits, protocol,
        fault_active=False,
    )
    assert torch.equal(routed, query)
    assert aux["clean_bypass"].all()


def test_source_bank_excludes_failed_cam_back_and_appends_temporal():
    assert SOURCE_NAMES == ("CAM_BACK_LEFT", "CAM_BACK_RIGHT", "TEMPORAL_ANCHOR")
    assert 3 not in SOURCE_CAMERA_INDICES
    camera = torch.randn(2, 2, 256)
    reliability = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    temporal = torch.randn(2, 256)
    sources, source_reliability = build_source_bank(camera, reliability, temporal)
    assert sources.shape == (2, 3, 256)
    assert torch.equal(sources[:, 2], temporal)
    assert torch.equal(source_reliability[:, 2], torch.ones(2))


def test_projected_camera_sampling_uses_predicted_center_and_visibility():
    p0 = torch.zeros(6, 1, 3, 3)
    p0[4, 0] = torch.arange(9, dtype=torch.float32).view(3, 3)
    p0[5, 0] = 10 + torch.arange(9, dtype=torch.float32).view(3, 3)
    matrices = torch.eye(4).repeat(6, 1, 1)
    centers = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, -1.0]])
    tokens, reliability = sample_projected_camera_tokens(
        p0, matrices, centers, (3, 3)
    )
    assert tokens.shape == (2, 2, 1)
    assert reliability.shape == (2, 2)
    assert torch.allclose(tokens[0, :, 0], torch.tensor([4.0, 14.0]))
    assert reliability[0].eq(1).all()
    assert reliability[1].eq(0).all()
    assert tokens[1].eq(0).all()


def test_projected_sampling_rejects_failed_camera_source():
    p0 = torch.zeros(6, 1, 2, 2)
    matrices = torch.eye(4).repeat(6, 1, 1)
    centers = torch.tensor([[0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        sample_projected_camera_tokens(
            p0, matrices, centers, (2, 2), camera_indices=(3,)
        )


def test_source_contract_requires_temporal_reliability():
    sources = np.zeros((2, 3, 3, 256), dtype=np.float32)
    reliability = np.ones((2, 3, 3), dtype=np.float32)
    assert_source_contract(sources, reliability)
    reliability[0, 0, 2] = 0
    with pytest.raises(RuntimeError):
        assert_source_contract(sources, reliability)


def test_p1_score_loss_is_finite_with_missing_positive_or_negative_minibatch():
    batch, classes = 4, 10
    query = torch.randn(batch, 256, requires_grad=True)
    clean = torch.randn(batch, 256)
    fault = torch.randn(batch, 256)
    routed_logits = torch.randn(batch, classes, requires_grad=True)
    fault_logits = torch.randn(batch, classes)
    target_class = torch.tensor([0, 1, 2, 3])
    clean_score = torch.rand(batch)
    fault_score = torch.rand(batch)
    threshold = torch.full((batch,), 0.2)
    for label in (torch.ones(batch), torch.zeros(batch)):
        losses = p1_score_routing_loss(
            routed_query=query,
            clean_query=clean,
            fault_query=fault,
            routed_logits=routed_logits,
            fault_logits=fault_logits,
            target_class=target_class,
            clean_score=clean_score,
            fault_score=fault_score,
            fault_topk_threshold=threshold,
            cross_topk=label,
        )
        assert torch.isfinite(losses["total"])


def test_p1_alignment_and_unique_query_guards():
    assert_exact_sample_alignment(["a", "b"], ["a", "b"])
    with pytest.raises(RuntimeError):
        assert_exact_sample_alignment(["a", "b"], ["b", "a"])
    assert_unique_queries([1, 3, 9])
    with pytest.raises(RuntimeError):
        assert_unique_queries([1, 3, 1])


def test_query_collision_eligibility_excludes_entire_collision_group():
    audit = query_collision_eligibility(
        [3, 3, 3, 4, 4, 5],
        [7, 7, 9, 7, 8, 7],
    )
    assert audit["policy"] == QUERY_COLLISION_POLICY
    assert audit["eligible"].tolist() == [False, False, True, True, True, True]
    assert audit["multiplicity"].tolist() == [2, 2, 1, 1, 1, 1]
    assert audit["p0_rows_total"] == 6
    assert audit["p1_eligible_rows"] == 4
    assert audit["query_collision_excluded_rows"] == 2
    assert audit["query_collision_groups"] == 1


def test_filter_aligned_rows_uses_metadata_only_and_preserves_order():
    frame = pd.DataFrame({
        "sample_id": ["a", "b", "c", "d"],
        "target_frame_idx": [3, 3, 3, 4],
        "target_clean_query_index": [11, 11, 12, 11],
        "cross_topk_like_metadata_not_used": [1, 0, 1, 0],
    })
    arrays = {
        "object_features": np.arange(4 * 2, dtype=np.float32).reshape(4, 2),
        "labels": np.asarray([1, 0, 1, 0], dtype=np.int8),
    }
    filtered, packed, audit = filter_aligned_rows(frame, arrays)
    assert filtered.sample_id.tolist() == ["c", "d"]
    assert packed["labels"].tolist() == [1, 0]
    assert packed["object_features"].tolist() == [[4.0, 5.0], [6.0, 7.0]]
    assert int(packed["_p1_query_collision_excluded_rows"]) == 2
    assert audit["query_collision_groups"] == 1
    for _, group in filtered.groupby("target_frame_idx"):
        assert len(group.target_clean_query_index) == len(set(group.target_clean_query_index))


def test_p1_gate_requires_recovery_no_harm_fp_and_clean_identity():
    point = {
        "lost_recovery_rate": 0.2,
        "net_tp_delta": 0.01,
        "cross_topk_recovery_rate": 0.3,
        "target_score_delta_on_cross": 0.04,
        "retained_damage_rate": 0.001,
        "fp_inflation_rate": 0.002,
        "clean_identity_pass": True,
    }
    ci = {
        "lost_recovery_ci_low": 0.05,
        "net_tp_delta_ci_low": 0.001,
        "cross_topk_recovery_ci_low": 0.1,
        "target_score_delta_ci_low": 0.01,
        "retained_damage_ci_high": 0.004,
    }
    fp = {"ci_high": 0.008}
    flags = p1_gate_flags(
        point, ci, ci, fp,
        max_retained_damage_rate=0.005,
        max_retained_damage_ci_high=0.01,
        max_fp_inflation_rate=0.01,
        max_fp_inflation_ci_high=0.02,
    )
    assert flags["seed_protocol_pass"] is True
    failed = dict(point)
    failed["fp_inflation_rate"] = 0.02
    flags = p1_gate_flags(
        failed, ci, ci, fp,
        max_retained_damage_rate=0.005,
        max_retained_damage_ci_high=0.01,
        max_fp_inflation_rate=0.01,
        max_fp_inflation_ci_high=0.02,
    )
    assert flags["fp_control_pass"] is False
    assert flags["seed_protocol_pass"] is False
