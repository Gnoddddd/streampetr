"""Contracts for R2 memory isolation and confirmed reacquisition."""

from __future__ import annotations

import copy

import pytest
import torch

from models.evidence_ledger import EvidenceLedger
from models.pending_reacquisition import PendingReacquisitionTracker
from models.temporal_update import EvidenceConservingTemporalUpdate


def _tracker(confirm=True, dtype=torch.float32, max_age=3):
    return PendingReacquisitionTracker(
        capacity=4,
        num_sources=2,
        confirmation_frames=2,
        pending_max_age=max_age,
        class_consistency_required=True,
        center_distance_threshold=2.0,
        motion_distance_threshold=2.0,
        minimum_confirmation_score=0.075,
        minimum_confirmation_reliability=0.65,
        enable_confirmation=confirm,
    )


def _step(
    tracker,
    *,
    seed=(True, False),
    classes=(1, 2),
    centers=((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)),
    velocity=((1.0, 0.0), (0.0, 0.0)),
    scores=(0.5, 0.5),
    reliability=(0.8, 0.8),
    proposed_bonus=0.02,
    timestamp=0.0,
    scene=("scene-a",),
    reset=False,
    dtype=torch.float32,
    device=None,
):
    queries = len(seed)
    source = torch.ones(1, queries, 2, dtype=dtype, device=device)
    center_tensor = (
        torch.tensor([centers], dtype=dtype, device=device)
        if queries
        else torch.zeros(1, 0, 3, dtype=dtype, device=device)
    )
    velocity_tensor = (
        torch.tensor([velocity], dtype=dtype, device=device)
        if queries
        else torch.zeros(1, 0, 2, dtype=dtype, device=device)
    )
    return tracker.step(
        scene_tokens=scene,
        seed_mask=torch.tensor([seed], dtype=torch.bool, device=device),
        predicted_class=torch.tensor([classes], dtype=torch.long, device=device),
        center=center_tensor,
        velocity=velocity_tensor,
        score=torch.tensor([scores], dtype=dtype, device=device),
        reliability=torch.tensor([reliability], dtype=dtype, device=device),
        proposed_bonus=torch.full(
            (1, queries), proposed_bonus, dtype=dtype, device=device
        ),
        prior_alpha=torch.full(
            (1, queries), 2.0, dtype=dtype, device=device
        ),
        prior_beta=torch.full(
            (1, queries), 1.5, dtype=dtype, device=device
        ),
        prior_source_evidence=source,
        timestamp=torch.tensor(
            [timestamp], dtype=torch.float64, device=device
        ),
        query_source=torch.tensor(
            [[1] * queries], dtype=torch.long, device=device
        ),
        reset_mask=torch.tensor([reset], device=device),
    )


def test_single_frame_candidate_is_pending_not_confirmed():
    state = _step(_tracker())
    assert state["candidate_created"][0, 0]
    assert state["pending_mask"][0, 0]
    assert not state["confirmation_ready_mask"].any()


def test_second_consistent_frame_confirms_after_query_reorder():
    tracker = _tracker()
    first = _step(tracker)
    runtime_id = first["runtime_id"][0, 0].item()
    second = _step(
        tracker,
        seed=(False, False),
        classes=(2, 1),
        centers=((10.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
        timestamp=0.5,
    )
    assert second["confirmation_ready_mask"][0, 1]
    assert second["runtime_id"][0, 1].item() == runtime_id
    confirmed = tracker.finalize(
        second["confirmation_ready_mask"], second["slot_for_query"]
    )
    assert confirmed[0, 1].item() == runtime_id
    assert not tracker.active.any()


def test_consistent_candidate_without_positive_budget_is_rejected_not_confirmed():
    tracker = _tracker()
    _step(tracker, proposed_bonus=0.0)
    second = _step(
        tracker,
        seed=(False, False),
        centers=((0.5, 0.0, 0.0), (10.0, 0.0, 0.0)),
        timestamp=0.5,
    )
    assert not second["confirmation_ready_mask"].any()
    assert second["rejected_query_mask"][0, 0]
    assert second["rejected_count"].item() == 1
    assert not tracker.active.any()


@pytest.mark.parametrize(
    "change",
    ("class", "center", "motion", "score", "reliability"),
)
def test_inconsistent_or_low_quality_candidate_is_rejected(change):
    tracker = _tracker()
    _step(tracker)
    kwargs = dict(
        seed=(False, False),
        classes=(1, 2),
        centers=((0.5, 0.0, 0.0), (10.0, 0.0, 0.0)),
        velocity=((1.0, 0.0), (0.0, 0.0)),
        scores=(0.5, 0.5),
        reliability=(0.8, 0.8),
        timestamp=0.5,
    )
    if change == "class":
        kwargs["classes"] = (3, 2)
    elif change == "center":
        kwargs["centers"] = ((8.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    elif change == "motion":
        kwargs["centers"] = ((-3.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    elif change == "score":
        kwargs["scores"] = (0.01, 0.5)
    else:
        kwargs["reliability"] = (0.1, 0.8)
    state = _step(tracker, **kwargs)
    assert not state["confirmation_ready_mask"].any()
    assert state["rejected_count"].item() == 1
    assert not tracker.active.any()


def test_r2a_never_confirms_but_keeps_stable_pending_identity():
    tracker = _tracker(confirm=False)
    _step(tracker)
    second = _step(
        tracker,
        seed=(False, False),
        centers=((0.5, 0.0, 0.0), (10.0, 0.0, 0.0)),
        timestamp=0.5,
    )
    assert second["pending_mask"][0, 0]
    assert not second["confirmation_ready_mask"].any()
    assert tracker.active.any()


def test_empty_queries_expire_pending_at_fixed_age():
    tracker = _tracker(max_age=2)
    _step(tracker)
    for timestamp in (0.5, 1.0):
        state = _step(
            tracker,
            seed=(),
            classes=(),
            centers=(),
            velocity=(),
            scores=(),
            reliability=(),
            timestamp=timestamp,
        )
        assert state["expired_count"].item() == 0
    state = _step(
        tracker,
        seed=(),
        classes=(),
        centers=(),
        velocity=(),
        scores=(),
        reliability=(),
        timestamp=1.5,
    )
    assert state["expired_count"].item() == 1
    assert not tracker.active.any()


def test_scene_and_explicit_reset_clear_pending():
    tracker = _tracker()
    _step(tracker)
    state = _step(tracker, seed=(False, False), scene=("scene-b",))
    assert not state["confirmation_ready_mask"].any()
    assert not tracker.active.any()
    _step(tracker)
    state = _step(tracker, seed=(False, False), reset=True)
    assert not state["confirmation_ready_mask"].any()
    assert not tracker.active.any()


def test_batch_rows_are_independent():
    tracker = _tracker()
    queries = 1
    tracker.step(
        scene_tokens=("a", "b"),
        seed_mask=torch.tensor([[True], [False]]),
        predicted_class=torch.tensor([[1], [1]]),
        center=torch.zeros(2, queries, 3),
        velocity=torch.zeros(2, queries, 2),
        score=torch.ones(2, queries),
        reliability=torch.ones(2, queries),
        proposed_bonus=torch.ones(2, queries),
        prior_alpha=torch.ones(2, queries),
        prior_beta=torch.ones(2, queries),
        prior_source_evidence=torch.zeros(2, queries, 2),
        timestamp=torch.zeros(2, dtype=torch.float64),
        query_source=torch.zeros(2, queries, dtype=torch.long),
    )
    assert tracker.active[0].any()
    assert not tracker.active[1].any()


def test_query_count_change_and_invalid_velocity_are_safe():
    tracker = _tracker(dtype=torch.float16)
    _step(
        tracker,
        velocity=((float("nan"), float("inf")), (0.0, 0.0)),
        dtype=torch.float16,
    )
    state = _step(
        tracker,
        seed=(False, False, False),
        classes=(1, 2, 3),
        centers=((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0)),
        velocity=((0.0, 0.0),) * 3,
        scores=(0.5,) * 3,
        reliability=(0.8,) * 3,
        timestamp=0.5,
        dtype=torch.float16,
    )
    assert state["pending_mask"].shape == (1, 3)
    assert torch.isfinite(tracker.velocity).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gpu_fp16_pending_confirmation_is_finite_and_device_safe():
    tracker = _tracker(dtype=torch.float16).cuda().half()
    first = _step(tracker, dtype=torch.float16, device="cuda")
    assert first["pending_mask"].is_cuda
    second = _step(
        tracker,
        seed=(False, False),
        centers=((0.5, 0.0, 0.0), (10.0, 0.0, 0.0)),
        timestamp=0.5,
        dtype=torch.float16,
        device="cuda",
    )
    assert second["confirmation_ready_mask"][0, 0]
    for name in ("center", "velocity", "proposed_bonus"):
        value = getattr(tracker, name)
        assert value.is_cuda
        assert value.dtype == torch.float16
        assert torch.isfinite(value).all()


def test_pending_runtime_is_nonpersistent_and_explicitly_exportable():
    tracker = _tracker()
    _step(tracker)
    assert not tracker.state_dict()
    exported = tracker.export_runtime_state()
    restored = _tracker()
    restored.load_runtime_state(exported)
    torch.testing.assert_close(restored.center, tracker.center)
    assert restored._scene_tokens == tracker._scene_tokens


def _ledger():
    ledger = EvidenceLedger(
        memory_len=4,
        num_cameras=2,
        temporal_update=EvidenceConservingTemporalUpdate(
            enable_conservation=True,
            conservation_tolerance=1e-5,
        ),
        enable_source_ledger=True,
        feature_dim=2,
        class_dim=2,
        innovation_cfg=dict(
            mode="active",
            innovation_active_strategy="budgeted_reacquisition",
            restore_ratio=0.5,
            max_relative_bonus=0.08,
            max_absolute_bonus=0.05,
            minimum_gap_age=2,
            use_motion_gate=False,
            use_source_recovery_gate=False,
        ),
    )
    ledger.pre_update(torch.ones(1), ["scene-a"])
    return ledger


def _ledger_update(ledger, queries=2):
    source = torch.tensor([[[1.0, 0.0]] * queries])
    return ledger.update_queries(
        torch.tensor([[[0.8, 0.1, 0.1]] * queries]),
        torch.ones(1, queries),
        source,
        torch.ones(1, queries),
        torch.ones(1, queries),
        num_base_queries=0,
        num_propagated=queries,
        raw_source_vector=source,
        current_feature=torch.ones(1, queries, 2),
        current_geometry=torch.zeros(1, queries, 6),
        current_class_probability=torch.tensor([[[0.8, 0.2]] * queries]),
        source_quality=torch.ones(1, queries),
    )


def test_isolation_zeroes_formal_evidence_source_and_write_only_for_candidate():
    ledger = _ledger()
    state = _ledger_update(ledger)
    reference = copy.deepcopy(state)
    isolated = torch.tensor([[True, False]])
    controlled = ledger.apply_reacquisition_control(
        state,
        isolated,
        raw_source_vector=state["current_source_vector"],
    )
    assert controlled["actual_added_positive_evidence"][0, 0].item() == 0.0
    assert controlled["actual_added_negative_evidence"][0, 0].item() == 0.0
    assert controlled["current_source_increment"][0, 0].sum().item() == 0.0
    assert not controlled["write_mask"][0, 0]
    assert controlled["action"][0, 0].item() == 2
    for key in (
        "alpha",
        "beta",
        "source_evidence",
        "action",
        "write_mask",
        "conservation_residual",
        "source_mass_residual",
    ):
        torch.testing.assert_close(controlled[key][0, 1], reference[key][0, 1])
    assert controlled["conservation_residual"][0, 0].abs().item() < 1e-5
    assert not controlled["source_mass_violation"][0, 0]


def test_confirmation_adds_one_bonus_and_enables_formal_write():
    ledger = _ledger()
    state = _ledger_update(ledger, queries=1)
    bonus = torch.tensor([[0.02]])
    controlled = ledger.apply_reacquisition_control(
        state,
        torch.ones(1, 1, dtype=torch.bool),
        confirmed_mask=torch.ones(1, 1, dtype=torch.bool),
        confirmation_bonus=bonus,
        confirmation_prior_alpha=torch.tensor([[2.0]]),
        confirmation_prior_beta=torch.tensor([[1.5]]),
        confirmation_prior_source_evidence=torch.ones(1, 1, 2),
        raw_source_vector=state["current_source_vector"],
    )
    torch.testing.assert_close(
        controlled["actual_added_positive_evidence"],
        controlled["base_positive_evidence"] + bonus,
    )
    assert controlled["write_mask"].item()
    assert controlled["confirmed_reacquisition_mask"].item()
    assert controlled["restoration_bonus"].item() == pytest.approx(0.02)
    assert controlled["conservation_residual"].abs().item() < 1e-5
    assert not controlled["source_mass_violation"].item()
