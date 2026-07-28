"""Contracts for S2.3 evidence-budgeted reacquisition."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from models.evidence_ledger import EvidenceLedger
from models.innovation import ReliabilityCalibratedInnovation
from models.temporal_update import EvidenceConservingTemporalUpdate


def _ledger(
    strategy="budgeted_reacquisition",
    dtype=torch.float32,
    **innovation,
):
    cfg = dict(
        mode="active",
        innovation_active_strategy=strategy,
        restore_ratio=0.5,
        max_relative_bonus=0.08,
        max_absolute_bonus=0.05,
        minimum_gap_age=2,
        reacquisition_time_tau=3.0,
        use_motion_gate=False,
        use_source_recovery_gate=True,
    )
    cfg.update(innovation)
    ledger = EvidenceLedger(
        memory_len=4,
        num_cameras=2,
        temporal_update=EvidenceConservingTemporalUpdate(
            enable_conservation=True,
            conservation_tolerance=2e-5,
        ),
        enable_source_ledger=True,
        feature_dim=2,
        class_dim=2,
        innovation_cfg=cfg,
    )
    ledger.pre_update(torch.ones(1, dtype=dtype), scene_tokens=["scene-a"])
    return ledger


def _update(
    ledger,
    observability=1.0,
    source=(1.0, 0.0),
    geometry=(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    probabilities=(0.8, 0.1, 0.1),
    delta_time=1.0,
    num_queries=1,
):
    dtype = ledger.alpha.dtype
    ternary = torch.tensor([[probabilities] * num_queries], dtype=dtype)
    obs = torch.full((1, num_queries), observability, dtype=dtype)
    src = torch.tensor([[source] * num_queries], dtype=dtype)
    geom = torch.tensor([[geometry] * num_queries], dtype=dtype)
    return ledger.update_queries(
        ternary,
        obs,
        src,
        torch.ones_like(obs),
        torch.ones_like(obs),
        num_base_queries=0,
        num_propagated=num_queries,
        raw_source_vector=src,
        current_feature=torch.ones(1, num_queries, 2, dtype=dtype),
        current_geometry=geom,
        current_class_probability=torch.tensor(
            [[[0.8, 0.2]] * num_queries], dtype=dtype
        ),
        source_quality=(src.sum(-1) > 0).to(dtype),
        delta_time=torch.full_like(obs, delta_time),
    )


def _commit(ledger, state, valid=True, index=0):
    indexes = torch.tensor([[[index]]])
    mask = torch.full(
        state["write_mask"].shape, valid, dtype=torch.bool
    )
    ledger.commit_topk(state, indexes, valid_write_mask=mask)
    ledger.pre_update(torch.ones(1, dtype=ledger.alpha.dtype), ["scene-a"])


def _gap_sequence(ledger, recovery_geometry=None, recovery_source=(1.0, 0.0)):
    initial = _update(ledger)
    _commit(ledger, initial)
    gap1 = _update(ledger, observability=0.0, source=(0.0, 0.0))
    _commit(ledger, gap1)
    gap2 = _update(ledger, observability=0.0, source=(0.0, 0.0))
    _commit(ledger, gap2)
    recovery = _update(
        ledger,
        source=recovery_source,
        geometry=(
            recovery_geometry
            if recovery_geometry is not None
            else (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        ),
    )
    return initial, gap1, gap2, recovery


def test_disabled_strategy_matches_s22_tensorwise():
    reference = _ledger()
    reference.innovation.mode = "off"
    candidate = _ledger()
    state_a = _update(reference)
    state_b = _update(candidate)
    for key in ("alpha", "beta", "actual_added_positive_evidence",
                "actual_added_negative_evidence"):
        torch.testing.assert_close(state_a[key], state_b[key])


def test_base_evidence_is_exact_s22():
    ledger = _ledger()
    state = _update(ledger)
    torch.testing.assert_close(
        state["base_positive_evidence"],
        state["actual_added_positive_evidence"],
    )
    torch.testing.assert_close(
        state["base_negative_evidence"],
        state["actual_added_negative_evidence"],
    )


def test_continuous_clean_has_zero_bonus():
    ledger = _ledger()
    _commit(ledger, _update(ledger))
    assert _update(ledger)["restoration_bonus"].item() == 0.0


def test_new_query_has_zero_bonus():
    assert _update(_ledger())["restoration_bonus"].item() == 0.0


def test_unreliable_observation_has_zero_bonus():
    ledger = _ledger()
    _commit(ledger, _update(ledger))
    assert _update(ledger, observability=0.0)["restoration_bonus"].item() == 0.0


def test_gap_start_freezes_anchor():
    ledger = _ledger()
    initial = _update(ledger)
    _commit(ledger, initial)
    gap = _update(ledger, observability=0.0, source=(0.0, 0.0))
    assert gap["gap_active"].item()
    assert gap["gap_age"].item() == 1.0
    torch.testing.assert_close(
        gap["pre_gap_strength"], initial["strength"]
    )


def test_gap_anchor_does_not_decay():
    _, gap1, gap2, _ = _gap_sequence(_ledger())
    torch.testing.assert_close(
        gap1["pre_gap_strength"], gap2["pre_gap_strength"]
    )


def test_first_recovery_is_reacquired():
    *_, recovery = _gap_sequence(_ledger())
    assert recovery["is_reacquired"].item()
    assert recovery["reacquisition_consumed"].item()


def test_second_recovery_has_no_repeat_bonus():
    ledger = _ledger()
    *_, first = _gap_sequence(ledger)
    _commit(ledger, first)
    second = _update(ledger)
    assert second["restoration_bonus"].item() == 0.0
    assert not second["gap_active"].item()


def test_lost_strength_is_nonnegative():
    *_, recovery = _gap_sequence(_ledger())
    assert recovery["lost_strength"].min().item() >= 0.0


def test_bonus_respects_lost_budget():
    ledger = _ledger(restore_ratio=0.25, max_relative_bonus=10.0)
    *_, state = _gap_sequence(ledger)
    assert state["restoration_bonus"].item() <= (
        0.25 * state["lost_strength"].item() + 1e-6
    )


def test_bonus_respects_relative_cap():
    ledger = _ledger(restore_ratio=10.0, max_relative_bonus=0.05)
    *_, state = _gap_sequence(ledger)
    assert state["restoration_bonus"].item() <= (
        0.05 * state["base_positive_evidence"].item() + 1e-6
    )


def test_bonus_respects_absolute_cap():
    ledger = _ledger(
        restore_ratio=10.0,
        max_relative_bonus=10.0,
        max_absolute_bonus=0.001,
    )
    *_, state = _gap_sequence(ledger)
    assert state["restoration_bonus"].item() <= 0.001 + 1e-6


def test_motion_consistency_is_high_for_prediction():
    ledger = _ledger(use_motion_gate=True, use_source_recovery_gate=False)
    *_, state = _gap_sequence(
        ledger, recovery_geometry=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    )
    assert state["motion_consistency"].item() > 0.95


def test_motion_jump_reduces_gate():
    good = _ledger(use_motion_gate=True, use_source_recovery_gate=False)
    bad = _ledger(use_motion_gate=True, use_source_recovery_gate=False)
    *_, good_state = _gap_sequence(good)
    *_, bad_state = _gap_sequence(
        bad, recovery_geometry=(50.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    )
    assert (
        bad_state["motion_consistency"].item()
        < good_state["motion_consistency"].item()
    )


def test_constant_velocity_is_not_suppressed():
    ledger = _ledger(
        use_motion_gate=True,
        use_source_recovery_gate=False,
        motion_sigma=2.0,
    )
    *_, state = _gap_sequence(ledger)
    assert state["motion_consistency"].item() > 0.8


def test_source_recovery_raises_gate():
    ledger = _ledger(use_motion_gate=False)
    *_, state = _gap_sequence(ledger, recovery_source=(1.0, 0.0))
    assert state["source_recovery"].item() > 0.99


def test_missing_source_recovery_zeroes_gate():
    ledger = _ledger(use_motion_gate=False)
    *_, state = _gap_sequence(ledger, recovery_source=(0.0, 1.0))
    assert state["source_recovery"].item() == 0.0
    assert state["restoration_bonus"].item() == 0.0


def test_negative_evidence_is_exact_s22():
    *_, state = _gap_sequence(_ledger())
    torch.testing.assert_close(
        state["actual_added_negative_evidence"],
        state["base_negative_evidence"],
    )


def test_conservation_residual_is_within_tolerance():
    *_, state = _gap_sequence(_ledger())
    assert state["conservation_residual"].abs().max().item() < 2e-5


def test_unsupported_growth_is_zero():
    *_, state = _gap_sequence(_ledger())
    assert not state["unsupported_growth"].any()


def test_source_mass_conservation_is_preserved():
    *_, state = _gap_sequence(_ledger())
    assert state["source_mass_residual"].abs().max().item() < 2e-5


def test_topk_reordering_keeps_gap_state_aligned():
    ledger = _ledger()
    state = _update(ledger, num_queries=2)
    state["pre_gap_strength"][0] = torch.tensor([3.0, 7.0])
    state["gap_age"][0] = torch.tensor([2.0, 5.0])
    _commit(ledger, state, index=1)
    assert ledger.pre_gap_strength[0, 0].item() == 7.0
    assert ledger.gap_age[0, 0].item() == 5.0


def test_defer_invalid_write_clears_gap_anchor():
    ledger = _ledger()
    state = _update(ledger)
    state["gap_active"].fill_(True)
    state["pre_gap_strength"].fill_(9.0)
    _commit(ledger, state, valid=False)
    assert not ledger.gap_active[0, 0]
    assert ledger.pre_gap_strength[0, 0].item() == 0.0


@pytest.mark.parametrize("change", ["scene", "batch", "query"])
def test_scene_batch_and_query_changes_are_safe(change):
    ledger = _ledger()
    _commit(ledger, _update(ledger))
    if change == "scene":
        ledger.pre_update(torch.ones(1), ["scene-b"])
        assert not ledger.reference_valid.any()
    elif change == "batch":
        ledger.pre_update(torch.ones(2), ["a", "b"])
        assert ledger.alpha.shape[0] == 2
        assert not ledger.gap_active.any()
    else:
        state = _update(ledger, num_queries=3)
        assert state["gap_active"].shape == (1, 3)


def test_runtime_export_import_restores_gap_state():
    ledger = _ledger()
    _, gap1, _, _ = _gap_sequence(ledger)
    _commit(ledger, gap1)
    exported = ledger.export_runtime_state()
    restored = _ledger()
    restored.load_runtime_state(exported)
    torch.testing.assert_close(restored.gap_age, ledger.gap_age)
    torch.testing.assert_close(
        restored.pre_gap_strength, ledger.pre_gap_strength
    )


def test_state_dict_excludes_gap_runtime():
    ledger = _ledger()
    keys = tuple(ledger.state_dict())
    for name in EvidenceLedger._STATE_NAMES:
        assert not any(key.endswith(name) for key in keys)


def test_checkpoint_hook_strips_gap_runtime():
    ledger = _ledger()
    destination = {
        "head.evidence_ledger." + name: torch.ones(1)
        for name in EvidenceLedger._STATE_NAMES
    }
    ledger._strip_runtime_checkpoint_state(
        ledger, destination, "head.evidence_ledger.", {}
    )
    assert not destination


def test_cpu_fp16_budget_path_is_safe():
    ledger = _ledger(dtype=torch.float16)
    *_, state = _gap_sequence(ledger)
    assert state["alpha"].dtype == torch.float16
    assert torch.isfinite(state["restoration_bonus"]).all()


def test_legacy_default_and_old_configs_are_unchanged():
    assert (
        ReliabilityCalibratedInnovation().active_strategy
        == "legacy_multiplicative"
    )
    root = Path(__file__).resolve().parents[1]
    for index in range(1, 7):
        text = (
            root
            / "configs/evidence_conserving"
            / f"mini_stage2_innovation_n{index}.py"
        ).read_text()
        assert "innovation_active_strategy" not in text


def test_protocol_metadata_is_not_a_model_gate():
    signature = inspect.signature(EvidenceLedger.update_queries)
    forbidden = {"protocol", "fault_camera", "scene_metadata"}
    assert forbidden.isdisjoint(signature.parameters)


def test_reset_clears_all_gap_runtime():
    ledger = _ledger()
    ledger.gap_active.fill_(True)
    ledger.pre_gap_strength.fill_(4.0)
    ledger.reset()
    for name in EvidenceLedger._STATE_NAMES:
        assert getattr(ledger, name) is None


def test_residual_preserving_is_positive_only_trust_region():
    ledger = _ledger(strategy="residual_preserving")
    state = _update(ledger)
    base = state["base_positive_evidence"]
    candidate = state["candidate_positive_evidence"]
    expected = 0.95 * base + 0.05 * candidate
    torch.testing.assert_close(
        state["actual_added_positive_evidence"], expected
    )
    torch.testing.assert_close(
        state["actual_added_negative_evidence"],
        state["base_negative_evidence"],
    )
