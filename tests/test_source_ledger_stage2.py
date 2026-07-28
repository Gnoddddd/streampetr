"""S2.2 source-ledger contracts."""

from __future__ import annotations

import io

import pytest
import torch
from torch import nn

from models.evidence_ledger import EvidenceLedger
from models.temporal_update import EvidenceConservingTemporalUpdate


def _ledger(
    cameras: int = 2,
    dtype: torch.dtype = torch.float32,
    enabled: bool = True,
) -> EvidenceLedger:
    ledger = EvidenceLedger(
        memory_len=3,
        num_cameras=cameras,
        temporal_update=EvidenceConservingTemporalUpdate(
            gamma=0.9,
            evidence_scale=2.0,
            enable_conservation=True,
            conservation_tolerance=1e-5,
        ),
        enable_source_ledger=enabled,
        source_decay=0.8,
        source_mass_tolerance=1e-5,
    )
    ledger.pre_update(torch.zeros(1, dtype=dtype), scene_tokens=["scene-a"])
    return ledger


def _update(ledger, raw_source, probabilities=(1.0, 0.0, 0.0), obs=1.0):
    raw = torch.tensor([[raw_source]], dtype=ledger.alpha.dtype)
    normalized = raw.float()
    normalized = normalized / normalized.sum(-1, keepdim=True).clamp_min(1e-6)
    normalized = normalized.to(raw.dtype)
    return ledger.update_queries(
        torch.tensor([[probabilities]], dtype=ledger.alpha.dtype),
        torch.tensor([[obs]], dtype=ledger.alpha.dtype),
        normalized,
        torch.ones(1, 1, dtype=ledger.alpha.dtype),
        torch.ones(1, 1, dtype=ledger.alpha.dtype),
        num_base_queries=0,
        num_propagated=1,
        raw_source_vector=raw,
    )


def test_single_camera_source_increment_equals_added_evidence():
    state = _update(_ledger(), [0.7, 0.0])
    total_added = (
        state["actual_added_positive_evidence"]
        + state["actual_added_negative_evidence"]
    )
    assert torch.allclose(
        state["current_source_increment"].sum(-1), total_added
    )
    assert state["current_source_increment"][0, 0, 1] == 0


def test_two_camera_source_distribution_is_normalized():
    state = _update(_ledger(), [1.0, 3.0])
    assert torch.allclose(
        state["current_source_distribution"],
        torch.tensor([[[0.25, 0.75]]]),
    )


def test_all_zero_source_only_decays_history():
    ledger = _ledger()
    ledger.source_evidence[:, 0] = torch.tensor([2.0, 1.0])
    state = _update(ledger, [0.0, 0.0])
    assert torch.allclose(
        state["source_evidence"], torch.tensor([[[1.6, 0.8]]])
    )
    assert torch.count_nonzero(state["current_source_increment"]) == 0


def test_failed_camera_gets_no_current_increment():
    state = _update(_ledger(), [0.9, 0.0])
    assert state["current_source_increment"][0, 0, 1] == 0


def test_source_strength_is_unnormalized_source_mass():
    state = _update(_ledger(), [1.0, 1.0])
    assert torch.allclose(
        state["source_strength"], state["source_evidence"].sum(-1)
    )


def test_provenance_is_nonnegative_and_sums_to_one_with_mass():
    state = _update(_ledger(), [1.0, 2.0])
    assert torch.all(state["provenance"] >= 0)
    assert torch.allclose(
        state["provenance"].sum(-1),
        torch.ones_like(state["source_strength"]),
    )


def test_zero_source_has_no_nan():
    state = _update(_ledger(), [0.0, 0.0])
    for key in (
        "current_source_distribution",
        "current_source_increment",
        "source_evidence",
        "provenance",
    ):
        assert torch.isfinite(state[key]).all()


def test_source_mass_residual_is_within_tolerance():
    state = _update(_ledger(), [0.2, 0.8])
    assert state["source_mass_residual"].abs().max() < 1e-5
    assert not torch.any(state["source_mass_violation"])


def test_scene_reset_clears_source_history():
    ledger = _ledger()
    ledger.source_evidence[:, 0] = 4.0
    ledger.pre_update(torch.ones(1), scene_tokens=["scene-b"])
    assert torch.count_nonzero(ledger.source_evidence) == 0


def test_batch_size_change_clears_source_history():
    ledger = _ledger()
    ledger.source_evidence[:, 0] = 4.0
    ledger.pre_update(
        torch.ones(2), scene_tokens=["scene-a", "scene-b"]
    )
    assert ledger.source_evidence.shape == (2, 3, 2)
    assert torch.count_nonzero(ledger.source_evidence) == 0


def test_query_count_changes_do_not_reuse_out_of_range_source_slots():
    ledger = _ledger()
    ledger.source_evidence[:, 0] = torch.tensor([1.0, 0.0])
    for query_count in (1, 5, 2):
        probabilities = torch.tensor(
            [[[0.0, 0.0, 1.0]]]
        ).expand(1, query_count, 3)
        state = ledger.update_queries(
            probabilities,
            torch.zeros(1, query_count),
            torch.zeros(1, query_count, 2),
            torch.zeros(1, query_count),
            torch.zeros(1, query_count),
            0,
            query_count,
            raw_source_vector=torch.zeros(1, query_count, 2),
        )
        assert state["source_evidence"].shape == (1, query_count, 2)


def test_commit_topk_keeps_source_alignment():
    ledger = _ledger()
    state = ledger.update_queries(
        torch.tensor([[[1.0, 0.0, 0.0]]]).expand(1, 3, 3),
        torch.ones(1, 3),
        torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]]),
        torch.ones(1, 3),
        torch.ones(1, 3),
        3,
        0,
        raw_source_vector=torch.tensor(
            [[[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]]
        ),
    )
    ledger.commit_topk(state, torch.tensor([[[1], [0]]]))
    assert torch.equal(
        ledger.source_evidence[:, :2],
        state["source_evidence"][:, [1, 0]],
    )


def test_defer_does_not_write_source_state():
    ledger = _ledger()
    state = _update(ledger, [1.0, 0.0], obs=0.0)
    ledger.commit_topk(
        state,
        torch.tensor([[[0]]]),
        valid_write_mask=state["write_mask"],
    )
    assert torch.count_nonzero(ledger.source_evidence[:, 0]) == 0


def test_runtime_export_import_round_trip_includes_source_state():
    ledger = _ledger()
    state = _update(ledger, [1.0, 2.0])
    ledger.commit_topk(state, torch.tensor([[[0]]]))
    restored = _ledger()
    restored.load_runtime_state(ledger.export_runtime_state())
    for name in ledger._STATE_NAMES:
        assert torch.equal(getattr(ledger, name), getattr(restored, name))


def test_state_dict_and_mmcv_checkpoint_exclude_source_runtime():
    ledger = _ledger()
    ledger.source_evidence[:, 0] = 9.0
    model = nn.Module()
    model.add_module("ledger", ledger)
    assert "ledger.source_evidence" not in model.state_dict()
    from mmcv.runner.checkpoint import get_state_dict

    assert "ledger.source_evidence" not in get_state_dict(model)
    payload = io.BytesIO()
    torch.save({"state_dict": get_state_dict(model)}, payload)
    payload.seek(0)
    assert "ledger.source_evidence" not in torch.load(payload)["state_dict"]


def test_cpu_fp16_source_update_is_safe():
    state = _update(_ledger(dtype=torch.float16), [0.2, 0.8])
    assert state["source_evidence"].dtype == torch.float16
    assert torch.isfinite(state["source_evidence"]).all()


def test_source_tracking_disabled_is_s21_compatible():
    disabled = _ledger(enabled=False)
    enabled = _ledger(enabled=True)
    for raw_source in ([1.0, 0.0], [0.3, 0.7], [0.0, 0.0]):
        left = _update(disabled, raw_source)
        right = _update(enabled, raw_source)
        for key in ("alpha", "beta", "action", "score_scale", "write_mask"):
            assert torch.equal(left[key], right[key])
        disabled.commit_topk(left, torch.tensor([[[0]]]), left["write_mask"])
        enabled.commit_topk(right, torch.tensor([[[0]]]), right["write_mask"])
        disabled.pre_update(torch.ones(1), scene_tokens=["scene-a"])
        enabled.pre_update(torch.ones(1), scene_tokens=["scene-a"])


def test_source_coupling_flags_are_rejected_in_tracking_only_stage():
    with pytest.raises(ValueError, match="tracking only"):
        EvidenceLedger(
            memory_len=2,
            use_source_ledger_for_evidence=True,
        )
