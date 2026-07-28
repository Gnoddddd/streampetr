"""S2.1 evidence-conservation ledger contracts."""

from __future__ import annotations

import io

import torch
from torch import nn

from models.evidence_ledger import EvidenceLedger
from models.keep_recover_defer import Action
from models.temporal_update import EvidenceConservingTemporalUpdate


def _ledger(dtype: torch.dtype = torch.float32) -> EvidenceLedger:
    temporal_update = EvidenceConservingTemporalUpdate(
        gamma=0.9,
        evidence_scale=2.0,
        enable_conservation=True,
        reliable_observation_threshold=0.05,
    )
    ledger = EvidenceLedger(
        memory_len=2,
        num_cameras=2,
        temporal_update=temporal_update,
    )
    ledger.pre_update(torch.zeros(1, dtype=dtype))
    return ledger


def _update(
    ledger: EvidenceLedger,
    probabilities,
    observability: float,
    effective_count: float,
):
    dtype = ledger.alpha.dtype
    return ledger.update_queries(
        ternary_probabilities=torch.tensor(
            [[probabilities]], dtype=dtype
        ),
        observability=torch.tensor([[observability]], dtype=dtype),
        source_vector=torch.tensor([[[1.0, 0.0]]], dtype=dtype),
        fresh_ratio=torch.tensor([[1.0]], dtype=dtype),
        effective_count=torch.tensor([[effective_count]], dtype=dtype),
        num_base_queries=0,
        num_propagated=1,
    )


def test_no_new_observation_only_decays_toward_beta_prior():
    ledger = _ledger()
    ledger.alpha[:, 0] = 5.0
    ledger.beta[:, 0] = 3.0
    state = _update(ledger, [0.9, 0.1, 0.0], 0.0, 0.0)

    assert torch.allclose(state["alpha"], torch.tensor([[4.6]]))
    assert torch.allclose(state["beta"], torch.tensor([[2.8]]))
    assert state["strength"].item() <= 6.0
    assert state["no_new_evidence"].item()
    assert abs(state["conservation_residual"].item()) < 1e-6


def test_stage1_default_path_keeps_legacy_soft_observation_gate():
    update = EvidenceConservingTemporalUpdate(
        gamma=0.9,
        evidence_scale=2.0,
    )
    state = update(
        torch.ones(1, 1),
        torch.ones(1, 1),
        torch.ones(1, 1),
        torch.zeros(1, 1),
        torch.full((1, 1), 0.01),
        torch.ones(1, 1),
        torch.ones(1, 1),
    )

    assert not update.enable_conservation
    assert torch.allclose(state["positive_evidence"], torch.tensor([[0.02]]))


def test_reliable_positive_evidence_increases_alpha_only():
    ledger = _ledger()
    state = _update(ledger, [1.0, 0.0, 0.0], 1.0, 1.0)

    assert state["alpha"].item() > 1.0
    assert state["beta"].item() == 1.0
    assert state["positive_evidence"].item() > 0.0


def test_reliable_negative_evidence_increases_beta_only():
    ledger = _ledger()
    state = _update(ledger, [0.0, 1.0, 0.0], 1.0, 1.0)

    assert state["alpha"].item() == 1.0
    assert state["beta"].item() > 1.0
    assert state["negative_evidence"].item() > 0.0


def test_unobserved_does_not_add_positive_or_negative_evidence():
    ledger = _ledger()
    state = _update(ledger, [0.0, 0.0, 1.0], 1.0, 1.0)

    assert state["alpha"].item() == 1.0
    assert state["beta"].item() == 1.0
    assert state["positive_evidence"].item() == 0.0
    assert state["negative_evidence"].item() == 0.0


def test_scene_boundary_via_prev_exists_resets_all_runtime_state():
    ledger = _ledger()
    ledger.alpha[:, 0] = 4.0
    ledger.beta[:, 0] = 2.0
    ledger.provenance[:, 0] = torch.tensor([0.75, 0.25])
    ledger.age[:, 0] = 3.0

    ledger.pre_update(torch.zeros(1))

    assert torch.allclose(ledger.alpha, torch.ones_like(ledger.alpha))
    assert torch.allclose(ledger.beta, torch.ones_like(ledger.beta))
    assert torch.count_nonzero(ledger.provenance) == 0
    assert torch.count_nonzero(ledger.age) == 0
    assert torch.all(ledger.action == int(Action.DEFER))


def test_defer_never_writes_to_memory():
    ledger = _ledger()
    state = _update(ledger, [0.0, 0.0, 1.0], 0.0, 0.0)

    assert state["action"].item() == int(Action.DEFER)
    assert not state["write_mask"].item()
    ledger.commit_topk(
        state,
        torch.tensor([[[0]]]),
        valid_write_mask=state["write_mask"],
    )
    assert ledger.alpha[0, 0].item() == 1.0
    assert ledger.beta[0, 0].item() == 1.0


def test_cpu_fp16_update_is_safe_and_preserves_public_dtype():
    ledger = _ledger(torch.float16)
    state = _update(ledger, [0.8, 0.1, 0.1], 1.0, 1.0)

    assert state["alpha"].dtype == torch.float16
    assert state["conservation_residual"].dtype == torch.float16
    assert torch.isfinite(state["alpha"]).all()
    assert torch.isfinite(state["beta"]).all()


def test_model_state_dict_excludes_scene_runtime_buffers():
    ledger = _ledger()
    first = _update(ledger, [0.9, 0.1, 0.0], 1.0, 1.0)
    ledger.commit_topk(first, torch.tensor([[[0]]]), first["write_mask"])

    model = nn.Module()
    model.add_module("ledger", ledger)
    keys = tuple(model.state_dict())

    for name in ledger._STATE_NAMES:
        assert f"ledger.{name}" not in keys


def test_mmcv_checkpoint_state_dict_excludes_runtime_buffers():
    from mmcv.runner.checkpoint import get_state_dict

    ledger = _ledger()
    ledger.alpha[:, 0] = 17.0
    model = nn.Module()
    model.add_module("ledger", ledger)

    keys = tuple(get_state_dict(model))

    for name in ledger._STATE_NAMES:
        assert f"ledger.{name}" not in keys


def test_explicit_runtime_state_round_trip_preserves_next_update():
    ledger = _ledger()
    first = _update(ledger, [0.9, 0.1, 0.0], 1.0, 1.0)
    ledger.commit_topk(first, torch.tensor([[[0]]]), first["write_mask"])
    saved = ledger.export_runtime_state()

    restored = _ledger()
    restored.load_runtime_state(saved)
    for name in ledger._STATE_NAMES:
        original = getattr(ledger, name)
        loaded = getattr(restored, name)
        assert torch.equal(original, loaded)

    ledger.pre_update(torch.ones(1))
    restored.pre_update(torch.ones(1))
    expected = _update(ledger, [0.7, 0.2, 0.1], 0.0, 0.0)
    actual = _update(restored, [0.7, 0.2, 0.1], 0.0, 0.0)
    for key in ("alpha", "beta", "strength", "conservation_residual"):
        assert torch.equal(expected[key], actual[key])


def test_reset_clears_every_runtime_buffer_and_scene_identity():
    ledger = _ledger()
    state = _update(ledger, [1.0, 0.0, 0.0], 1.0, 1.0)
    ledger.commit_topk(state, torch.tensor([[[0]]]), state["write_mask"])
    ledger.pre_update(torch.ones(1), scene_tokens=["scene-a"])

    ledger.reset()

    for name in ledger._STATE_NAMES:
        assert getattr(ledger, name) is None
    assert ledger._scene_tokens is None
    assert ledger.scene_reset_count == 0


def test_batch_size_change_reinitializes_instead_of_reusing_old_scene():
    ledger = _ledger()
    ledger.alpha[:, 0] = 9.0

    ledger.pre_update(
        torch.ones(2),
        scene_tokens=["scene-a", "scene-b"],
    )

    assert ledger.alpha.shape == (2, ledger.memory_len)
    assert torch.allclose(ledger.alpha, torch.ones_like(ledger.alpha))


def test_query_count_change_only_reuses_valid_propagated_slots():
    ledger = _ledger()
    ledger.alpha[:, 0] = 5.0
    for query_count in (5, 1, 7):
        probabilities = torch.tensor(
            [[[0.0, 0.0, 1.0]]]
        ).expand(1, query_count, 3)
        state = ledger.update_queries(
            probabilities,
            torch.zeros(1, query_count),
            torch.zeros(1, query_count, 2),
            torch.zeros(1, query_count),
            torch.zeros(1, query_count),
            num_base_queries=0,
            num_propagated=query_count,
        )
        assert state["alpha"].shape == (1, query_count)
        assert state["alpha"][0, 0] > 1.0


def test_scene_token_change_resets_even_if_prev_exists_is_true():
    ledger = _ledger()
    ledger.pre_update(torch.zeros(1), scene_tokens=["scene-a"])
    ledger.alpha[:, 0] = 8.0
    ledger.pre_update(torch.ones(1), scene_tokens=["scene-b"])

    assert torch.allclose(ledger.alpha, torch.ones_like(ledger.alpha))
    assert ledger.last_scene_reset
    assert ledger._scene_tokens == ("scene-b",)


def test_enabled_conservation_residual_and_unsupported_growth_contracts():
    ledger = _ledger()
    ledger.alpha[:, 0] = 4.0
    ledger.beta[:, 0] = 2.0

    reliable = _update(ledger, [0.8, 0.2, 0.0], 1.0, 2.0)
    assert reliable["conservation_residual"].abs().max() < 1e-5
    assert not torch.any(reliable["conservation_violation_mask"])

    unsupported = _update(ledger, [0.8, 0.2, 0.0], 0.049, 2.0)
    assert not torch.any(unsupported["reliable_observation"])
    assert not torch.any(unsupported["unsupported_growth"])
    assert torch.allclose(
        unsupported["strength"],
        unsupported["no_new_evidence_strength"],
        atol=1e-6,
    )


def test_serialized_checkpoint_excludes_last_batch_runtime_state():
    ledger = _ledger()
    ledger.alpha[:, 0] = 123.0
    model = nn.Module()
    model.add_module("ledger", ledger)

    payload = io.BytesIO()
    torch.save({"state_dict": model.state_dict()}, payload)
    payload.seek(0)
    checkpoint = torch.load(payload, map_location="cpu")

    assert not any(
        key.startswith("ledger.") and key.split(".")[-1] in ledger._STATE_NAMES
        for key in checkpoint["state_dict"]
    )
