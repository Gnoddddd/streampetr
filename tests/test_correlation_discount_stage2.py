"""S2.4 correlation-discount isolation and disabled-path contracts."""

from __future__ import annotations

import io

import pytest
import torch
from torch import nn

from models.evidence_ledger import EvidenceLedger
from models.keep_recover_defer import Action
from models.observability_head import GeometricObservabilityHead
from models.temporal_update import EvidenceConservingTemporalUpdate


def _camera_matrix(dtype=torch.float32, device="cpu"):
    matrix = torch.eye(4, dtype=dtype, device=device)
    matrix[0, 0] = 100.0
    matrix[1, 1] = 100.0
    matrix[0, 2] = 50.0
    matrix[1, 2] = 50.0
    return matrix


def _observability(head, dtype=torch.float32, device="cpu", batch=1):
    query = torch.tensor(
        [[[0.0, 0.0, 5.0]]],
        dtype=dtype,
        device=device,
    ).expand(batch, 1, 3)
    matrices = _camera_matrix(dtype, device).view(1, 1, 4, 4)
    matrices = matrices.repeat(batch, head.num_cameras, 1, 1)
    return head(query, matrices, (100, 100))


def _ledger(batch=1, dtype=torch.float32):
    ledger = EvidenceLedger(
        memory_len=4,
        num_cameras=2,
        temporal_update=EvidenceConservingTemporalUpdate(
            gamma=0.9,
            evidence_scale=2.0,
            enable_conservation=True,
        ),
        enable_source_ledger=True,
    )
    ledger.pre_update(
        torch.zeros(batch, dtype=dtype),
        scene_tokens=[f"scene-{index}" for index in range(batch)],
    )
    return ledger


def _ledger_update(ledger, queries, effective_count):
    batch = ledger.alpha.shape[0]
    dtype = ledger.alpha.dtype
    probabilities = torch.tensor(
        [0.8, 0.1, 0.1], dtype=dtype
    ).view(1, 1, 3).expand(batch, queries, 3)
    source = torch.tensor(
        [0.5, 0.5], dtype=dtype
    ).view(1, 1, 2).expand(batch, queries, 2)
    return ledger.update_queries(
        ternary_probabilities=probabilities,
        observability=torch.ones(batch, queries, dtype=dtype),
        source_vector=source,
        fresh_ratio=torch.ones(batch, queries, dtype=dtype),
        effective_count=effective_count,
        num_base_queries=max(queries - 2, 0),
        num_propagated=min(queries, 2),
        raw_source_vector=source,
    )


def test_disabled_path_does_not_read_correlation_matrix():
    head = GeometricObservabilityHead(
        num_cameras=2,
        enable_correlation_discount=False,
    )
    head.camera_correlation.fill_(float("nan"))
    output = _observability(head)
    assert torch.equal(
        output["effective_count"],
        torch.ones_like(output["effective_count"]),
    )
    assert torch.isfinite(output["observability"]).all()


def test_enabled_path_reproduces_historical_fixed_matrix_formula():
    head = GeometricObservabilityHead(
        num_cameras=2,
        correlation_matrix=((1.0, 0.5), (0.5, 1.0)),
        enable_correlation_discount=True,
    )
    output = _observability(head)
    expected = torch.tensor([[4.0 / 3.0]])
    assert torch.allclose(output["effective_count"], expected, atol=1e-5)


def test_historical_scaffold_changed_s22_evidence_vs_true_bypass():
    legacy_head = GeometricObservabilityHead(
        num_cameras=2,
        enable_correlation_discount=True,
    )
    disabled_head = GeometricObservabilityHead(
        num_cameras=2,
        enable_correlation_discount=False,
    )
    legacy_count = _observability(legacy_head)["effective_count"]
    disabled_count = _observability(disabled_head)["effective_count"]
    assert torch.any(legacy_count != disabled_count)

    update = EvidenceConservingTemporalUpdate(
        gamma=0.9,
        evidence_scale=2.0,
        enable_conservation=True,
    )
    common = (
        torch.ones(1, 1),
        torch.ones(1, 1),
        torch.ones(1, 1),
        torch.zeros(1, 1),
        torch.ones(1, 1),
        torch.ones(1, 1),
    )
    historical = update(*common, legacy_count)
    bypassed = update(*common, None)
    assert not torch.equal(historical["alpha"], bypassed["alpha"])


def test_none_effective_count_uses_legacy_temporal_update_path():
    ledger = _ledger()
    state = _ledger_update(ledger, queries=3, effective_count=None)
    assert torch.equal(
        state["effective_count"],
        torch.ones_like(state["effective_count"]),
    )
    assert torch.allclose(
        state["actual_added_positive_evidence"],
        torch.full((1, 3), 1.6),
    )
    assert state["conservation_residual"].abs().max() < 1e-5
    assert not torch.any(state["source_mass_violation"])


def test_previous_action_batch_query_and_topk_are_aligned():
    ledger = _ledger(batch=2)
    first = _ledger_update(ledger, queries=5, effective_count=None)
    indexes = torch.tensor(
        [[[4], [1], [3]], [[0], [2], [4]]]
    )
    ledger.commit_topk(first, indexes, first["write_mask"])
    ledger.pre_update(
        torch.ones(2),
        scene_tokens=["scene-0", "scene-1"],
    )
    second = _ledger_update(ledger, queries=3, effective_count=None)
    assert second["previous_action"].shape == (2, 3)
    assert torch.equal(
        second["previous_action"][:, 1:3],
        ledger.action[:, :2],
    )

    ledger.pre_update(
        torch.ones(1),
        scene_tokens=["new-scene"],
    )
    assert ledger.alpha.shape == (1, 4)
    assert torch.all(ledger.action == int(Action.DEFER))


def test_old_observability_state_dict_loads_strictly_when_disabled():
    historical = GeometricObservabilityHead(
        num_cameras=2,
        enable_correlation_discount=True,
    )
    disabled = GeometricObservabilityHead(
        num_cameras=2,
        enable_correlation_discount=False,
    )
    result = disabled.load_state_dict(historical.state_dict(), strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys
    assert "camera_correlation" in disabled.state_dict()
    assert all(
        "enable_correlation_discount" not in key
        for key in disabled.state_dict()
    )


def test_disabled_path_adds_no_runtime_checkpoint_state():
    head = GeometricObservabilityHead(
        num_cameras=2,
        enable_correlation_discount=False,
    )
    ledger = _ledger()
    model = nn.Module()
    model.add_module("observability", head)
    model.add_module("ledger", ledger)
    keys = tuple(model.state_dict())
    for runtime_name in ledger._STATE_NAMES:
        assert f"ledger.{runtime_name}" not in keys
    payload = io.BytesIO()
    torch.save({"state_dict": model.state_dict()}, payload)
    payload.seek(0)
    checkpoint = torch.load(payload)
    assert tuple(checkpoint["state_dict"]) == keys


def test_disabled_path_cpu_fp16_is_finite():
    head = GeometricObservabilityHead(
        num_cameras=2,
        enable_correlation_discount=False,
    )
    output = _observability(head, dtype=torch.float16)
    assert output["effective_count"].dtype == torch.float16
    assert torch.isfinite(output["effective_count"]).all()
    state = _ledger_update(
        _ledger(dtype=torch.float16),
        queries=2,
        effective_count=None,
    )
    assert state["alpha"].dtype == torch.float16
    assert torch.isfinite(state["alpha"]).all()


def test_disabled_path_gpu_fp16_is_finite():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the float16 isolation test")
    device = torch.device("cuda")
    head = GeometricObservabilityHead(
        num_cameras=2,
        enable_correlation_discount=False,
    ).to(device)
    output = _observability(
        head,
        dtype=torch.float16,
        device=device,
    )
    assert output["effective_count"].is_cuda
    assert output["effective_count"].dtype == torch.float16
    assert torch.isfinite(output["effective_count"]).all()
