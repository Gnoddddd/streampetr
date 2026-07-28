import torch

from models.evidence_ledger import EvidenceLedger


def test_fresh_same_camera_is_new_but_stale_repeat_is_not():
    ledger = EvidenceLedger(memory_len=4, num_cameras=2)
    current = torch.tensor([[[1.0, 0.0]]])
    prior = torch.tensor([[[1.0, 0.0]]])
    stale = ledger.compute_novelty(current, prior, torch.tensor([[0.0]]))
    fresh = ledger.compute_novelty(current, prior, torch.tensor([[1.0]]))
    assert stale.item() < 1e-5
    assert fresh.item() > 0.999


def test_ledger_refreshes_on_scene_boundary_and_commits():
    ledger = EvidenceLedger(memory_len=4, num_cameras=2)
    ledger.pre_update(torch.tensor([0.0]))
    state = ledger.update_queries(
        ternary_probabilities=torch.tensor([[[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]]]),
        observability=torch.tensor([[1.0, 0.0]]),
        source_vector=torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
        fresh_ratio=torch.tensor([[1.0, 0.0]]),
        effective_count=torch.tensor([[1.0, 0.0]]),
        num_base_queries=2,
        num_propagated=0,
    )
    ledger.commit_topk(state, torch.tensor([[[0]]]), state["write_mask"])
    assert ledger.alpha is not None
    assert ledger.alpha.shape[1] == 5
    ledger.pre_update(torch.tensor([0.0]))
    assert torch.allclose(ledger.alpha, torch.ones_like(ledger.alpha))


def test_recover_preserves_historical_provenance_without_new_observation():
    ledger = EvidenceLedger(memory_len=2, num_cameras=2)
    ledger.pre_update(torch.tensor([0.0]))
    ledger.alpha[:, 0] = 5.0
    ledger.beta[:, 0] = 1.0
    ledger.provenance[:, 0] = torch.tensor([1.0, 0.0])
    state = ledger.update_queries(
        ternary_probabilities=torch.tensor([[[0.9, 0.05, 0.05]]]),
        observability=torch.tensor([[0.0]]),
        source_vector=torch.tensor([[[0.0, 0.0]]]),
        fresh_ratio=torch.tensor([[0.0]]),
        effective_count=torch.tensor([[0.0]]),
        num_base_queries=0,
        num_propagated=1,
    )
    assert state["write_mask"].item()
    assert torch.allclose(state["provenance"], torch.tensor([[[1.0, 0.0]]]))
