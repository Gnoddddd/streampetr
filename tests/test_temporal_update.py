import torch

from models.temporal_update import EvidenceConservingTemporalUpdate


def test_no_new_observation_cannot_inflate_evidence():
    update = EvidenceConservingTemporalUpdate(gamma=0.9, evidence_scale=2.0)
    alpha = torch.tensor([6.0])
    beta = torch.tensor([4.0])
    result = update(
        alpha,
        beta,
        torch.tensor([0.9]),
        torch.tensor([0.1]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
    )
    prior_strength = alpha + beta - 2.0
    assert torch.allclose(result["strength"], 0.9 * prior_strength)
    prior_uncertainty = 2.0 / (alpha + beta)
    assert result["uncertainty"] >= prior_uncertainty


def test_new_independent_observation_adds_evidence():
    update = EvidenceConservingTemporalUpdate(gamma=0.9, evidence_scale=2.0)
    result = update(
        torch.ones(1),
        torch.ones(1),
        torch.tensor([0.8]),
        torch.tensor([0.2]),
        torch.ones(1),
        torch.ones(1),
        torch.tensor([2.0]),
    )
    assert result["strength"].item() > 0
    assert result["existence_probability"].item() > 0.5


def test_conservation_diagnostics_ignore_legitimate_new_evidence():
    updater = EvidenceConservingTemporalUpdate(gamma=0.9, evidence_scale=2.0)
    prior_alpha = torch.tensor([4.0, 4.0])
    prior_beta = torch.tensor([2.0, 2.0])
    result = updater(
        prior_alpha,
        prior_beta,
        torch.tensor([0.9, 0.9]),
        torch.tensor([0.1, 0.1]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.0, 1.0]),
    )
    assert bool(result["no_new_evidence"][0])
    assert not bool(result["no_new_evidence"][1])
    assert torch.allclose(result["conservation_ratio"][0], torch.tensor(0.9))
    assert torch.allclose(result["conservation_ratio"][1], torch.tensor(1.0))
    assert result["conservation_violation"].max().item() == 0.0
