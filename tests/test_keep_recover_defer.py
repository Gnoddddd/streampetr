import torch

from models.keep_recover_defer import Action, KeepRecoverDeferPolicy


def test_recover_requires_historical_evidence_and_weak_current_observation():
    policy = KeepRecoverDeferPolicy(recover_min_prior_strength=0.5)
    result = policy(
        observability=torch.tensor([0.1, 0.1, 0.9]),
        existence_probability=torch.tensor([0.7, 0.7, 0.7]),
        uncertainty=torch.tensor([0.6, 0.6, 0.6]),
        age_since_observation=torch.tensor([1.0, 1.0, 1.0]),
        negative_probability=torch.tensor([0.1, 0.1, 0.1]),
        prior_strength=torch.tensor([0.0, 2.0, 2.0]),
    )
    assert result["action"].tolist() == [
        int(Action.DEFER),
        int(Action.RECOVER),
        int(Action.DEFER),
    ]
