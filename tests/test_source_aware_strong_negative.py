import torch

from models.keep_recover_defer import (
    Action,
    KeepRecoverDeferPolicy,
)


def build_policy():
    return KeepRecoverDeferPolicy(
        keep_observability=0.45,
        keep_presence=0.28,
        keep_max_uncertainty=0.55,
        recover_presence=0.30,
        recover_max_uncertainty=0.80,
        recover_max_age=3,
        recover_min_prior_strength=0.50,
        strong_negative=0.60,
    )


def recover_inputs():
    return dict(
        observability=torch.tensor([0.20]),
        existence_probability=torch.tensor([0.35]),
        uncertainty=torch.tensor([0.70]),
        age_since_observation=torch.tensor([1.0]),
        negative_probability=torch.tensor([0.70]),
        prior_strength=torch.tensor([1.0]),
    )


def test_ternary_source_keeps_strong_negative_gate():
    policy = build_policy()

    result = policy(
        **recover_inputs(),
        use_strong_negative=True,
    )

    assert result["action"].item() == int(
        Action.DEFER
    )


def test_classification_source_skips_strong_negative_gate():
    policy = build_policy()

    result = policy(
        **recover_inputs(),
        use_strong_negative=False,
    )

    assert result["action"].item() == int(
        Action.RECOVER
    )


def test_keep_action_is_not_changed_by_source():
    policy = build_policy()

    inputs = dict(
        observability=torch.tensor([0.90]),
        existence_probability=torch.tensor([0.35]),
        uncertainty=torch.tensor([0.50]),
        age_since_observation=torch.tensor([0.0]),
        negative_probability=torch.tensor([0.90]),
        prior_strength=torch.tensor([1.0]),
    )

    ternary_result = policy(
        **inputs,
        use_strong_negative=True,
    )
    classification_result = policy(
        **inputs,
        use_strong_negative=False,
    )

    assert ternary_result["action"].item() == int(
        Action.KEEP
    )
    assert classification_result[
        "action"
    ].item() == int(Action.KEEP)
