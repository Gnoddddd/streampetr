import numpy as np

from analysis.reviewer_recovery import (
    Candidate,
    binary_auroc,
    cluster_deduplicate,
    delayed_promotions,
    motion_allocation,
    secondary_allocation,
    survival_state,
)


def candidate(index, score=None, label=0, center=None):
    return Candidate(
        index,
        label,
        float(index if score is None else score),
        np.asarray(center if center is not None else [index, 0, 0], float),
        np.zeros(2),
        np.asarray([index], float),
        np.zeros(9),
    )


def test_fixed_budget_allocations_replace_not_append():
    values = [candidate(i) for i in range(120)]
    secondary = secondary_allocation(
        values, {i: float(120 - i) for i in range(120)}, 100
    )
    motion = motion_allocation(
        values, [candidate(999, label=0, center=[0, 0, 0])], 1.0, 100
    )
    assert len(secondary) == len({value.lineage for value in secondary}) == 100
    assert len(motion) == len({value.lineage for value in motion}) == 100


def test_cluster_fallback_deduplicates_cross_layer_refinement():
    retained = cluster_deduplicate(
        np.asarray([1, 1, 2]),
        np.asarray([[0, 0, 0], [0.1, 0, 0], [0, 0, 0]]),
        np.zeros((3, 2)),
        np.asarray([0, 5, 5]),
        np.asarray([0.5, 0.6, 0.9]),
    )
    assert retained.tolist() == [2, 1]


def test_delayed_confirmation_is_causal_and_does_not_rewrite_past():
    values = ["a", None, "a", "a", "a"]
    assert delayed_promotions(values, 3, 2) == [False, False, True, True, True]
    assert delayed_promotions(values, 5, 5) == [False] * 5


def test_survival_termination_precedes_observability():
    bounds = np.asarray([-10, -10, -2, 10, 10, 2])
    assert survival_state(np.asarray([11, 0, 0]), False, True, bounds).startswith(
        "Terminated"
    )
    assert survival_state(np.zeros(3), False, False, bounds) == "Unobserved"
    assert survival_state(np.zeros(3), False, True, bounds) == "Absent"
    assert survival_state(np.zeros(3), True, False, bounds) == "Present"


def test_binary_auroc_exact_ordering():
    assert binary_auroc(
        np.asarray([1, 1, 0, 0]), np.asarray([0.9, 0.8, 0.2, 0.1])
    ) == 1.0
