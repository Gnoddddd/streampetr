from evaluation.metrics import (
    evidence_inflation_ratio,
    reacquisition_delay,
    stale_object_persistence,
    unsupported_false_positive_rate,
)


def test_partialobs_metrics():
    assert unsupported_false_positive_rate([True, False], [False, False]) == 0.5
    assert stale_object_persistence([2, 4]) == 3.0
    assert evidence_inflation_ratio([10, 9, 8.1], [False, True, True]) <= 1.0
    assert reacquisition_delay([False, False, True], 1) == 1
