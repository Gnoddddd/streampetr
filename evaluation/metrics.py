"""PartialObs-3D diagnostic metrics independent of the nuScenes evaluator."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np


def unsupported_false_positive_rate(
    is_false_positive: Sequence[bool], supported: Sequence[bool]
) -> float:
    false_positive = np.asarray(is_false_positive, dtype=bool)
    supported = np.asarray(supported, dtype=bool)
    unsupported = ~supported
    denominator = int(unsupported.sum())
    if denominator == 0:
        return 0.0
    return float((false_positive & unsupported).sum() / denominator)


def stale_object_persistence(stale_lengths: Iterable[int]) -> float:
    values = np.asarray(list(stale_lengths), dtype=np.float64)
    return float(values.mean()) if values.size else 0.0


def evidence_inflation_ratio(
    evidence_strength: Sequence[float], no_new_observation: Sequence[bool]
) -> float:
    strength = np.asarray(evidence_strength, dtype=np.float64)
    no_new = np.asarray(no_new_observation, dtype=bool)
    if strength.size < 2:
        return 1.0
    ratios = []
    for index in range(1, strength.size):
        if no_new[index] and strength[index - 1] > 1e-9:
            ratios.append(strength[index] / strength[index - 1])
    return float(max(ratios)) if ratios else 1.0


def reacquisition_delay(
    correct_after_recovery: Sequence[bool], recovery_index: int
) -> int:
    correct = np.asarray(correct_after_recovery, dtype=bool)
    for index in range(max(int(recovery_index), 0), len(correct)):
        if correct[index]:
            return index - int(recovery_index)
    return max(len(correct) - int(recovery_index), 0)


def risk_coverage_curve(
    errors: Sequence[float], uncertainties: Sequence[float], steps: int = 20
) -> Dict[str, np.ndarray]:
    errors = np.asarray(errors, dtype=np.float64)
    uncertainties = np.asarray(uncertainties, dtype=np.float64)
    if errors.shape != uncertainties.shape:
        raise ValueError("errors and uncertainties must have the same shape")
    if errors.size == 0:
        return {"coverage": np.array([]), "risk": np.array([])}
    order = np.argsort(uncertainties)
    sorted_errors = errors[order]
    coverages = np.linspace(0.05, 1.0, max(int(steps), 1))
    risks = []
    for coverage in coverages:
        count = max(1, int(np.ceil(coverage * len(sorted_errors))))
        risks.append(sorted_errors[:count].mean())
    return {"coverage": coverages, "risk": np.asarray(risks)}
