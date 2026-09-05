"""Pure helpers for the fault-aware assignment counterfactual gradient audit."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


SELECTION_METRICS = (
    "lost_degraded",
    "boundary_crossing",
    "retained",
    "easy_retained",
)


def scalar_gradient_relation(current: float, auxiliary: float) -> dict:
    """Describe focal-logit gradient directions before and after addition."""
    current, auxiliary = float(current), float(auxiliary)
    combined = current + auxiliary
    return {
        "current_gradient": current,
        "aux_gradient": auxiliary,
        "combined_gradient": combined,
        "current_update": -current,
        "aux_update_gain": -auxiliary,
        "combined_update": -combined,
        "harmful_current": bool(current > 0.0),
        "gradient_conflict": bool(current * auxiliary < 0.0),
        "harmful_reversed": bool(current > 0.0 and combined < 0.0),
    }


def vector_gradient_relation(current: Sequence[float], auxiliary: Sequence[float]) -> dict:
    """Project current/combined box gradients onto the same-GT aux direction."""
    current = np.asarray(current, dtype=np.float64)
    auxiliary = np.asarray(auxiliary, dtype=np.float64)
    if current.shape != auxiliary.shape:
        raise ValueError("gradient vectors must have identical shapes")
    current_norm = float(np.linalg.norm(current))
    auxiliary_norm = float(np.linalg.norm(auxiliary))
    combined = current + auxiliary
    combined_norm = float(np.linalg.norm(combined))
    if auxiliary_norm == 0.0:
        current_projection = combined_projection = cosine = float("nan")
    else:
        current_projection = float(np.dot(current, auxiliary) / auxiliary_norm)
        combined_projection = float(np.dot(combined, auxiliary) / auxiliary_norm)
        cosine = (
            float(np.dot(current, auxiliary) / (current_norm * auxiliary_norm))
            if current_norm > 0.0 else 0.0
        )
    return {
        "current_norm": current_norm,
        "aux_norm": auxiliary_norm,
        "combined_norm": combined_norm,
        "current_desired_projection": current_projection,
        "combined_desired_projection": combined_projection,
        "current_aux_cosine": cosine,
        "current_conflict": bool(np.isfinite(current_projection) and current_projection < 0.0),
        "combined_desired_positive": bool(
            np.isfinite(combined_projection) and combined_projection > 0.0
        ),
    }


def unit_key(row: dict) -> tuple[str, str]:
    return str(row["sample_token"]), str(row["gt_token"])


def select_equal_budget(rows: Iterable[dict], budget: int) -> dict:
    """Apply the preregistered generic and starvation-selective orderings."""
    rows = list(rows)
    budget = int(budget)
    if budget < 0 or budget > len(rows):
        raise ValueError("budget must be between zero and population size")
    if len({unit_key(row) for row in rows}) != len(rows):
        raise ValueError("selection units must be unique")
    generic = sorted(
        rows,
        key=lambda row: (
            float(row["pair_cost"]),
            str(row["sample_token"]),
            str(row["gt_token"]),
        ),
    )[:budget]
    selective = sorted(
        rows,
        key=lambda row: (
            -int(row["non_same_layer_count"]),
            -int(bool(row["final_non_same"])),
            float(row["fault_margin"]),
            float(row["pair_cost"]),
            str(row["sample_token"]),
            str(row["gt_token"]),
        ),
    )[:budget]
    return {
        "budget": budget,
        "generic": {unit_key(row) for row in generic},
        "selective": {unit_key(row) for row in selective},
    }


def selection_metrics(rows: Iterable[dict], selected: set[tuple[str, str]]) -> dict:
    chosen = [row for row in rows if unit_key(row) in selected]
    if not chosen:
        rates = {key: float("nan") for key in SELECTION_METRICS}
    else:
        rates = {
            key: float(np.mean([bool(row[key]) for row in chosen]))
            for key in SELECTION_METRICS
        }
    rates["n"] = len(chosen)
    rates["concentration"] = (
        (rates["lost_degraded"] + rates["boundary_crossing"]
         - rates["retained"] - rates["easy_retained"]) / 2.0
        if chosen else float("nan")
    )
    return rates


def bootstrap_selection_difference(
    rows: Iterable[dict],
    generic: set[tuple[str, str]],
    selective: set[tuple[str, str]],
    metric: str,
    seed: int,
    iterations: int = 5000,
) -> dict:
    """Paired universe bootstrap for fixed-selection rate differences."""
    rows = list(rows)
    if metric not in (*SELECTION_METRICS, "concentration"):
        raise KeyError(metric)
    generic_metrics = selection_metrics(rows, generic)
    selective_metrics = selection_metrics(rows, selective)
    estimate = selective_metrics[metric] - generic_metrics[metric]
    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(iterations)):
        sampled = [rows[index] for index in rng.integers(0, len(rows), len(rows))]
        left = selection_metrics(sampled, generic)[metric]
        right = selection_metrics(sampled, selective)[metric]
        if np.isfinite(left) and np.isfinite(right):
            values.append(right - left)
    if not values:
        low = high = float("nan")
    else:
        low, high = np.percentile(values, [2.5, 97.5])
    return {
        "estimate": float(estimate),
        "ci_low": float(low),
        "ci_high": float(high),
        "iterations": int(iterations),
    }
