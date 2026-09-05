"""Pure helpers for the prospective fault stress-reserve audit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.fault_boundary_root_cause import average_ranks


RISK_LABELS = ("<-0.2", "[-0.2,-0.1)", "[-0.1,0)", "[0,0.1)", "[0.1,0.2)", ">=0.2")
RISK_EDGES = np.asarray([-np.inf, -.2, -.1, 0., .1, .2, np.inf])


def stress_reserve(s_clean: float, s_k: float, s_probe: float) -> dict:
    values = np.asarray([s_clean, s_k, s_probe], dtype=float)
    if not np.isfinite(values).all():
        return {"R_K": np.nan, "J_p": np.nan, "M_p": np.nan}
    reserve = float(s_clean - s_k)
    stress = float(s_clean - s_probe)
    return {"R_K": reserve, "J_p": stress, "M_p": float(stress - reserve)}


def risk_bin(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, -1, dtype=np.int64)
    finite = np.isfinite(values)
    result[finite] = np.searchsorted(RISK_EDGES[1:-1], values[finite], side="right")
    return result


def spearman(left, right) -> float:
    left, right = np.asarray(left, float), np.asarray(right, float)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 2:
        return np.nan
    x, y = average_ranks(left[finite]), average_ranks(right[finite])
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def weighted_spearman(left, right, weight) -> float:
    left, right, weight = np.asarray(left, float), np.asarray(right, float), np.asarray(weight, float)
    finite = np.isfinite(left) & np.isfinite(right) & np.isfinite(weight) & (weight > 0)
    if finite.sum() < 2:
        return np.nan
    x, y, w = average_ranks(left[finite]), average_ranks(right[finite]), weight[finite]
    w = w / w.sum()
    x, y = x - np.sum(w * x), y - np.sum(w * y)
    denominator = np.sqrt(np.sum(w * x * x) * np.sum(w * y * y))
    return float(np.sum(w * x * y) / denominator) if denominator > 0 else np.nan


def two_stage_bootstrap(rows: pd.DataFrame, columns: list[str], statistic, *,
                        n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap scenes then trajectories; one row per trajectory is expected."""
    if rows.duplicated(["scene_token", "instance_token"]).any():
        raise ValueError("expected one row per scene/instance trajectory")
    data = rows[columns].to_numpy()
    positions = np.arange(len(rows))
    scene_groups = [positions[index] for index in rows.groupby("scene_token", sort=False).indices.values()]
    rng = np.random.default_rng(seed)
    output = np.full(n_boot, np.nan)
    for replicate in range(n_boot):
        sampled = []
        for scene_index in rng.integers(0, len(scene_groups), len(scene_groups)):
            group = scene_groups[int(scene_index)]
            sampled.append(group[rng.integers(0, len(group), len(group))])
        indexes = np.concatenate(sampled) if sampled else np.empty(0, dtype=int)
        if len(indexes):
            output[replicate] = statistic(data[indexes])
    return output


def interval(samples) -> tuple[float, float, int]:
    samples = np.asarray(samples, float)
    samples = samples[np.isfinite(samples)]
    if not len(samples):
        return np.nan, np.nan, 0
    return float(np.quantile(samples, .025)), float(np.quantile(samples, .975)), int(len(samples))
