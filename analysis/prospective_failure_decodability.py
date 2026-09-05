"""Metrics and cluster-bootstrap helpers for frozen prospective probes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def ece_equal_width(y, probability, bins: int = 10) -> float:
    y, probability = np.asarray(y, int), np.asarray(probability, float)
    edges = np.linspace(0., 1., bins + 1)
    indexes = np.minimum(np.searchsorted(edges, probability, side="right") - 1, bins - 1)
    indexes = np.maximum(indexes, 0)
    value = 0.
    for index in range(bins):
        selected = indexes == index
        if selected.any():
            value += selected.mean() * abs(float(y[selected].mean()) - float(probability[selected].mean()))
    return float(value)


def classification_metrics(y, probability) -> dict:
    y, probability = np.asarray(y, int), np.asarray(probability, float)
    if len(y) == 0 or np.unique(y).size < 2:
        return {"positive_base_rate": float(np.mean(y)) if len(y) else np.nan,
                "auprc": np.nan, "auroc": np.nan, "brier": np.nan, "ece10": np.nan}
    return {"positive_base_rate": float(y.mean()),
            "auprc": float(average_precision_score(y, probability)),
            "auroc": float(roc_auc_score(y, probability)),
            "brier": float(brier_score_loss(y, probability)),
            "ece10": ece_equal_width(y, probability, 10)}


def train_standardize(train, other):
    train, other = np.asarray(train, float), np.asarray(other, float)
    mean, scale = train.mean(axis=0), train.std(axis=0, ddof=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.
    return (train - mean) / scale, (other - mean) / scale, mean, scale


def clustered_metric_differences(rows: pd.DataFrame, representation: str, *,
                                 n_boot: int, seed: int) -> dict:
    """Paired scene then instance-trajectory bootstrap over all transition rows."""
    grouped = {
        str(scene): [trajectory.index.to_numpy() for _, trajectory in scene_rows.groupby("instance_token")]
        for scene, scene_rows in rows.groupby("scene_token", sort=False)
    }
    scenes = list(grouped)
    rng = np.random.default_rng(seed)
    delta_ap = np.full(n_boot, np.nan); delta_roc = np.full(n_boot, np.nan)
    above_base = np.full(n_boot, np.nan)
    for replicate in range(n_boot):
        indexes = []
        for scene in rng.choice(scenes, len(scenes), replace=True) if scenes else []:
            trajectories = grouped[str(scene)]
            for chosen in rng.integers(0, len(trajectories), len(trajectories)):
                indexes.extend(trajectories[int(chosen)].tolist())
        if not indexes:
            continue
        sampled = rows.loc[indexes]
        y = sampled.y_tp_to_fn.to_numpy(int)
        if np.unique(y).size < 2:
            continue
        baseline = classification_metrics(y, sampled.observable_probability)
        probe = classification_metrics(y, sampled[representation])
        delta_ap[replicate] = probe["auprc"] - baseline["auprc"]
        delta_roc[replicate] = probe["auroc"] - baseline["auroc"]
        above_base[replicate] = probe["auprc"] - float(y.mean())

    def summary(values, prefix):
        finite = values[np.isfinite(values)]
        return {f"{prefix}_ci_low": float(np.quantile(finite, .025)) if len(finite) else np.nan,
                f"{prefix}_ci_high": float(np.quantile(finite, .975)) if len(finite) else np.nan,
                f"{prefix}_finite_bootstraps": int(len(finite))}
    y = rows.y_tp_to_fn.to_numpy(int)
    baseline = classification_metrics(y, rows.observable_probability)
    probe = classification_metrics(y, rows[representation])
    return {"delta_auprc": probe["auprc"] - baseline["auprc"],
            "delta_auroc": probe["auroc"] - baseline["auroc"],
            "auprc_minus_base_rate": probe["auprc"] - probe["positive_base_rate"],
            **summary(delta_ap, "delta_auprc"), **summary(delta_roc, "delta_auroc"),
            **summary(above_base, "auprc_minus_base_rate")}
