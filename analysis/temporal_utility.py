"""Pure cohort, prediction and cluster-bootstrap helpers for temporal utility."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


CLASSES = (
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)


def utility_values(s_b: float, s_d: float, s_f: float) -> dict:
    values = np.asarray([s_b, s_d, s_f], dtype=float)
    if not np.isfinite(values).all():
        return {"U_available": np.nan, "U_realized": np.nan,
                "TU_loss": np.nan, "TU_retention": np.nan}
    available = float(s_b - s_f)
    realized = float(s_d - s_f)
    return {
        "U_available": available,
        "U_realized": realized,
        "TU_loss": float(available - realized),
        "TU_retention": float(realized / available) if available > 0 else np.nan,
    }


def classify_trajectory(rows: pd.DataFrame) -> tuple[str, int | None]:
    active = rows[rows.frame_idx >= 3].sort_values("frame_idx")
    if active.empty:
        return "no_fault_observation", None
    fault_miss = active.A_tp.astype(bool) & ~active.D_tp.astype(bool)
    if fault_miss.any():
        return "future_lost", int(active.loc[fault_miss, "frame_idx"].min())
    if active.D_tp.astype(bool).all():
        return "always_retained", None
    return "ambiguous_clean_failure", None


def match_trajectory_controls(anchors: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """One-to-one protocol-level matching using only frozen frame-2 factors."""

    records = []
    for protocol, protocol_outcomes in outcomes.groupby("protocol", sort=False):
        lost_ids = protocol_outcomes.loc[
            protocol_outcomes.trajectory_outcome == "future_lost", "trajectory_id"
        ].tolist()
        control_ids = protocol_outcomes.loc[
            protocol_outcomes.trajectory_outcome == "always_retained", "trajectory_id"
        ].tolist()
        left = anchors.set_index("trajectory_id").loc[lost_ids].reset_index()
        right = anchors.set_index("trajectory_id").loc[control_ids].reset_index()
        if len(right) < len(left):
            raise RuntimeError(f"insufficient retained trajectories for {protocol}")
        costs = np.empty((len(left), len(right)), dtype=float)
        for i, source in left.iterrows():
            for j, target in right.iterrows():
                costs[i, j] = (
                    100.0 * (source.gt_class != target.gt_class)
                    + abs(float(source.distance_m) - float(target.distance_m)) / 20.0
                    + abs(int(source.visibility_token) - int(target.visibility_token)) / 4.0
                    + abs(int(source.alternative_view_count) - int(target.alternative_view_count))
                    + j * 1e-12
                )
        rows, columns = linear_sum_assignment(costs)
        first_miss = protocol_outcomes.set_index("trajectory_id").first_miss_frame
        for i, j in zip(rows, columns):
            records.append({
                "protocol": protocol,
                "lost_trajectory_id": left.iloc[i].trajectory_id,
                "retained_trajectory_id": right.iloc[j].trajectory_id,
                "lost_scene_token": left.iloc[i].scene_token,
                "lost_instance_token": left.iloc[i].instance_token,
                "pseudo_miss_frame": int(first_miss[left.iloc[i].trajectory_id]),
                "match_cost": float(costs[i, j]),
            })
    return pd.DataFrame(records)


def clustered_scalar_ci(rows: pd.DataFrame, value: str, *, n_boot: int,
                        seed: int) -> dict:
    """Two-stage scene/trajectory bootstrap for a scalar estimate."""

    finite = rows[np.isfinite(rows[value])]
    grouped = {
        str(scene): [group[value].to_numpy(float) for _, group in scene_rows.groupby("instance_token")]
        for scene, scene_rows in finite.groupby("scene_token", sort=False)
    }
    scenes = list(grouped)
    observed = finite.groupby(["scene_token", "instance_token"])[value].median().to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = np.full(n_boot, np.nan)
    for b in range(n_boot):
        chosen_scenes = rng.choice(scenes, len(scenes), replace=True) if scenes else []
        values = []
        for scene in chosen_scenes:
            trajectories = grouped[str(scene)]
            for index in rng.integers(0, len(trajectories), len(trajectories)):
                values.append(float(np.median(trajectories[index])))
        if values:
            samples[b] = float(np.median(values))
    valid = samples[np.isfinite(samples)]
    return {
        "estimate": float(np.median(observed)) if len(observed) else np.nan,
        "ci_low": float(np.quantile(valid, .025)) if len(valid) else np.nan,
        "ci_high": float(np.quantile(valid, .975)) if len(valid) else np.nan,
        "n_rows": int(len(finite)),
        "n_trajectories": int(len(observed)),
        "n_scenes": int(len(scenes)),
    }


def clustered_mean_ci(rows: pd.DataFrame, value: str, *, n_boot: int,
                      seed: int) -> dict:
    """Two-stage scene/trajectory bootstrap for a mean/proportion."""

    finite = rows[np.isfinite(rows[value])]
    grouped = {
        str(scene): [group[value].to_numpy(float) for _, group in scene_rows.groupby("instance_token")]
        for scene, scene_rows in finite.groupby("scene_token", sort=False)
    }
    scenes = list(grouped)
    trajectory_values = finite.groupby(["scene_token", "instance_token"])[value].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = np.full(n_boot, np.nan)
    for b in range(n_boot):
        chosen_scenes = rng.choice(scenes, len(scenes), replace=True) if scenes else []
        values = []
        for scene in chosen_scenes:
            trajectories = grouped[str(scene)]
            for index in rng.integers(0, len(trajectories), len(trajectories)):
                values.append(float(np.mean(trajectories[index])))
        if values:
            samples[b] = float(np.mean(values))
    valid = samples[np.isfinite(samples)]
    return {
        "estimate": float(np.mean(trajectory_values)) if len(trajectory_values) else np.nan,
        "ci_low": float(np.quantile(valid, .025)) if len(valid) else np.nan,
        "ci_high": float(np.quantile(valid, .975)) if len(valid) else np.nan,
        "n_rows": int(len(finite)), "n_trajectories": int(len(trajectory_values)),
        "n_scenes": int(len(scenes)),
    }


def auroc(y, score) -> float:
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=float)
    positive, negative = y == 1, y == 0
    n_pos, n_neg = int(positive.sum()), int(negative.sum())
    if not n_pos or not n_neg:
        return np.nan
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    sorted_score = score[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def cluster_bootstrap_delta_auroc(rows: pd.DataFrame, *, n_boot: int,
                                  seed: int) -> dict:
    finite = rows[np.isfinite(rows.baseline_score) & np.isfinite(rows.augmented_score)]
    grouped = {
        str(scene): [group.index.to_numpy() for _, group in scene_rows.groupby("instance_token")]
        for scene, scene_rows in finite.groupby("scene_token", sort=False)
    }
    scenes = list(grouped)
    rng = np.random.default_rng(seed)
    deltas = np.full(n_boot, np.nan)
    for b in range(n_boot):
        sampled = []
        for scene in rng.choice(scenes, len(scenes), replace=True) if scenes else []:
            trajectories = grouped[str(scene)]
            for index in rng.integers(0, len(trajectories), len(trajectories)):
                sampled.extend(trajectories[index].tolist())
        if sampled:
            chosen = finite.loc[sampled]
            base = auroc(chosen.outcome, chosen.baseline_score)
            augmented = auroc(chosen.outcome, chosen.augmented_score)
            if np.isfinite(base) and np.isfinite(augmented):
                deltas[b] = augmented - base
    valid = deltas[np.isfinite(deltas)]
    baseline = auroc(finite.outcome, finite.baseline_score)
    augmented = auroc(finite.outcome, finite.augmented_score)
    return {
        "baseline_auroc": baseline,
        "augmented_auroc": augmented,
        "delta_auroc": augmented - baseline,
        "delta_ci_low": float(np.quantile(valid, .025)) if len(valid) else np.nan,
        "delta_ci_high": float(np.quantile(valid, .975)) if len(valid) else np.nan,
        "n": int(len(finite)), "positives": int(finite.outcome.sum()),
        "negatives": int((1 - finite.outcome).sum()),
        "n_scenes": int(finite.scene_token.nunique()),
        "n_trajectories": int(finite.trajectory_id.nunique()),
        "finite_bootstraps": int(len(valid)),
    }
