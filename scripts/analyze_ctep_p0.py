#!/usr/bin/env python3
"""Analyze the preregistered train-split CTEP P0 replay."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/full_nuscenes/ctep_method_activation"
PROTOCOLS = ("blur_back", "crash_back", "dark_back")
GROUPS = {
    "lost": "lost",
    "retained": "retained",
    "history_sensitive_lost": "history_sensitive_lost",
    "easy": "easy",
}
METRICS = ("A_minus_C_s_pos", "B_minus_D_s_pos")
BOOTSTRAPS = 5000
SEED = 424242


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def atomic_json(value: object, path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def selected(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    return frame[frame[GROUPS[group]].astype(bool)]


def scene_aggregates(frame: pd.DataFrame, metric: str) -> pd.Series:
    values = frame[["scene_token", "instance_token", metric]].copy()
    values[metric] = pd.to_numeric(values[metric], errors="coerce")
    values = values[np.isfinite(values[metric])]
    if values.empty:
        return pd.Series(dtype=float)
    trajectories = values.groupby(
        ["scene_token", "instance_token"], observed=True, sort=False
    )[metric].median()
    return trajectories.groupby(level="scene_token", sort=False).mean()


def bootstrap(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan, math.nan
    estimate = float(values.mean())
    if len(values) == 1:
        return estimate, estimate, estimate
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    samples = values[indexes].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return estimate, float(low), float(high)


def ci(frame: pd.DataFrame, metric: str, seed: int) -> dict:
    scenes = scene_aggregates(frame, metric)
    estimate, low, high = bootstrap(scenes.to_numpy(), seed)
    return {
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "event_n": int(pd.to_numeric(frame[metric], errors="coerce").notna().sum()),
        "trajectory_n": int(frame.loc[
            pd.to_numeric(frame[metric], errors="coerce").notna(),
            ["scene_token", "instance_token"],
        ].drop_duplicates().shape[0]),
        "scene_n": len(scenes),
        "bootstrap_n": BOOTSTRAPS,
        "seed": seed,
    }


def contrast(frame: pd.DataFrame, left: str, right: str, metric: str, seed: int) -> dict:
    left_values = scene_aggregates(selected(frame, left), metric)
    right_values = scene_aggregates(selected(frame, right), metric)
    shared = left_values.index.intersection(right_values.index)
    estimate, low, high = bootstrap(
        (left_values.loc[shared] - right_values.loc[shared]).to_numpy(), seed
    )
    return {
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "event_n": int(len(selected(frame, left)) + len(selected(frame, right))),
        "trajectory_n": int(
            pd.concat([selected(frame, left), selected(frame, right)])[
                ["scene_token", "instance_token"]
            ].drop_duplicates().shape[0]
        ),
        "scene_n": len(shared),
        "bootstrap_n": BOOTSTRAPS,
        "seed": seed,
    }


def main() -> None:
    frames = []
    coverage_rows = []
    for protocol in PROTOCOLS:
        directory = REPORT / "incremental/p0" / protocol
        paths = sorted(directory.glob("*.csv")) if directory.exists() else []
        metas = [json.loads(path.read_text()) for path in sorted(directory.glob("*.complete.json"))]
        if paths:
            part = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
            frames.append(part)
        else:
            part = pd.DataFrame()
        coverage_rows.append({
            "protocol": protocol,
            "completed_scenes": len(metas),
            "expected_scenes": 16,
            "rows": len(part),
            "lost_events": int(part.lost.sum()) if not part.empty else 0,
            "lost_scenes": int(part.loc[part.lost, "scene_token"].nunique()) if not part.empty else 0,
            "retained_events": int(part.retained.sum()) if not part.empty else 0,
            "retained_scenes": int(part.loc[part.retained, "scene_token"].nunique()) if not part.empty else 0,
            "history_sensitive_lost_events": int(part.history_sensitive_lost.sum()) if not part.empty else 0,
            "history_sensitive_lost_scenes": int(
                part.loc[part.history_sensitive_lost, "scene_token"].nunique()
            ) if not part.empty else 0,
            "complete": len(metas) == 16,
        })
    if not frames:
        raise RuntimeError("no P0 checkpoints")
    data = pd.concat(frames, ignore_index=True).sort_values(
        ["protocol", "scene_token", "frame_idx", "gt_token"]
    ).reset_index(drop=True)
    atomic_csv(data, REPORT / "per_gt_p0.csv")
    atomic_csv(pd.DataFrame(coverage_rows), REPORT / "p0_coverage.csv")

    summary_rows = []
    scene_rows = []
    for protocol in (*PROTOCOLS, "pooled"):
        protocol_frame = data if protocol == "pooled" else data[data.protocol == protocol]
        for group in GROUPS:
            group_frame = selected(protocol_frame, group)
            row = {
                "protocol": protocol,
                "population": group,
                "event_n": len(group_frame),
                "trajectory_n": group_frame[["scene_token", "instance_token"]].drop_duplicates().shape[0],
                "scene_n": group_frame.scene_token.nunique(),
                "AC_eligibility_rate": float(group_frame.eligible_AC.mean()) if len(group_frame) else math.nan,
                "BD_eligibility_rate": float(group_frame.eligible_BD.mean()) if len(group_frame) else math.nan,
                "CTEP_activation_rate": float(group_frame.ctep_active.mean()) if len(group_frame) else math.nan,
                "median_L_CTEP": float(group_frame.L_CTEP.median()) if len(group_frame) else math.nan,
            }
            for metric in METRICS:
                finite = pd.to_numeric(group_frame[metric], errors="coerce")
                row[f"median_{metric}"] = float(finite.median()) if finite.notna().any() else math.nan
            summary_rows.append(row)
        for scene, group, group_frame in (
            (scene, group, selected(scene_frame, group))
            for scene, scene_frame in protocol_frame.groupby("scene_token", sort=True)
            for group in GROUPS
        ):
            if group_frame.empty:
                continue
            row = {"protocol": protocol, "scene_token": scene, "population": group,
                   "event_n": len(group_frame), "trajectory_n": group_frame.instance_token.nunique()}
            for metric in METRICS:
                row[f"trajectory_median_mean_{metric}"] = scene_aggregates(group_frame, metric).iloc[0]
            scene_rows.append(row)
    atomic_csv(pd.DataFrame(summary_rows), REPORT / "p0_summary.csv")
    atomic_csv(pd.DataFrame(scene_rows), REPORT / "per_scene_p0.csv")

    ci_rows = []
    seed_index = 0
    for protocol in (*PROTOCOLS, "pooled"):
        protocol_frame = data if protocol == "pooled" else data[data.protocol == protocol]
        for group in GROUPS:
            for metric in METRICS:
                result = ci(selected(protocol_frame, group), metric, SEED + seed_index)
                seed_index += 1
                ci_rows.append({
                    "category": "scene_mean_of_trajectory_medians",
                    "protocol": protocol,
                    "population": group,
                    "metric": metric,
                    "cluster": "scene_bootstrap_on_trajectory_aggregates",
                    **result,
                })
        for left, right in (("lost", "retained"),
                            ("history_sensitive_lost", "retained")):
            for metric in METRICS:
                result = contrast(protocol_frame, left, right, metric, SEED + seed_index)
                seed_index += 1
                ci_rows.append({
                    "category": "paired_scene_population_contrast",
                    "protocol": protocol,
                    "population": f"{left}_minus_{right}",
                    "metric": metric,
                    "cluster": "scene_bootstrap_on_trajectory_aggregates",
                    **result,
                })
    cis = pd.DataFrame(ci_rows)
    atomic_csv(cis, REPORT / "p0_cluster_bootstrap_ci.csv")

    complete_coverage = all(row["complete"] for row in coverage_rows)
    minimum_coverage = all(
        row["lost_events"] >= 20 and row["lost_scenes"] >= 6
        and row["retained_events"] >= 50 and row["retained_scenes"] >= 8
        and row["history_sensitive_lost_events"] >= 10
        and row["history_sensitive_lost_scenes"] >= 4
        for row in coverage_rows
    )
    index = {
        (row.category, row.protocol, row.population, row.metric): row
        for row in cis.itertuples(index=False)
    }
    lost_positive = {
        protocol: {
            metric: index[("scene_mean_of_trajectory_medians", protocol, "lost", metric)].ci_low > 0
            for metric in METRICS
        } for protocol in PROTOCOLS
    }
    lost_contrast = {
        protocol: {
            metric: index[("paired_scene_population_contrast", protocol,
                           "lost_minus_retained", metric)].ci_low > 0
            for metric in METRICS
        } for protocol in PROTOCOLS
    }
    sensitive_contrast = {
        protocol: {
            metric: index[("paired_scene_population_contrast", protocol,
                           "history_sensitive_lost_minus_retained", metric)].ci_low > 0
            for metric in METRICS
        } for protocol in PROTOCOLS
    }
    mechanism = bool(
        complete_coverage and minimum_coverage
        and all(all(value.values()) for value in lost_positive.values())
        and all(all(value.values()) for value in lost_contrast.values())
        and all(all(value.values()) for value in sensitive_contrast.values())
    )
    if not complete_coverage:
        verdict = "PARTIAL_INSUFFICIENT_COVERAGE"
    elif not minimum_coverage:
        verdict = "NO_GO_CTEP_P0_INSUFFICIENT_POPULATION"
    elif mechanism:
        verdict = "P0_GO_P1_REQUIRED"
    else:
        verdict = "NO_GO_CTEP_TRAIN_MECHANISM_NOT_REPRODUCED"
    decision = {
        "verdict": verdict,
        "complete_coverage": complete_coverage,
        "minimum_population_coverage": minimum_coverage,
        "lost_gap_positive_each_protocol": lost_positive,
        "lost_minus_retained_positive_each_protocol": lost_contrast,
        "history_sensitive_lost_minus_retained_positive_each_protocol": sensitive_contrast,
        "train_mechanism_reproduced": mechanism,
        "bootstrap": {"replicates": BOOTSTRAPS, "base_seed": SEED,
                      "unit": "scene after instance-trajectory aggregation"},
    }
    atomic_json(decision, REPORT / "p0_decision.json")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
