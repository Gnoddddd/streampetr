"""Pure statistics and gate helpers for the BD temporal-support audit."""

from __future__ import annotations

import numpy as np


def _trajectory_arrays(rows, metric: str, group: str) -> dict[str, list[np.ndarray]]:
    selected = rows[(rows["population"] == group) & np.isfinite(rows[metric])]
    output: dict[str, list[np.ndarray]] = {}
    for scene, scene_rows in selected.groupby("scene_token", sort=False):
        values = []
        for _, trajectory in scene_rows.groupby("instance_token", sort=False):
            value = float(np.median(trajectory[metric].to_numpy(float)))
            if np.isfinite(value):
                values.append(value)
        if values:
            output[str(scene)] = [np.asarray(values, dtype=np.float64)]
    return output


def two_stage_cluster_bootstrap(rows, metric: str, group: str, *,
                                n_boot: int = 5000, seed: int = 626262) -> dict:
    """Median CI with scene resampling and nested trajectory resampling."""

    grouped = _trajectory_arrays(rows, metric, group)
    scenes = list(grouped)
    all_values = np.concatenate([grouped[scene][0] for scene in scenes]) if scenes else np.array([])
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=np.float64)
    for bootstrap in range(n_boot):
        sampled_scenes = rng.choice(scenes, len(scenes), replace=True)
        pieces = []
        for scene in sampled_scenes:
            values = grouped[str(scene)][0]
            pieces.append(rng.choice(values, len(values), replace=True))
        estimates[bootstrap] = np.median(np.concatenate(pieces)) if pieces else np.nan
    finite = estimates[np.isfinite(estimates)]
    return {
        "estimate": float(np.median(all_values)) if len(all_values) else float("nan"),
        "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else float("nan"),
        "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else float("nan"),
        "n_gt": int(np.isfinite(rows.loc[rows.population == group, metric]).sum()),
        "n_trajectories": int(len(all_values)),
        "n_scenes": int(len(scenes)),
    }


def two_group_cluster_contrast(rows, metric: str, *, n_boot: int = 5000,
                               seed: int = 626262) -> dict:
    """Lost-minus-retained median contrast under a shared two-stage bootstrap."""

    lost = _trajectory_arrays(rows, metric, "lost")
    retained = _trajectory_arrays(rows, metric, "retained")
    scenes = sorted(set(lost) | set(retained))
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=np.float64)
    for bootstrap in range(n_boot):
        sampled_scenes = rng.choice(scenes, len(scenes), replace=True)
        values = {"lost": [], "retained": []}
        for scene in sampled_scenes:
            for name, source in (("lost", lost), ("retained", retained)):
                if str(scene) not in source:
                    continue
                array = source[str(scene)][0]
                values[name].append(rng.choice(array, len(array), replace=True))
        if values["lost"] and values["retained"]:
            estimates[bootstrap] = (
                np.median(np.concatenate(values["lost"]))
                - np.median(np.concatenate(values["retained"]))
            )
        else:
            estimates[bootstrap] = np.nan
    finite = estimates[np.isfinite(estimates)]
    lost_values = np.concatenate([value[0] for value in lost.values()]) if lost else np.array([])
    retained_values = np.concatenate([value[0] for value in retained.values()]) if retained else np.array([])
    estimate = (float(np.median(lost_values) - np.median(retained_values))
                if len(lost_values) and len(retained_values) else float("nan"))
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else float("nan"),
        "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else float("nan"),
        "n_boot_finite": int(len(finite)),
    }


def classify_protocol(gb: dict, gd: dict, gb_contrast: dict, gd_contrast: dict,
                      equivalence: float = 0.01) -> dict:
    """Apply the preregistered compensation/contamination pattern gate."""

    gb_positive = float(gb["ci_low"]) > 0.0
    gd_negative = float(gd["ci_high"]) < 0.0
    gb_zero = float(gb["ci_low"]) >= -equivalence and float(gb["ci_high"]) <= equivalence
    gd_zero = float(gd["ci_low"]) >= -equivalence and float(gd["ci_high"]) <= equivalence
    compensation = gb_positive and gd_zero and float(gb_contrast["ci_low"]) > 0.0
    contamination = gb_zero and gd_negative and float(gd_contrast["ci_high"]) < 0.0
    both = (gb_positive and gd_negative and float(gb_contrast["ci_low"]) > 0.0
            and float(gd_contrast["ci_high"]) < 0.0)
    if both:
        mechanism = "both"
    elif compensation:
        mechanism = "clean_history_compensation"
    elif contamination:
        mechanism = "fault_history_contamination"
    else:
        mechanism = "unexplained"
    return {
        "mechanism": mechanism,
        "pattern_pass": mechanism != "unexplained",
        "gb_positive": gb_positive,
        "gd_negative": gd_negative,
        "gb_equivalent_zero": gb_zero,
        "gd_equivalent_zero": gd_zero,
        "gb_lost_enriched": float(gb_contrast["ci_low"]) > 0.0,
        "gd_lost_enriched_negative": float(gd_contrast["ci_high"]) < 0.0,
    }
