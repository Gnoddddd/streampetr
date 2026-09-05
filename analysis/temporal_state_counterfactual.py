"""Pure statistics and decision helpers for the temporal 2x2 audit."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

from analysis.fault_boundary_root_cause import spearman


def _clusters(rows: Iterable[dict], cluster_fields: Sequence[str]):
    output = defaultdict(list)
    for row in rows:
        output[tuple(str(row[field]) for field in cluster_fields)].append(row)
    return list(output.values())


def cluster_bootstrap_median(rows: Iterable[dict], value_key: str,
                             cluster_fields: Sequence[str], seed: int,
                             iterations: int = 5000) -> dict:
    rows = list(rows)
    values = np.asarray([float(row[value_key]) for row in rows], float)
    values = values[np.isfinite(values)]
    groups = _clusters(rows, cluster_fields)
    if not len(values) or not groups:
        return {"estimate": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "iterations": int(iterations)}
    rng = np.random.default_rng(int(seed))
    estimates = []
    for _ in range(int(iterations)):
        sampled = []
        for index in rng.integers(0, len(groups), len(groups)):
            sampled.extend(float(row[value_key]) for row in groups[index]
                           if np.isfinite(float(row[value_key])))
        if sampled:
            estimates.append(np.median(sampled))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return {"estimate": float(np.median(values)), "ci_low": float(low),
            "ci_high": float(high), "iterations": int(iterations)}


def cluster_bootstrap_spearman(rows: Iterable[dict], x_key: str, y_key: str,
                               cluster_fields: Sequence[str], seed: int,
                               iterations: int = 5000) -> dict:
    rows = list(rows)
    groups = _clusters(rows, cluster_fields)
    x = np.asarray([float(row[x_key]) for row in rows], float)
    y = np.asarray([float(row[y_key]) for row in rows], float)
    estimate = spearman(x, y)
    if not groups:
        return {"estimate": estimate, "ci_low": float("nan"),
                "ci_high": float("nan"), "iterations": int(iterations)}
    rng, estimates = np.random.default_rng(int(seed)), []
    for _ in range(int(iterations)):
        sampled = []
        for index in rng.integers(0, len(groups), len(groups)):
            sampled.extend(groups[index])
        sx = np.asarray([float(row[x_key]) for row in sampled], float)
        sy = np.asarray([float(row[y_key]) for row in sampled], float)
        value = spearman(sx, sy)
        if np.isfinite(value):
            estimates.append(value)
    if estimates:
        low, high = np.percentile(estimates, [2.5, 97.5])
    else:
        low = high = float("nan")
    return {"estimate": estimate, "ci_low": float(low), "ci_high": float(high),
            "iterations": int(iterations)}


def temporal_decision(bd: dict, ca: dict, protocol_medians: dict,
                      age_protocol_rho: dict, age_pooled: dict,
                      equivalence: dict) -> dict:
    bd_cross = all(float(protocol_medians[p]["bd"]) > 0.0 for p in protocol_medians)
    ca_cross = all(float(protocol_medians[p]["ca"]) < 0.0 for p in protocol_medians)
    bd_age = (all(np.isfinite(float(age_protocol_rho[p]["bd"]))
                  and float(age_protocol_rho[p]["bd"]) > 0.0
                  for p in age_protocol_rho)
              and float(age_pooled["bd_ci_low"]) > 0.0)
    ca_age = (all(np.isfinite(float(age_protocol_rho[p]["ca"]))
                  and float(age_protocol_rho[p]["ca"]) < 0.0
                  for p in age_protocol_rho)
              and float(age_pooled["ca_ci_high"]) < 0.0)
    bd_arm = bool(
        float(bd["median"]) >= 0.01 and float(bd["ci_low"]) > 0.0
        and (float(bd["topk_event_rate"]) >= 0.2
             or float(bd["tp_event_rate"]) >= 0.2)
        and bd_cross and bd_age
    )
    ca_arm = bool(
        float(ca["median"]) <= -0.01 and float(ca["ci_high"]) < 0.0
        and (float(ca["topk_event_rate"]) >= 0.2
             or float(ca["tp_event_rate"]) >= 0.2)
        and ca_cross and ca_age
    )
    contamination = bd_arm or ca_arm
    equivalent = bool(
        float(equivalence["bd_ci_low"]) >= -0.01
        and float(equivalence["bd_ci_high"]) <= 0.01
        and float(equivalence["ca_ci_low"]) >= -0.01
        and float(equivalence["ca_ci_high"]) <= 0.01
        and float(equivalence["bd_topk_discordance"]) <= 0.1
        and float(equivalence["bd_tp_discordance"]) <= 0.1
        and float(equivalence["ca_topk_discordance"]) <= 0.1
        and float(equivalence["ca_tp_discordance"]) <= 0.1
        and all(abs(float(protocol_medians[p]["bd"])) <= 0.01
                and abs(float(protocol_medians[p]["ca"])) <= 0.01
                for p in protocol_medians)
    )
    if contamination:
        decision = "GO_TEMPORAL_STATE_CONTAMINATION"
    elif equivalent:
        decision = "NO_GO_CURRENT_OBSERVATION_DOMINANT_CLOSE_STAGE4"
    else:
        decision = "NO_GO_TEMPORAL_UNSUPPORTED_CLOSE_STAGE4"
    return {"decision": decision, "bd_arm": bd_arm, "ca_arm": ca_arm,
            "bd_cross_protocol": bd_cross, "ca_cross_protocol": ca_cross,
            "bd_age_stable": bd_age, "ca_age_stable": ca_age,
            "temporal_contamination": contamination,
            "current_state_equivalent": equivalent}
