"""Pure helpers for CARE-3D P2-A online query association.

P2-A replaces the oracle ``target_clean_query_index`` used by P1 with a
strictly online association between the currently tracked clean object at time
``t`` and the 900 fault-frame decoder queries at ``t+1``.  The oracle query is
accepted only by evaluation helpers and never by the association cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch.nn import functional as F


PROTOCOLS = ("blur_back", "crash_back", "dark_back")
QUERY_COUNT = 900
NUM_CURRENT_QUERIES = 644
NUM_PROPAGATED_QUERIES = 256
MAX_GEOMETRY_DISTANCE_M = 12.0

# P1 already excludes shared target-query rows.  P2-A additionally excludes
# shared anchor-query rows because two online tracks cannot be represented by
# the same detector query in a one-to-one association experiment.
P2A_QUERY_COLLISION_POLICY = (
    "exclude_all_rows_in_shared_anchor_or_target_query_frame"
)

WEIGHT_GRID: Tuple[Tuple[float, float, float], ...] = (
    (0.5, 0.3, 0.2),
    (0.4, 0.4, 0.2),
    (0.6, 0.2, 0.2),
    (0.4, 0.3, 0.3),
    (0.5, 0.2, 0.3),
)
MAX_COST_GRID: Tuple[float, ...] = (0.35, 0.45, 0.55)
BASELINE_WEIGHTS: Mapping[str, Tuple[float, float, float]] = {
    "geometry_only": (1.0, 0.0, 0.0),
    "embedding_only": (0.0, 1.0, 0.0),
    "class_geometry": (0.5, 0.0, 0.5),
}


@dataclass(frozen=True)
class AssociationConfig:
    geo_weight: float
    embedding_weight: float
    class_weight: float
    max_cost: float

    @property
    def weights(self) -> Tuple[float, float, float]:
        return (self.geo_weight, self.embedding_weight, self.class_weight)

    @property
    def config_id(self) -> str:
        return (
            f"g{self.geo_weight:.1f}_e{self.embedding_weight:.1f}_"
            f"c{self.class_weight:.1f}_t{self.max_cost:.2f}"
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "config_id": self.config_id,
            "geo_weight": float(self.geo_weight),
            "embedding_weight": float(self.embedding_weight),
            "class_weight": float(self.class_weight),
            "max_cost": float(self.max_cost),
        }


def association_grid() -> Tuple[AssociationConfig, ...]:
    configs = []
    for weights in WEIGHT_GRID:
        if not np.isclose(sum(weights), 1.0):
            raise RuntimeError(f"P2-A association weights do not sum to one: {weights}")
        for threshold in MAX_COST_GRID:
            configs.append(AssociationConfig(*weights, threshold))
    ids = [config.config_id for config in configs]
    if len(ids) != len(set(ids)) or len(configs) != 15:
        raise RuntimeError("P2-A association config grid changed")
    return tuple(configs)


def baseline_configs(max_cost: float) -> Tuple[Tuple[str, AssociationConfig], ...]:
    output = []
    for name, weights in BASELINE_WEIGHTS.items():
        output.append((name, AssociationConfig(*weights, float(max_cost))))
    return tuple(output)


def assert_query_layout(num_query: int, num_propagated: int, total_queries: int) -> None:
    if int(num_query) != NUM_CURRENT_QUERIES:
        raise RuntimeError(f"StreamPETR current-query count changed: {num_query}")
    if int(num_propagated) != NUM_PROPAGATED_QUERIES:
        raise RuntimeError(f"StreamPETR propagated-query count changed: {num_propagated}")
    if int(total_queries) != QUERY_COUNT:
        raise RuntimeError(f"StreamPETR total-query count changed: {total_queries}")


def _collision_mask(
    first: Sequence[int],
    second: Sequence[int],
) -> Tuple[np.ndarray, int]:
    first = np.asarray(first, dtype=np.int64)
    second = np.asarray(second, dtype=np.int64)
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise ValueError("collision keys must be aligned 1-D arrays")
    counts: Dict[Tuple[int, int], int] = {}
    for a, b in zip(first.tolist(), second.tolist()):
        key = (int(a), int(b))
        counts[key] = counts.get(key, 0) + 1
    multiplicity = np.asarray(
        [counts[(int(a), int(b))] for a, b in zip(first, second)],
        dtype=np.int64,
    )
    groups = int(sum(value > 1 for value in counts.values()))
    return multiplicity == 1, groups


def p2a_query_eligibility(frame: pd.DataFrame) -> Dict[str, object]:
    required = {
        "anchor_frame_idx",
        "target_frame_idx",
        "anchor_query_index",
        "target_clean_query_index",
    }
    if not required <= set(frame.columns):
        raise RuntimeError(f"P2-A metadata missing {sorted(required - set(frame.columns))}")

    anchor_ok, anchor_groups = _collision_mask(
        frame.anchor_frame_idx.to_numpy(dtype=int),
        frame.anchor_query_index.to_numpy(dtype=int),
    )
    target_ok, target_groups = _collision_mask(
        frame.target_frame_idx.to_numpy(dtype=int),
        frame.target_clean_query_index.to_numpy(dtype=int),
    )
    eligible = anchor_ok & target_ok
    return {
        "eligible": eligible,
        "p1_rows_total": int(len(frame)),
        "p2a_eligible_rows": int(eligible.sum()),
        "anchor_query_collision_excluded_rows": int((~anchor_ok).sum()),
        "anchor_query_collision_groups": int(anchor_groups),
        "target_query_collision_excluded_rows": int((~target_ok).sum()),
        "target_query_collision_groups": int(target_groups),
        "total_excluded_rows": int((~eligible).sum()),
        "policy": P2A_QUERY_COLLISION_POLICY,
    }


def filter_p2a_rows(
    frame: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, object]]:
    audit = p2a_query_eligibility(frame)
    mask = np.asarray(audit["eligible"], dtype=bool)
    raw_n = len(frame)
    filtered = frame.loc[mask].reset_index(drop=True).copy()
    filtered["p2a_query_collision_policy"] = P2A_QUERY_COLLISION_POLICY
    output: Dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.ndim > 0 and len(array) == raw_n:
            output[key] = array[mask].copy()
        else:
            output[key] = array.copy()
    for frame_idx, group in filtered.groupby("target_frame_idx", sort=False):
        anchors = group.anchor_query_index.to_numpy(dtype=int)
        targets = group.target_clean_query_index.to_numpy(dtype=int)
        if len(anchors) != len(set(anchors.tolist())):
            raise RuntimeError(f"P2-A anchor collision survived frame {frame_idx}")
        if len(targets) != len(set(targets.tolist())):
            raise RuntimeError(f"P2-A target collision survived frame {frame_idx}")
    return filtered, output, audit


def transform_lidar_centers_between_frames(
    centers_lidar: np.ndarray,
    source_context: Mapping[str, np.ndarray],
    target_context: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Transform source-frame predicted lidar centers into target lidar frame."""
    centers = np.asarray(centers_lidar, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers_lidar must be [N,3]")

    src_l2e_r = np.asarray(source_context["lidar2ego_rotation"], dtype=np.float64)
    src_l2e_t = np.asarray(source_context["lidar2ego_translation"], dtype=np.float64)
    src_e2g_r = np.asarray(source_context["ego2global_rotation"], dtype=np.float64)
    src_e2g_t = np.asarray(source_context["ego2global_translation"], dtype=np.float64)
    dst_l2e_r = np.asarray(target_context["lidar2ego_rotation"], dtype=np.float64)
    dst_l2e_t = np.asarray(target_context["lidar2ego_translation"], dtype=np.float64)
    dst_e2g_r = np.asarray(target_context["ego2global_rotation"], dtype=np.float64)
    dst_e2g_t = np.asarray(target_context["ego2global_translation"], dtype=np.float64)

    source_ego = centers @ src_l2e_r.T + src_l2e_t
    global_xyz = source_ego @ src_e2g_r.T + src_e2g_t
    target_ego = (global_xyz - dst_e2g_t) @ dst_e2g_r
    target_lidar = (target_ego - dst_l2e_t) @ dst_l2e_r
    if not np.isfinite(target_lidar).all():
        raise RuntimeError("P2-A pose transform produced non-finite centers")
    return target_lidar.astype(np.float32)


def association_cost_components(
    anchor_features: Tensor,
    anchor_centers_target_lidar: Tensor,
    anchor_classes: Tensor,
    fault_query_features: Tensor,
    fault_logits: Tensor,
    fault_centers_lidar: Tensor,
    *,
    max_geometry_distance_m: float = MAX_GEOMETRY_DISTANCE_M,
) -> Dict[str, Tensor]:
    """Build the fully vectorized online association component matrices.

    The function deliberately has no oracle-query, GT, clean-future, or outcome
    argument.  Shapes are ``N`` current tracks by ``Q=900`` fault queries.
    """
    if anchor_features.ndim != 2 or fault_query_features.ndim != 2:
        raise ValueError("association features must be 2-D")
    if anchor_features.shape[1] != fault_query_features.shape[1]:
        raise ValueError("association feature dimensions differ")
    if anchor_centers_target_lidar.shape != (anchor_features.shape[0], 3):
        raise ValueError("anchor centers must be [N,3]")
    if fault_centers_lidar.ndim != 2 or fault_centers_lidar.shape[1] != 3:
        raise ValueError("fault centers must be [Q,3]")
    if fault_logits.ndim != 2 or fault_logits.shape[0] != fault_query_features.shape[0]:
        raise ValueError("fault logits/query layout mismatch")
    if anchor_classes.ndim != 1 or anchor_classes.shape[0] != anchor_features.shape[0]:
        raise ValueError("anchor classes must be [N]")
    if fault_query_features.shape[0] != QUERY_COUNT:
        raise RuntimeError(f"P2-A expects {QUERY_COUNT} fault queries")
    if torch.any(anchor_classes < 0) or torch.any(anchor_classes >= fault_logits.shape[1]):
        raise ValueError("anchor class index out of range")

    xy_delta = (
        anchor_centers_target_lidar[:, None, :2]
        - fault_centers_lidar[None, :, :2]
    )
    distance = torch.linalg.vector_norm(xy_delta, dim=-1)
    geo = torch.clamp(distance / float(max_geometry_distance_m), min=0.0, max=1.0)

    anchor_norm = F.normalize(anchor_features.float(), p=2, dim=-1, eps=1e-8)
    fault_norm = F.normalize(fault_query_features.float(), p=2, dim=-1, eps=1e-8)
    cosine = anchor_norm @ fault_norm.transpose(0, 1)
    embedding = torch.clamp((1.0 - cosine) * 0.5, min=0.0, max=1.0)

    probabilities = fault_logits.float().sigmoid()
    class_probability = probabilities[:, anchor_classes.long()].transpose(0, 1).contiguous()
    class_cost = 1.0 - class_probability

    geometry_allowed = distance <= float(max_geometry_distance_m)
    for name, value in {
        "geometry": geo,
        "embedding": embedding,
        "class": class_cost,
        "distance_m": distance,
    }.items():
        if value.shape != (anchor_features.shape[0], QUERY_COUNT):
            raise RuntimeError(f"P2-A {name} cost shape changed")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"P2-A {name} cost contains non-finite values")
    return {
        "geometry": geo,
        "embedding": embedding,
        "class": class_cost,
        "distance_m": distance,
        "geometry_allowed": geometry_allowed,
    }


def weighted_cost(
    components: Mapping[str, Tensor],
    config: AssociationConfig,
) -> Tensor:
    weights = config.weights
    if any(value < 0.0 for value in weights) or not np.isclose(sum(weights), 1.0):
        raise ValueError(f"invalid association weights: {weights}")
    cost = (
        float(config.geo_weight) * components["geometry"]
        + float(config.embedding_weight) * components["embedding"]
        + float(config.class_weight) * components["class"]
    )
    return torch.where(
        components["geometry_allowed"],
        cost,
        torch.full_like(cost, float("inf")),
    )


def hungarian_with_unmatched(cost: Tensor, max_cost: float) -> Dict[str, np.ndarray]:
    """One-to-one Hungarian assignment with one private dummy per track."""
    if cost.ndim != 2:
        raise ValueError("cost must be [N,Q]")
    n, q = int(cost.shape[0]), int(cost.shape[1])
    if q != QUERY_COUNT:
        raise RuntimeError(f"P2-A Hungarian candidate count changed: {q}")
    if n == 0:
        return {
            "selected_query": np.empty((0,), dtype=np.int64),
            "selected_cost": np.empty((0,), dtype=np.float64),
            "matched": np.empty((0,), dtype=bool),
        }
    if not 0.0 < float(max_cost) <= 1.0:
        raise ValueError("max_cost must be in (0,1]")

    values = cost.detach().float().cpu().numpy().astype(np.float64, copy=False)
    feasible = np.isfinite(values) & (values <= float(max_cost))
    large = 1e6
    augmented = np.full((n, q + n), large, dtype=np.float64)
    augmented[:, :q] = np.where(feasible, values, large)
    dummy_cost = float(max_cost) + 1e-6
    augmented[np.arange(n), q + np.arange(n)] = dummy_cost
    rows, columns = linear_sum_assignment(augmented)
    if len(rows) != n or set(rows.tolist()) != set(range(n)):
        raise RuntimeError("P2-A Hungarian did not assign every track")

    selected_query = np.full(n, -1, dtype=np.int64)
    selected_cost = np.full(n, np.nan, dtype=np.float64)
    for row, column in zip(rows.tolist(), columns.tolist()):
        if column < q and feasible[row, column]:
            selected_query[row] = int(column)
            selected_cost[row] = float(values[row, column])
    matched = selected_query >= 0
    real = selected_query[matched]
    if len(real) != len(set(real.tolist())):
        raise RuntimeError("P2-A Hungarian produced duplicate real-query assignments")
    return {
        "selected_query": selected_query,
        "selected_cost": selected_cost,
        "matched": matched,
    }


def oracle_diagnostics(cost: Tensor, oracle_queries: Sequence[int]) -> Dict[str, np.ndarray]:
    """Offline diagnostics only; never called by association forward/cost."""
    values = cost.detach().float().cpu().numpy().astype(np.float64, copy=False)
    oracle = np.asarray(oracle_queries, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] != len(oracle):
        raise ValueError("oracle query vector/cost rows differ")
    if np.any(oracle < 0) or np.any(oracle >= values.shape[1]):
        raise ValueError("oracle query index out of range")

    oracle_cost = np.full(len(oracle), np.nan, dtype=np.float64)
    oracle_rank = np.full(len(oracle), values.shape[1] + 1, dtype=np.int64)
    margin = np.full(len(oracle), np.nan, dtype=np.float64)
    eligible = np.zeros(len(oracle), dtype=bool)
    query_ids = np.arange(values.shape[1], dtype=np.int64)
    for index, query in enumerate(oracle.tolist()):
        row = values[index]
        if not np.isfinite(row[query]):
            continue
        eligible[index] = True
        oracle_cost[index] = float(row[query])
        sortable = np.where(np.isfinite(row), row, np.inf)
        order = np.lexsort((query_ids, sortable))
        location = np.flatnonzero(order == query)
        if location.size != 1:
            raise RuntimeError("oracle query rank is not unique")
        oracle_rank[index] = int(location[0]) + 1
        wrong = sortable.copy()
        wrong[query] = np.inf
        best_wrong = float(np.min(wrong))
        if np.isfinite(best_wrong):
            margin[index] = best_wrong - float(row[query])
    return {
        "oracle_cost": oracle_cost,
        "oracle_rank": oracle_rank,
        "correct_vs_best_wrong_margin": margin,
        "oracle_geometry_eligible": eligible,
    }


def assignment_rows(
    assignment: Mapping[str, np.ndarray],
    oracle_queries: Sequence[int],
    diagnostics: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    oracle = np.asarray(oracle_queries, dtype=np.int64)
    selected = np.asarray(assignment["selected_query"], dtype=np.int64)
    matched = np.asarray(assignment["matched"], dtype=bool)
    if oracle.shape != selected.shape:
        raise ValueError("oracle and assignment lengths differ")
    exact = matched & (selected == oracle)
    wrong = matched & (selected != oracle)
    unmatched = ~matched
    if not np.all(exact.astype(int) + wrong.astype(int) + unmatched.astype(int) == 1):
        raise RuntimeError("P2-A assignment outcomes are not exhaustive")
    return {
        "selected_query": selected,
        "selected_cost": np.asarray(assignment["selected_cost"], dtype=np.float64),
        "exact_match": exact.astype(np.int8),
        "wrong_match": wrong.astype(np.int8),
        "unmatched": unmatched.astype(np.int8),
        "oracle_cost": np.asarray(diagnostics["oracle_cost"], dtype=np.float64),
        "oracle_rank": np.asarray(diagnostics["oracle_rank"], dtype=np.int64),
        "correct_vs_best_wrong_margin": np.asarray(
            diagnostics["correct_vs_best_wrong_margin"], dtype=np.float64
        ),
        "oracle_geometry_eligible": np.asarray(
            diagnostics["oracle_geometry_eligible"], dtype=bool
        ).astype(np.int8),
    }


def aggregate_summary(frame: pd.DataFrame) -> Dict[str, float]:
    total = int(len(frame))
    if total == 0:
        return {
            "rows": 0,
            "exact_n": 0,
            "wrong_n": 0,
            "unmatched_n": 0,
            "accepted_n": 0,
            "accepted_cost_sum": 0.0,
            "exact_recall": float("nan"),
            "wrong_match_rate": float("nan"),
            "unmatched_rate": float("nan"),
            "mean_accepted_cost": float("nan"),
        }
    selected = frame.selected_cost.to_numpy(dtype=float)
    accepted = np.isfinite(selected)
    return {
        "rows": total,
        "exact_n": int(frame.exact_match.astype(int).sum()),
        "wrong_n": int(frame.wrong_match.astype(int).sum()),
        "unmatched_n": int(frame.unmatched.astype(int).sum()),
        "accepted_n": int(accepted.sum()),
        "accepted_cost_sum": float(np.nansum(selected)),
        "exact_recall": float(frame.exact_match.astype(float).mean()),
        "wrong_match_rate": float(frame.wrong_match.astype(float).mean()),
        "unmatched_rate": float(frame.unmatched.astype(float).mean()),
        "mean_accepted_cost": float(np.nanmean(selected)) if accepted.any() else float("nan"),
    }


def select_global_config(train_summary: pd.DataFrame) -> Dict[str, object]:
    """Frozen train-only selection with deterministic preregistered tie-breaks."""
    required = {
        "protocol", "config_id", "rows", "exact_n", "wrong_n",
        "accepted_n", "accepted_cost_sum",
    }
    if not required <= set(train_summary.columns):
        raise RuntimeError(f"P2-A train summary missing {sorted(required - set(train_summary.columns))}")
    expected_ids = {config.config_id for config in association_grid()}
    observed_ids = set(train_summary.config_id.astype(str))
    if observed_ids != expected_ids:
        raise RuntimeError("P2-A train summary does not contain the frozen 15-config grid")

    candidate_rows = []
    protocol_rows = []
    for config in association_grid():
        per_protocol = []
        subset = train_summary[train_summary.config_id.astype(str) == config.config_id]
        for protocol in PROTOCOLS:
            group = subset[subset.protocol.astype(str) == protocol]
            rows = int(group.rows.astype(int).sum())
            if rows <= 0:
                raise RuntimeError(
                    f"empty P2-A train cohort config={config.config_id} protocol={protocol}"
                )
            exact = int(group.exact_n.astype(int).sum()) / rows
            wrong = int(group.wrong_n.astype(int).sum()) / rows
            accepted_n = int(group.accepted_n.astype(int).sum())
            accepted_cost = float(group.accepted_cost_sum.astype(float).sum())
            mean_cost = accepted_cost / accepted_n if accepted_n else float("inf")
            per_protocol.append((exact, wrong, mean_cost))
            protocol_rows.append({
                "config_id": config.config_id,
                "protocol": protocol,
                "rows": rows,
                "exact_recall": exact,
                "wrong_match_rate": wrong,
                "mean_accepted_cost": mean_cost,
            })
        macro_recall = float(np.mean([value[0] for value in per_protocol]))
        macro_wrong = float(np.mean([value[1] for value in per_protocol]))
        macro_cost = float(np.mean([value[2] for value in per_protocol]))
        candidate_rows.append({
            **config.as_dict(),
            "macro_exact_recall": macro_recall,
            "macro_wrong_match_rate": macro_wrong,
            "macro_mean_accepted_cost": macro_cost,
        })

    ranking = sorted(
        candidate_rows,
        key=lambda row: (
            -row["macro_exact_recall"],
            row["macro_wrong_match_rate"],
            row["macro_mean_accepted_cost"],
            row["config_id"],
        ),
    )
    winner = dict(ranking[0])
    return {
        "selected": winner,
        "ranking": ranking,
        "per_protocol": protocol_rows,
        "selection_rule": (
            "maximize_train_protocol_macro_exact_recall_then_minimize_macro_wrong_"
            "then_minimize_macro_accepted_cost_then_config_id_lexicographic"
        ),
    }


def p2a_protocol_gate(
    point: Mapping[str, float],
    scene_ci: Mapping[str, float],
    instance_ci: Mapping[str, float],
    *,
    min_exact_recall: float,
    min_cluster_ci_low: float,
    max_wrong_match_rate: float,
) -> Dict[str, bool]:
    recall_pass = bool(float(point["exact_recall"]) >= float(min_exact_recall))
    scene_pass = bool(float(scene_ci["ci_low"]) > float(min_cluster_ci_low))
    instance_pass = bool(float(instance_ci["ci_low"]) > float(min_cluster_ci_low))
    wrong_pass = bool(float(point["wrong_match_rate"]) <= float(max_wrong_match_rate))
    return {
        "exact_recall_pass": recall_pass,
        "scene_cluster_pass": scene_pass,
        "instance_cluster_pass": instance_pass,
        "wrong_match_pass": wrong_pass,
        "protocol_pass": bool(recall_pass and scene_pass and instance_pass and wrong_pass),
    }
