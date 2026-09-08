"""Deployment-valid relative candidate features for CARE-3D P2-A2-R1-F0.

The feature path has no oracle, GT, clean-future, or protocol input.  Labels
are attached only by :func:`offline_disagreement_labels` after feature values
and candidate identities have been frozen.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from analysis.care3d_p2a_association import QUERY_COUNT


EVIDENCE_FEATURE_COLUMNS = (
    "A_geometry", "L_geometry", "delta_geometry",
    "A_embedding", "L_embedding", "delta_embedding",
    "A_class_cost", "L_class_cost", "delta_class_cost",
    "A_total_cost", "L_total_cost", "delta_total_cost",
    "A_distance_m", "L_distance_m", "delta_distance_m",
    "A_anchor_class_probability", "L_anchor_class_probability",
    "delta_anchor_class_probability",
    "A_top1_probability", "L_top1_probability", "delta_top1_probability",
    "A_predicted_class", "L_predicted_class",
    "A_class_matches_anchor", "L_class_matches_anchor",
    "A_row_rank", "L_row_rank", "delta_row_rank",
    "A_row_margin", "L_row_margin", "delta_row_margin",
    "A_column_margin", "L_column_margin", "delta_column_margin",
    "A_mutual", "L_mutual",
    "anchor_is_propagated", "p2a0_is_propagated",
    "lineage_position_norm", "target_frame_norm",
)

CHEAP_BASELINE_DECISIVE_AUROC = {
    "blur_back": 0.756640,
    "crash_back": 0.732855,
    "dark_back": 0.745944,
}
COLUMN_NO_COMPETITOR_SENTINEL = 1.0


def finite_model_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Encode the extended-real no-column-competitor case without row loss.

    Raw exported margins retain the exact +inf/NaN results of their registered
    definitions.  Only the three column-margin inputs have a fixed, label-free
    model encoding: +inf (no second finite anchor) maps to the positive boundary
    sentinel 1, and inf-inf maps to zero difference.  Every other non-finite
    value remains an error.
    """
    matrix = frame.loc[:, EVIDENCE_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    indexes = {name: index for index, name in enumerate(EVIDENCE_FEATURE_COLUMNS)}
    allowed = {
        indexes["A_column_margin"],
        indexes["L_column_margin"],
        indexes["delta_column_margin"],
    }
    _bad_rows, bad_columns = np.nonzero(~np.isfinite(matrix))
    if any(int(column) not in allowed for column in bad_columns.tolist()):
        names = sorted({EVIDENCE_FEATURE_COLUMNS[int(column)] for column in bad_columns})
        raise RuntimeError(f"non-finite R1-F0 evidence outside column margins: {names}")
    output = matrix.copy()
    for name in ("A_column_margin", "L_column_margin"):
        column = indexes[name]
        values = output[:, column]
        if np.isneginf(values).any() or np.isnan(values).any():
            raise RuntimeError(f"invalid no-competitor encoding input: {name}")
        values[np.isposinf(values)] = COLUMN_NO_COMPETITOR_SENTINEL
    delta = output[:, indexes["delta_column_margin"]]
    if np.isinf(delta).any():
        raise RuntimeError("one-sided infinite delta_column_margin is unsupported")
    delta[np.isnan(delta)] = 0.0
    if not np.isfinite(output).all():
        raise RuntimeError("R1-F0 fixed model matrix remains non-finite")
    return output


def _arrays(
    components: Mapping[str, Tensor],
    frozen_cost: Tensor,
) -> Dict[str, np.ndarray]:
    required = {"geometry", "embedding", "class", "distance_m", "geometry_allowed"}
    if not required <= set(components):
        raise ValueError(f"association components missing {sorted(required - set(components))}")
    if frozen_cost.ndim != 2 or int(frozen_cost.shape[1]) != QUERY_COUNT:
        raise ValueError("frozen_cost must be [N,900]")
    output = {
        key: value.detach().float().cpu().numpy()
        for key, value in components.items() if key in required
    }
    output["total"] = frozen_cost.detach().float().cpu().numpy()
    expected = tuple(frozen_cost.shape)
    if any(tuple(value.shape) != expected for value in output.values()):
        raise RuntimeError("relative-evidence component layout changed")
    return output


def _row_rank_and_margin(cost: np.ndarray, candidate: int) -> tuple[int, float]:
    candidate = int(candidate)
    finite_queries = np.flatnonzero(np.isfinite(cost))
    if candidate not in finite_queries:
        return 0, float("nan")
    order = finite_queries[np.lexsort((finite_queries, cost[finite_queries]))]
    rank = int(np.flatnonzero(order == candidate)[0]) + 1
    other = finite_queries[finite_queries != candidate]
    margin = (
        float(np.min(cost[other]) - cost[candidate])
        if other.size else float("inf")
    )
    return rank, margin


def _column_ownership(
    cost: np.ndarray,
    row: int,
    candidate: int,
) -> tuple[bool, float]:
    candidate = int(candidate)
    column = cost[:, candidate]
    current = float(column[int(row)])
    if not np.isfinite(current):
        return False, float("nan")
    finite_rows = np.flatnonzero(np.isfinite(column))
    order = finite_rows[np.lexsort((finite_rows, column[finite_rows]))]
    if not order.size:
        raise RuntimeError("finite current candidate disappeared from its column")
    mutual = int(order[0]) == int(row)
    if mutual:
        margin = (
            float(column[order[1]] - current)
            if len(order) > 1 else float("inf")
        )
    else:
        margin = float(column[order[0]] - current)
    return bool(mutual), margin


def relative_candidate_features(
    components: Mapping[str, Tensor],
    frozen_cost: Tensor,
    fault_logits: Tensor,
    anchor_classes: Tensor | Sequence[int],
    anchor_queries: Sequence[int],
    p2a0_selected_queries: Sequence[int],
    lineage_child_queries: Sequence[int],
    lineage_positions: Sequence[int],
    *,
    target_frame_idx: int,
) -> Dict[str, np.ndarray]:
    """Freeze A/L evidence for disagreement rows using online tensors only."""
    values = _arrays(components, frozen_cost)
    n = int(frozen_cost.shape[0])
    anchor_classes_np = np.asarray(
        anchor_classes.detach().cpu().numpy()
        if torch.is_tensor(anchor_classes) else anchor_classes,
        dtype=np.int64,
    )
    anchor_queries_np = np.asarray(anchor_queries, dtype=np.int64)
    selected = np.asarray(p2a0_selected_queries, dtype=np.int64)
    lineage = np.asarray(lineage_child_queries, dtype=np.int64)
    positions = np.asarray(lineage_positions, dtype=np.int64)
    for name, array in {
        "anchor_classes": anchor_classes_np,
        "anchor_queries": anchor_queries_np,
        "p2a0_selected_queries": selected,
        "lineage_child_queries": lineage,
        "lineage_positions": positions,
    }.items():
        if array.shape != (n,):
            raise ValueError(f"{name} must have one value per online anchor row")
    if fault_logits.ndim != 2 or tuple(fault_logits.shape)[0] != QUERY_COUNT:
        raise ValueError("fault_logits must be [900,C]")
    if np.any(selected < 0) or np.any(selected >= QUERY_COUNT):
        raise RuntimeError("R1-F0 requires a matched frozen P2-A0 candidate")
    if np.any(lineage < 0) or np.any(lineage >= QUERY_COUNT):
        raise RuntimeError("R1-F0 requires an available explicit lineage candidate")
    if np.any(positions < 0) or np.any(positions >= 256):
        raise RuntimeError("R1-F0 lineage position outside [0,256)")
    if not 3 <= int(target_frame_idx) <= 12:
        raise ValueError("target_frame_idx must be in the frozen 3..12 window")

    disagreement = selected != lineage
    rows = np.flatnonzero(disagreement)
    logits_probability = fault_logits.detach().float().sigmoid()
    top1_probability, predicted_class = logits_probability.max(dim=-1)
    top1_probability_np = top1_probability.cpu().numpy()
    predicted_class_np = predicted_class.cpu().numpy().astype(np.int64, copy=False)

    output: Dict[str, list] = {
        "source_row_index": [],
        "p2a0_selected_query": [],
        "lineage_child_query": [],
    }
    output.update({column: [] for column in EVIDENCE_FEATURE_COLUMNS})
    component_pairs = (
        ("geometry", "geometry"),
        ("embedding", "embedding"),
        ("class_cost", "class"),
        ("total_cost", "total"),
        ("distance_m", "distance_m"),
    )
    for row in rows.tolist():
        a = int(selected[row])
        l = int(lineage[row])
        output["source_row_index"].append(row)
        output["p2a0_selected_query"].append(a)
        output["lineage_child_query"].append(l)
        for output_name, component_name in component_pairs:
            a_value = float(values[component_name][row, a])
            l_value = float(values[component_name][row, l])
            output[f"A_{output_name}"].append(a_value)
            output[f"L_{output_name}"].append(l_value)
            output[f"delta_{output_name}"].append(l_value - a_value)
        a_anchor_probability = 1.0 - float(values["class"][row, a])
        l_anchor_probability = 1.0 - float(values["class"][row, l])
        output["A_anchor_class_probability"].append(a_anchor_probability)
        output["L_anchor_class_probability"].append(l_anchor_probability)
        output["delta_anchor_class_probability"].append(
            l_anchor_probability - a_anchor_probability
        )
        output["A_top1_probability"].append(float(top1_probability_np[a]))
        output["L_top1_probability"].append(float(top1_probability_np[l]))
        output["delta_top1_probability"].append(
            float(top1_probability_np[l] - top1_probability_np[a])
        )
        output["A_predicted_class"].append(int(predicted_class_np[a]))
        output["L_predicted_class"].append(int(predicted_class_np[l]))
        output["A_class_matches_anchor"].append(
            int(predicted_class_np[a] == anchor_classes_np[row])
        )
        output["L_class_matches_anchor"].append(
            int(predicted_class_np[l] == anchor_classes_np[row])
        )

        a_rank, a_row_margin = _row_rank_and_margin(values["total"][row], a)
        l_rank, l_row_margin = _row_rank_and_margin(values["total"][row], l)
        output["A_row_rank"].append(a_rank)
        output["L_row_rank"].append(l_rank)
        output["delta_row_rank"].append(l_rank - a_rank)
        output["A_row_margin"].append(a_row_margin)
        output["L_row_margin"].append(l_row_margin)
        output["delta_row_margin"].append(l_row_margin - a_row_margin)

        a_mutual, a_column_margin = _column_ownership(values["total"], row, a)
        l_mutual, l_column_margin = _column_ownership(values["total"], row, l)
        output["A_column_margin"].append(a_column_margin)
        output["L_column_margin"].append(l_column_margin)
        output["delta_column_margin"].append(l_column_margin - a_column_margin)
        output["A_mutual"].append(int(a_mutual))
        output["L_mutual"].append(int(l_mutual))
        output["anchor_is_propagated"].append(int(anchor_queries_np[row] >= 644))
        output["p2a0_is_propagated"].append(int(a >= 644))
        output["lineage_position_norm"].append(float(positions[row] / 255.0))
        output["target_frame_norm"].append(float((int(target_frame_idx) - 3) / 9.0))

    typed: Dict[str, np.ndarray] = {}
    integer_columns = {
        "source_row_index", "p2a0_selected_query", "lineage_child_query",
        "A_predicted_class", "L_predicted_class", "A_class_matches_anchor",
        "L_class_matches_anchor", "A_row_rank", "L_row_rank", "delta_row_rank",
        "A_mutual", "L_mutual", "anchor_is_propagated", "p2a0_is_propagated",
    }
    for key, column in output.items():
        dtype = np.int64 if key in integer_columns else np.float64
        typed[key] = np.asarray(column, dtype=dtype)
    if len(typed["source_row_index"]) != int(disagreement.sum()):
        raise RuntimeError("relative evidence did not preserve all disagreement rows")
    return typed


def offline_disagreement_labels(
    p2a0_selected_queries: Sequence[int],
    lineage_child_queries: Sequence[int],
    oracle_queries: Sequence[int],
) -> Dict[str, np.ndarray]:
    """Create mutually exclusive offline labels after online feature freezing."""
    a = np.asarray(p2a0_selected_queries, dtype=np.int64)
    l = np.asarray(lineage_child_queries, dtype=np.int64)
    oracle = np.asarray(oracle_queries, dtype=np.int64)
    if a.shape != l.shape or a.shape != oracle.shape or a.ndim != 1:
        raise ValueError("candidate and oracle label vectors must align")
    if np.any(a == l):
        raise RuntimeError("offline labels require disagreement rows only")
    p2a0_wins = a == oracle
    lineage_wins = l == oracle
    both_wrong = (a != oracle) & (l != oracle)
    exhaustive = p2a0_wins.astype(int) + lineage_wins.astype(int) + both_wrong.astype(int)
    if not np.all(exhaustive == 1):
        raise RuntimeError("R1-F0 labels are not mutually exclusive and exhaustive")
    label = np.where(
        p2a0_wins,
        "P2A0_WINS",
        np.where(lineage_wins, "LINEAGE_WINS", "BOTH_WRONG"),
    )
    return {
        "p2a0_wins": p2a0_wins.astype(np.int8),
        "lineage_wins": lineage_wins.astype(np.int8),
        "both_wrong": both_wrong.astype(np.int8),
        "outcome_class": label.astype(object),
    }
