"""Pure helpers for the CARE-3D P2-A2-R0 one-step memory lineage audit.

Association functions in this module deliberately do not accept GT, oracle
queries, or clean future outputs.  Oracle labels are consumed only by the
separate :func:`evaluate_assignments` helper after all assignments are frozen.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from analysis.care3d_p1 import cluster_bootstrap_mean
from analysis.care3d_p2a_association import (
    NUM_CURRENT_QUERIES,
    NUM_PROPAGATED_QUERIES,
    QUERY_COUNT,
    AssociationConfig,
    hungarian_with_unmatched,
)


PRIMARY_METHOD = "lineage_first_hybrid"
BASELINE_METHODS = ("p2a0_frozen", "lineage_only")
FROZEN_P2A0_CONFIG = AssociationConfig(0.4, 0.4, 0.2, 0.45)
BOOTSTRAP_REPETITIONS = 5000


def query_origin(query: int) -> str:
    query = int(query)
    if 0 <= query < NUM_CURRENT_QUERIES:
        return "current"
    if NUM_CURRENT_QUERIES <= query < QUERY_COUNT:
        return "propagated"
    raise ValueError(f"query index outside frozen 900-query layout: {query}")


def recompute_topk_indexes(all_cls_scores_last: Tensor, topk_proposals: int) -> Tensor:
    """Reproduce StreamPETR ``post_update_memory`` proposal ordering exactly."""
    if all_cls_scores_last.ndim != 3:
        raise ValueError("all_cls_scores_last must be [B,Q,C]")
    if int(all_cls_scores_last.shape[1]) != QUERY_COUNT:
        raise RuntimeError("StreamPETR score query layout changed")
    if int(topk_proposals) != NUM_PROPAGATED_QUERIES:
        raise RuntimeError("StreamPETR topk_proposals changed")
    rec_score = (
        all_cls_scores_last.sigmoid().topk(1, dim=-1).values[..., 0:1]
    )
    _, topk_indexes = torch.topk(rec_score, int(topk_proposals), dim=1)
    return topk_indexes


def topk_gather_exact(feat: Tensor, topk_indexes: Tensor) -> Tensor:
    """Local torch.gather equivalent of StreamPETR's frozen ``topk_gather``."""
    if feat.ndim < 2 or topk_indexes.ndim < 2:
        raise ValueError("topk gather tensors must have batch and query dimensions")
    if int(feat.shape[0]) != int(topk_indexes.shape[0]):
        raise ValueError("topk gather batch dimensions differ")
    view_shape = [1] * feat.ndim
    view_shape[:2] = topk_indexes.shape[:2]
    indexes = topk_indexes.view(*view_shape)
    repeats = (1, 1, *feat.shape[2:])
    return torch.gather(feat, 1, indexes.repeat(*repeats))


def verify_post_update_memory(
    outs_dec_last: Tensor,
    all_cls_scores_last: Tensor,
    memory_embedding: Tensor,
    *,
    topk_proposals: int,
) -> Dict[str, object]:
    """Verify the recomputed proposal order is the actual new memory prefix."""
    if outs_dec_last.ndim == 2:
        outs_dec_last = outs_dec_last.unsqueeze(0)
    if outs_dec_last.ndim != 3:
        raise ValueError("outs_dec_last must be [B,Q,D] or [Q,D]")
    indexes = recompute_topk_indexes(all_cls_scores_last, topk_proposals)
    expected = topk_gather_exact(outs_dec_last, indexes).detach()
    actual = memory_embedding[:, : int(topk_proposals)].detach()
    if expected.shape != actual.shape:
        raise RuntimeError(
            f"memory prefix shape changed: expected {tuple(expected.shape)}, "
            f"actual {tuple(actual.shape)}"
        )
    equal = bool(torch.equal(expected, actual))
    max_abs_diff = (
        float((expected - actual).abs().max().item()) if expected.numel() else 0.0
    )
    if not equal and max_abs_diff != 0.0:
        raise RuntimeError(
            "recomputed rec_memory differs from StreamPETR memory prefix: "
            f"max_abs_diff={max_abs_diff}"
        )
    return {
        "topk_indexes": indexes,
        "torch_equal": equal,
        "max_abs_diff": max_abs_diff,
    }


def lineage_assignments(
    anchor_queries: Sequence[int],
    topk_indexes: Tensor | np.ndarray,
    *,
    num_query: int = NUM_CURRENT_QUERIES,
) -> Dict[str, np.ndarray]:
    """Map unique anchor queries to their strict one-step propagated children."""
    anchors = np.asarray(anchor_queries, dtype=np.int64)
    indexes = (
        topk_indexes.detach().cpu().numpy()
        if torch.is_tensor(topk_indexes)
        else np.asarray(topk_indexes)
    )
    indexes = np.asarray(indexes, dtype=np.int64).reshape(-1)
    if anchors.ndim != 1:
        raise ValueError("anchor_queries must be one-dimensional")
    if len(anchors) != len(set(anchors.tolist())):
        raise RuntimeError("lineage anchor queries must be unique within a frame")
    if len(indexes) != NUM_PROPAGATED_QUERIES:
        raise RuntimeError("lineage Top-K length changed")
    if len(indexes) != len(set(indexes.tolist())):
        raise RuntimeError("StreamPETR Top-K indexes are not unique")
    if np.any(anchors < 0) or np.any(anchors >= QUERY_COUNT):
        raise ValueError("anchor query index outside frozen query layout")

    position_by_query = {int(query): pos for pos, query in enumerate(indexes.tolist())}
    positions = np.full(len(anchors), -1, dtype=np.int64)
    children = np.full(len(anchors), -1, dtype=np.int64)
    available = np.zeros(len(anchors), dtype=bool)
    for row, query in enumerate(anchors.tolist()):
        if int(query) not in position_by_query:
            continue
        position = int(position_by_query[int(query)])
        child = int(num_query) + position
        if not NUM_CURRENT_QUERIES <= child < QUERY_COUNT:
            raise RuntimeError(f"lineage child outside [644,900): {child}")
        positions[row] = position
        children[row] = child
        available[row] = True
    real_children = children[available]
    if len(real_children) != len(set(real_children.tolist())):
        raise RuntimeError("duplicate lineage child within frame")
    return {
        "anchor_topk_member": available.copy(),
        "lineage_position": positions,
        "lineage_child_query": children,
        "lineage_available": available,
    }


def lineage_first_assign(
    anchor_queries: Sequence[int],
    topk_indexes: Tensor | np.ndarray,
    frozen_cost: Tensor,
    *,
    max_cost: float = FROZEN_P2A0_CONFIG.max_cost,
) -> Dict[str, object]:
    """Perform frame-level lineage-first assignment and frozen P2-A0 fallback.

    Reserved lineage children remain represented in the 900-column tensor only
    because the frozen P2-A0 Hungarian helper asserts that layout.  Setting
    those columns to infinity is exactly removal from its feasible pool.
    """
    anchors = np.asarray(anchor_queries, dtype=np.int64)
    if frozen_cost.ndim != 2 or frozen_cost.shape != (len(anchors), QUERY_COUNT):
        raise ValueError("frozen_cost must be [N,900]")
    lineage = lineage_assignments(anchors, topk_indexes)
    available = lineage["lineage_available"]
    children = lineage["lineage_child_query"]

    p2a0 = hungarian_with_unmatched(frozen_cost, float(max_cost))
    lineage_only_query = np.where(available, children, -1).astype(np.int64)
    hybrid_query = lineage_only_query.copy()
    hybrid_cost = np.full(len(anchors), np.nan, dtype=np.float64)
    hybrid_source = np.where(available, "lineage", "unmatched").astype(object)

    fallback_rows = np.flatnonzero(~available)
    reserved = children[available]
    if fallback_rows.size:
        fallback_cost = frozen_cost[fallback_rows].clone()
        if reserved.size:
            reserved_tensor = torch.as_tensor(
                reserved, device=fallback_cost.device, dtype=torch.long
            )
            fallback_cost[:, reserved_tensor] = float("inf")
        fallback = hungarian_with_unmatched(fallback_cost, float(max_cost))
        hybrid_query[fallback_rows] = fallback["selected_query"]
        hybrid_cost[fallback_rows] = fallback["selected_cost"]
        matched = fallback["matched"]
        hybrid_source[fallback_rows[matched]] = "p2a0_fallback"
    if reserved.size and np.intersect1d(hybrid_query[fallback_rows], reserved).size:
        raise RuntimeError("frozen fallback selected a reserved lineage child")
    real_hybrid = hybrid_query[hybrid_query >= 0]
    if len(real_hybrid) != len(set(real_hybrid.tolist())):
        raise RuntimeError("lineage-first hybrid is not one-to-one")

    return {
        **lineage,
        "p2a0_selected_query": np.asarray(p2a0["selected_query"], dtype=np.int64),
        "p2a0_selected_cost": np.asarray(p2a0["selected_cost"], dtype=np.float64),
        "lineage_only_selected_query": lineage_only_query,
        "hybrid_selected_query": hybrid_query,
        "hybrid_source": hybrid_source,
        "fallback_selected_cost": hybrid_cost,
    }


def evaluate_assignments(
    assignments: Mapping[str, np.ndarray],
    oracle_queries: Sequence[int],
) -> Dict[str, np.ndarray]:
    """Attach offline oracle outcomes after all online assignments are complete."""
    oracle = np.asarray(oracle_queries, dtype=np.int64)
    output: Dict[str, np.ndarray] = {}
    for prefix, key in (
        ("lineage", "lineage_only_selected_query"),
        ("p2a0", "p2a0_selected_query"),
        ("hybrid", "hybrid_selected_query"),
    ):
        selected = np.asarray(assignments[key], dtype=np.int64)
        if selected.shape != oracle.shape:
            raise ValueError("oracle and selected query lengths differ")
        matched = selected >= 0
        exact = matched & (selected == oracle)
        wrong = matched & (selected != oracle)
        unmatched = ~matched
        if not np.all(exact.astype(int) + wrong.astype(int) + unmatched.astype(int) == 1):
            raise RuntimeError("lineage outcomes are not exhaustive")
        output[f"{prefix}_exact"] = exact.astype(np.int8)
        output[f"{prefix}_wrong"] = wrong.astype(np.int8)
        output[f"{prefix}_unmatched"] = unmatched.astype(np.int8)
    return output


def paired_delta_bootstrap(
    frame: pd.DataFrame,
    *,
    cluster_column: str,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    if int(repetitions) != BOOTSTRAP_REPETITIONS:
        raise RuntimeError("P2-A2-R0 requires exactly 5000 bootstrap repetitions")
    if cluster_column not in {"scene_token", "instance_token"}:
        raise ValueError("cluster must be scene_token or instance_token")
    clusters = frame[cluster_column].astype(str).tolist()
    return {
        "delta_exact": cluster_bootstrap_mean(
            frame.hybrid_exact.to_numpy(float) - frame.p2a0_exact.to_numpy(float),
            clusters,
            repetitions,
            int(seed),
        ),
        "delta_wrong": cluster_bootstrap_mean(
            frame.hybrid_wrong.to_numpy(float) - frame.p2a0_wrong.to_numpy(float),
            clusters,
            repetitions,
            int(seed) + 1,
        ),
    }


def protocol_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    if len(frame) == 0:
        raise ValueError("protocol metrics require at least one row")
    available = frame.lineage_available.astype(bool)
    p2a0_wrong = frame.p2a0_wrong.astype(bool)
    p2a0_exact = frame.p2a0_exact.astype(bool)
    hybrid_exact = frame.hybrid_exact.astype(bool)
    hybrid_wrong = frame.hybrid_wrong.astype(bool)
    hybrid_unmatched = frame.hybrid_unmatched.astype(bool)
    wrong_repaired = p2a0_wrong & hybrid_exact
    correct_broken = p2a0_exact & hybrid_wrong
    correct_to_unmatched = p2a0_exact & hybrid_unmatched
    return {
        "rows": int(len(frame)),
        "lineage_coverage": float(available.mean()),
        "lineage_conditional_exact": float(
            frame.loc[available, "lineage_exact"].astype(float).mean()
        ) if available.any() else float("nan"),
        "lineage_conditional_wrong": float(
            frame.loc[available, "lineage_wrong"].astype(float).mean()
        ) if available.any() else float("nan"),
        "p2a0_exact_recall": float(p2a0_exact.mean()),
        "p2a0_wrong_match_rate": float(p2a0_wrong.mean()),
        "p2a0_unmatched_rate": float(frame.p2a0_unmatched.astype(float).mean()),
        "hybrid_exact_recall": float(hybrid_exact.mean()),
        "hybrid_wrong_match_rate": float(hybrid_wrong.mean()),
        "hybrid_unmatched_rate": float(hybrid_unmatched.mean()),
        "wrong_repaired_n": int(wrong_repaired.sum()),
        "correct_broken_n": int(correct_broken.sum()),
        "correct_to_unmatched_n": int(correct_to_unmatched.sum()),
        "net_exact_gain": float(hybrid_exact.mean() - p2a0_exact.mean()),
        "wrong_repair_fraction": float(wrong_repaired.sum() / p2a0_wrong.sum())
        if p2a0_wrong.any() else float("nan"),
        "break_fraction": float(correct_broken.sum() / p2a0_exact.sum())
        if p2a0_exact.any() else float("nan"),
    }
