"""Train-only paired Fault rank-margin preservation objective."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def _stable_rank(flat_scores: Tensor, flat_index: int) -> int:
    values = flat_scores.detach().float().reshape(-1)
    index = int(flat_index)
    target = values[index]
    return int((values > target).sum()) + int((values[:index] == target).sum()) + 1


def _branch_candidate(
    logits: Tensor,
    boxes: Tensor,
    gt_center: Tensor,
    gt_class: int,
    topk: int,
    geometry_threshold: float,
) -> Dict[str, float]:
    """Select one GT-level candidate pool representative under no-grad."""
    scores = logits.detach().float().sigmoid()
    flat = scores.reshape(-1)
    count = min(max(int(topk), 1), int(flat.numel()))
    boundary_result = torch.topk(flat, count, sorted=True)
    boundary = float(boundary_result.values[-1])
    boundary_flat_index = int(boundary_result.indices[-1])
    distances = torch.linalg.vector_norm(
        boxes.detach().float()[:, :3] - gt_center.detach().float()[:3], dim=-1
    )
    near = torch.nonzero(
        torch.isfinite(distances) & (distances <= float(geometry_threshold)),
        as_tuple=False,
    ).flatten()
    if not len(near):
        return {
            "available": False, "query": -1, "flat_index": -1,
            "score": float("nan"), "rank": -1, "s_k": boundary,
            "s_k_flat_index": boundary_flat_index,
            "margin": float("nan"), "distance": float("nan"),
            "near_count": 0,
        }
    label = int(gt_class)
    # Stable tie break by query index. Selection never enters autograd.
    choices = sorted(
        (float(scores[int(query), label]), -int(query), int(query))
        for query in near
    )
    query = choices[-1][2]
    flat_index = query * logits.shape[-1] + label
    score = float(scores[query, label])
    return {
        "available": True, "query": query, "flat_index": flat_index,
        "score": score, "rank": _stable_rank(flat, flat_index),
        "s_k": boundary, "margin": score - boundary,
        "s_k_flat_index": boundary_flat_index,
        "distance": float(distances[query]), "near_count": int(len(near)),
    }


def select_paired_margin_events(
    clean_logits: Tensor,
    clean_boxes: Tensor,
    fault_logits: Tensor,
    fault_boxes: Tensor,
    gt_centers: Tensor,
    gt_labels: Tensor,
    topk: int = 100,
    geometry_threshold: float = 2.0,
    delta: float = 0.10,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """Detached, query-ID-free selection over paired GT candidate pools."""
    if clean_logits.shape != fault_logits.shape:
        raise ValueError("Clean/Fault logits shape mismatch")
    if clean_boxes.shape != fault_boxes.shape:
        raise ValueError("Clean/Fault boxes shape mismatch")
    if len(gt_centers) != len(gt_labels):
        raise ValueError("GT center/label count mismatch")
    events: List[Dict[str, float]] = []
    with torch.no_grad():
        for gt, (center, label_tensor) in enumerate(zip(gt_centers, gt_labels)):
            label = int(label_tensor)
            clean = _branch_candidate(
                clean_logits, clean_boxes, center, label, topk,
                geometry_threshold,
            )
            fault = _branch_candidate(
                fault_logits, fault_boxes, center, label, topk,
                geometry_threshold,
            )
            paired = bool(clean["available"] and fault["available"])
            clean_in = bool(paired and clean["margin"] > 0.0)
            fault_in = bool(paired and fault["margin"] > 0.0)
            delta_margin = (
                float(fault["margin"] - clean["margin"])
                if paired else float("nan")
            )
            eligible = bool(clean_in and fault["margin"] < clean["margin"])
            strong = bool(eligible and delta_margin <= -float(delta))
            events.append({
                "gt": gt, "gt_class": label,
                "clean_candidate_available": bool(clean["available"]),
                "fault_candidate_available": bool(fault["available"]),
                "paired_candidate_available": paired,
                "clean_query": int(clean["query"]),
                "fault_query": int(fault["query"]),
                "same_query_id": bool(paired and clean["query"] == fault["query"]),
                "fault_positive_flat_index": int(fault["flat_index"]),
                "clean_score": clean["score"], "fault_score": fault["score"],
                "clean_rank": int(clean["rank"]), "fault_rank": int(fault["rank"]),
                "clean_s_k": clean["s_k"], "fault_s_k": fault["s_k"],
                "clean_s_k_flat_index": int(clean["s_k_flat_index"]),
                "fault_s_k_flat_index": int(fault["s_k_flat_index"]),
                "clean_margin": clean["margin"], "fault_margin": fault["margin"],
                "delta_margin": delta_margin,
                "clean_distance": clean["distance"],
                "fault_distance": fault["distance"],
                "clean_near_count": int(clean["near_count"]),
                "fault_near_count": int(fault["near_count"]),
                "clean_in_topk": clean_in, "fault_in_topk": fault_in,
                "boundary_crossing": bool(clean_in and not fault_in),
                "lost_risk": bool(clean_in and not fault_in),
                "retained_like": bool(clean_in and fault_in),
                "old_absolute_fault_rank_out": bool(
                    fault["available"] and fault["rank"] > int(topk)
                ),
                "generic_clean_hard": bool(
                    clean["available"] and clean["rank"] > int(topk)
                ),
                "collapse_eligible": eligible,
                "strong_margin_collapse": strong,
                "delta": float(delta),
            })
    return events, {
        "gt": len(gt_labels),
        "paired_candidates": sum(e["paired_candidate_available"] for e in events),
        "clean_in_topk": sum(e["clean_in_topk"] for e in events),
        "collapse_eligible": sum(e["collapse_eligible"] for e in events),
        "strong_collapse": sum(e["strong_margin_collapse"] for e in events),
    }


def paired_margin_preservation_loss(
    fault_logits: Tensor,
    events: List[Dict[str, float]],
    delta: float = 0.10,
    enabled: bool = True,
) -> Tuple[Tensor, List[Dict[str, float]]]:
    """Hinge gradients only through the independently selected Fault q+."""
    zero = fault_logits.sum() * 0.0
    if not enabled:
        return zero, []
    flat = fault_logits.float().sigmoid().reshape(-1)
    terms, diagnostics = [], []
    for event in events:
        if not event["collapse_eligible"]:
            continue
        positive = flat[int(event["fault_positive_flat_index"])]
        fault_margin = positive - positive.new_tensor(float(event["fault_s_k"]))
        target = positive.new_tensor(float(event["clean_margin"]) - float(delta))
        term = F.relu(target - fault_margin)
        terms.append(term)
        diagnostics.append({
            **event,
            "live_fault_margin": float(fault_margin.detach()),
            "target_margin": float(target),
            "loss": float(term.detach()),
            "nonzero": bool(term.detach() > 0),
        })
    return (torch.stack(terms).mean() if terms else zero), diagnostics
