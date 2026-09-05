"""Train-only hard-positive Top-K boundary objective helpers."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor


def deployment_pair_rank(flat_scores: Tensor, flat_index: int) -> int:
    """Stable one-based rank in flattened query-major deployment order."""
    values = flat_scores.detach().float().reshape(-1)
    target = values[int(flat_index)]
    better = int((values > target).sum())
    tied_before = int((values[:int(flat_index)] == target).sum())
    return better + tied_before + 1


def select_hard_positive_pairs(
    logits: Tensor,
    boxes: Tensor,
    gt_centers: Tensor,
    gt_labels: Tensor,
    topk: int = 100,
    geometry_threshold: float = 2.0,
) -> tuple[List[Dict[str, float]], Dict[str, float]]:
    """Detached selection of strict GT-near, rank-out positive pairs.

    The actual loss is intentionally not computed here. Returning integer
    indices keeps selection outside autograd while the caller indexes the
    original logits to retain gradients only through selected scores.
    """
    if logits.ndim != 2 or boxes.ndim != 2:
        raise ValueError("logits and boxes must be [query, ...]")
    if len(gt_centers) != len(gt_labels):
        raise ValueError("GT center/label count mismatch")
    with torch.no_grad():
        scores = logits.detach().float().sigmoid()
        flat = scores.reshape(-1)
        count = min(max(int(topk), 1), flat.numel())
        boundary_values, boundary_indices = torch.topk(flat, count)
        negative_flat = int(boundary_indices[-1])
        boundary = float(boundary_values[-1])
        class_count = logits.shape[-1]
        distances = torch.cdist(boxes[:, :3].detach().float(),
                                gt_centers[:, :3].detach().float())
        candidates = []
        per_gt = []
        for gt in range(len(gt_labels)):
            label = int(gt_labels[gt])
            near = torch.nonzero(
                torch.isfinite(distances[:, gt])
                & (distances[:, gt] <= float(geometry_threshold)),
                as_tuple=False,
            ).flatten()
            ranks = []
            ranked = []
            for query_tensor in near:
                query = int(query_tensor)
                flat_index = query * class_count + label
                rank = deployment_pair_rank(flat, flat_index)
                ranks.append(rank)
                ranked.append((rank, float(distances[query, gt]), query, flat_index))
            # A GT is hard only when its *best-scoring* geometry-qualified
            # query is outside deployment Top-K. Lower-ranked duplicate near
            # queries must never turn an otherwise easy GT into an event.
            if ranked:
                rank, distance, query, flat_index = min(ranked)
                if rank > count:
                    candidates.append((rank, distance, gt, query, flat_index))
            per_gt.append({
                "gt": gt,
                "near_query_count": int(near.numel()),
                "best_near_rank": min(ranks) if ranks else -1,
                "strict_rank_out_candidates": sum(rank > count for rank in ranks),
            })
        # Prefer the positive closest to the real boundary. Global uniqueness
        # prevents one query receiving contradictory GT-class objectives.
        used_gt, used_query = set(), set()
        selected = []
        for rank, distance, gt, query, flat_index in sorted(candidates):
            if gt in used_gt or query in used_query:
                continue
            used_gt.add(gt); used_query.add(query)
            selected.append({
                "gt": gt,
                "positive_query": query,
                "positive_flat_index": flat_index,
                "positive_rank": rank,
                "rank_distance_from_k": rank - count,
                "center_distance": distance,
                "negative_flat_index": negative_flat,
                "negative_query": negative_flat // class_count,
                "negative_class": negative_flat % class_count,
                "negative_rank": count,
                "boundary_score_detached": boundary,
            })
        return selected, {
            "topk": count,
            "boundary_score": boundary,
            "eligible_gt": len(used_gt),
            "gt_count": len(gt_labels),
            "per_gt": per_gt,
        }


def hard_positive_boundary_loss(
    logits: Tensor,
    selected: List[Dict[str, float]],
    margin: float = 0.10,
) -> tuple[Tensor, List[Dict[str, float]]]:
    """Pairwise hinge with gradients only through selected q+ and q- scores."""
    flat = logits.float().sigmoid().reshape(-1)
    terms, diagnostics = [], []
    for item in selected:
        positive = flat[int(item["positive_flat_index"])]
        negative = flat[int(item["negative_flat_index"])]
        loss = F.relu(positive.new_tensor(float(margin)) + negative - positive)
        terms.append(loss)
        diagnostics.append({
            **item,
            "s_pos": float(positive.detach()),
            "s_neg": float(negative.detach()),
            "score_gap": float((positive - negative).detach()),
            "loss": float(loss.detach()),
            "nonzero": bool(loss.detach() > 0),
            "negative_truly_outranks": bool(negative.detach() > positive.detach()),
        })
    zero = logits.sum() * 0.0
    return (torch.stack(terms).mean() if terms else zero), diagnostics
