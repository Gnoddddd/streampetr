"""Parameter-free training losses for Fault-Equivariant Query Learning."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

PRESENT, ABSENT, UNOBSERVED = 1, 0, 2


def supervision_weights(states: Tensor, reliable_history: Tensor) -> Tensor:
    """Return the frozen tri-state FEQ weights (Present=1, Unobserved=.35)."""
    result = torch.zeros_like(states, dtype=torch.float32)
    result = torch.where(states == PRESENT, result.new_tensor(1.0), result)
    eligible = (states == UNOBSERVED) & reliable_history.bool()
    return torch.where(eligible, result.new_tensor(0.35), result)


def greedy_auxiliary_assignment(
    cost: Tensor,
    main_query_for_gt: Tensor,
    eligible_gt: Tensor,
    max_aux: int = 3,
) -> Tuple[List[List[int]], int]:
    """Detached globally-greedy one-to-many assignment with unique queries."""
    if max_aux < 0:
        raise ValueError("max_aux must be non-negative")
    query_count, gt_count = cost.shape
    result: List[List[int]] = [[] for _ in range(gt_count)]
    used = {int(q) for q in main_query_for_gt.tolist() if int(q) >= 0}
    contenders: Dict[int, int] = {}
    triples = []
    detached = cost.detach().float().cpu()
    for gt in range(gt_count):
        if not bool(eligible_gt[gt]):
            continue
        for query in range(query_count):
            if query in used:
                continue
            triples.append((float(detached[query, gt]), query, gt))
            contenders[query] = contenders.get(query, 0) + 1
    conflicts = sum(value - 1 for value in contenders.values() if value > 1)
    for _, query, gt in sorted(triples):
        if query in used or len(result[gt]) >= max_aux:
            continue
        result[gt].append(query)
        used.add(query)
    return result, conflicts


def geometric_auxiliary_cost(
    boxes: Tensor,
    normalized_gt: Tensor,
    pc_range: Sequence[float],
    box_weight: float = 1.0,
    motion_distance: Tensor = None,
) -> Tensor:
    """Detached geometry-first cost with no classification dependency.

    ``boxes`` and ``normalized_gt`` use StreamPETR's ten-value regression
    encoding.  World-space xyz is normalized by the detector range; the
    existing encoded box L1 term supplies size/yaw/velocity geometry.  An
    optional precomputed motion distance is accepted only by the caller for
    reliable-history Unobserved objects.
    """
    span = boxes.new_tensor(pc_range[3:6]) - boxes.new_tensor(pc_range[:3])
    center = torch.cdist(
        (boxes[:, :3] / span).float(),
        (normalized_gt[:, :3] / span).float(),
    )
    box = torch.cdist(boxes[:, :10].float(), normalized_gt[:, :10].float(), p=1)
    result = center + float(box_weight) * box / 10.0
    if motion_distance is not None:
        if motion_distance.shape != result.shape:
            raise ValueError("motion_distance must have [query, gt] shape")
        result = result + motion_distance.detach().float()
    return result.detach()


def deployment_topk_boundary(scores: Tensor, max_outputs: int) -> Tuple[Tensor, Tensor]:
    """Return the exact flattened query×class deployment boundary and indices."""
    # fp32 keeps the audit valid on CPU FP16 and matches force_fp32 training.
    flat = scores.float().sigmoid().reshape(-1)
    count = min(max(int(max_outputs), 1), flat.numel())
    values, indices = torch.topk(flat, count)
    return values[-1].detach(), indices


def topk_boundary_loss(
    logits: Tensor,
    labels: Tensor,
    positive_sets: Sequence[Sequence[int]],
    weights: Tensor,
    max_outputs: int,
    margin: float = 0.10,
) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
    """Single-best-positive hinge against a detached deployment Top-K edge."""
    boundary, topk_indices = deployment_topk_boundary(logits, max_outputs)
    class_count = logits.shape[-1]
    details: List[Dict[str, Tensor]] = []
    terms = []
    for gt, positives in enumerate(positive_sets):
        if not positives or float(weights[gt]) == 0.0:
            continue
        label = int(labels[gt])
        query_indices = torch.as_tensor(positives, device=logits.device)
        positive_scores = logits[query_indices, label].float().sigmoid()
        positive_offset = int(torch.argmax(positive_scores))
        positive_query = query_indices[positive_offset]
        positive = positive_scores[positive_offset]
        loss = weights[gt] * F.relu(positive.new_tensor(margin) - positive + boundary)
        flat_positive_index = positive_query * class_count + label
        in_topk = torch.any(topk_indices == flat_positive_index)
        terms.append(loss)
        details.append({
            "gt": logits.new_tensor(gt, dtype=torch.long),
            "positive_query": positive_query.detach(),
            "s_pos": positive.detach(),
            "s_k": boundary,
            "gap": (positive - boundary).detach(),
            "violation": (positive - boundary < margin).detach(),
            "positive_in_topk": in_topk.detach(),
            "weighted_loss": loss.detach(),
        })
    zero = logits.sum() * 0.0
    return (torch.stack(terms).mean() if terms else zero), details


def ranking_loss(
    logits: Tensor,
    labels: Tensor,
    centers: Tensor,
    gt_centers: Tensor,
    positive_sets: Sequence[Sequence[int]],
    weights: Tensor,
    margin: float = 0.20,
    hard_negative_count: int = 5,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Positive-set logsumexp versus the strongest nearby unassigned query."""
    positive_union = {q for values in positive_sets for q in values}
    terms, margins = [], []
    for gt, positives in enumerate(positive_sets):
        if not positives or float(weights[gt]) == 0.0:
            continue
        candidates = [q for q in range(logits.shape[0]) if q not in positive_union]
        if not candidates:
            continue
        candidate_tensor = torch.as_tensor(candidates, device=logits.device)
        # CPU torch does not implement Half top-k; selection is detached and
        # therefore safely performed in fp32 on every device.
        distances = torch.linalg.vector_norm(
            (centers[candidate_tensor, :3] - gt_centers[gt, :3]).float(), dim=-1
        )
        count = min(int(hard_negative_count), len(candidates))
        negatives = candidate_tensor[torch.topk(distances, count, largest=False).indices]
        label = int(labels[gt])
        pos = torch.logsumexp(logits[list(positives), label].float(), dim=0)
        neg = logits[negatives, label].float().max()
        signed_margin = pos - neg
        margins.append(signed_margin.detach())
        terms.append(weights[gt] * F.relu(logits.new_tensor(margin) - signed_margin))
    zero = logits.sum() * 0.0
    loss = torch.stack(terms).mean() if terms else zero
    values = torch.stack(margins) if margins else logits.new_empty(0)
    return loss, {
        "margin_mean": values.mean() if values.numel() else zero.detach(),
        "violation_ratio": (values < margin).float().mean() if values.numel() else zero.detach(),
    }


def soft_best_quality(logits: Tensor, boxes: Tensor, label: int, gt_center: Tensor,
                      queries: Sequence[int]) -> Tensor:
    if not queries:
        return logits.sum() * 0.0
    q = torch.as_tensor(queries, device=logits.device)
    cls = logits[q, label].sigmoid()
    geometry = torch.exp(-torch.linalg.vector_norm(boxes[q, :3] - gt_center[:3], dim=-1))
    quality = cls * geometry
    return (torch.softmax(quality, dim=0) * quality).sum()


def adjacent_survival_loss(
    layer_logits: Sequence[Tensor],
    layer_boxes: Sequence[Tensor],
    labels: Tensor,
    gt_centers: Tensor,
    layer_positive_sets: Sequence[Sequence[Sequence[int]]],
    weights: Tensor,
    delta: float = 0.05,
) -> Tuple[Tensor, int]:
    """Set-level adjacent-layer survival; query indices are not lineages."""
    terms = []
    comparisons = 0
    for layer in range(len(layer_logits) - 1):
        for gt, previous in enumerate(layer_positive_sets[layer]):
            if not previous or float(weights[gt]) == 0.0:
                continue
            current = layer_positive_sets[layer + 1][gt]
            q_prev = soft_best_quality(layer_logits[layer], layer_boxes[layer],
                                       int(labels[gt]), gt_centers[gt], previous)
            q_next = soft_best_quality(layer_logits[layer + 1], layer_boxes[layer + 1],
                                       int(labels[gt]), gt_centers[gt], current)
            terms.append(weights[gt] * F.relu(q_prev - delta - q_next))
            comparisons += 1
    zero = layer_logits[0].sum() * 0.0
    return (torch.stack(terms).mean() if terms else zero), comparisons
