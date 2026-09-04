"""Pure, deterministic utilities for the reviewer-proof recovery audit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Candidate:
    lineage: int
    label: int
    score: float
    center: np.ndarray
    velocity: np.ndarray
    feature: np.ndarray
    box: np.ndarray


def deduplicate_lineages(
    lineages: np.ndarray,
    layers: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Return one (latest, score-tiebroken) row per query lineage."""
    chosen = {}
    for index, (lineage, layer, score) in enumerate(
        zip(lineages.tolist(), layers.tolist(), scores.tolist())
    ):
        key = int(lineage)
        rank = (int(layer), float(score), -index)
        if key not in chosen or rank > chosen[key][0]:
            chosen[key] = (rank, index)
    return np.asarray(
        [value[1] for _, value in sorted(chosen.items())], dtype=np.int64
    )


def cluster_deduplicate(
    labels: np.ndarray,
    centers: np.ndarray,
    velocities: np.ndarray,
    layers: np.ndarray,
    scores: np.ndarray,
    center_threshold: float = 1.0,
    velocity_threshold: float = 1.0,
) -> np.ndarray:
    """Fallback cross-layer clustering when lineage metadata is unavailable."""
    order = sorted(
        range(len(scores)),
        key=lambda i: (int(layers[i]), float(scores[i])),
        reverse=True,
    )
    retained = []
    for index in order:
        duplicate = any(
            int(labels[index]) == int(labels[other])
            and np.linalg.norm(centers[index] - centers[other])
            <= center_threshold
            and np.linalg.norm(velocities[index] - velocities[other])
            <= velocity_threshold
            for other in retained
        )
        if not duplicate:
            retained.append(index)
    return np.asarray(retained, dtype=np.int64)


def topk_unique(candidates: list[Candidate], k: int) -> list[Candidate]:
    ordered = sorted(candidates, key=lambda value: (-value.score, value.lineage))
    unique = {}
    for value in ordered:
        unique.setdefault(value.lineage, value)
    result = list(unique.values())[:k]
    if len(result) != k:
        raise ValueError(f"candidate budget {k} cannot be filled ({len(result)})")
    return result


def secondary_allocation(
    candidates: list[Candidate],
    earlier_scores: dict[int, float],
    k: int,
    replacement_fraction: float = 0.2,
) -> list[Candidate]:
    base = topk_unique(candidates, k)
    replace = max(1, int(round(k * replacement_fraction)))
    keep = base[: k - replace]
    excluded = {value.lineage for value in keep}
    pool = [value for value in candidates if value.lineage not in excluded]
    pool.sort(
        key=lambda value: (
            -float(earlier_scores.get(value.lineage, -np.inf)),
            -value.score,
            value.lineage,
        )
    )
    return keep + pool[:replace]


def motion_allocation(
    candidates: list[Candidate],
    anchors: list[Candidate],
    elapsed_seconds: float,
    k: int,
    replacement_fraction: float = 0.2,
) -> list[Candidate]:
    base = topk_unique(candidates, k)
    replace = max(1, int(round(k * replacement_fraction)))
    keep = base[: k - replace]
    excluded = {value.lineage for value in keep}
    threshold = max(2.0, 2.0 * elapsed_seconds)

    def association(value: Candidate) -> tuple[float, float, int]:
        distances = [
            np.linalg.norm(
                value.center[:2]
                - (anchor.center[:2] + anchor.velocity[:2] * elapsed_seconds)
            )
            for anchor in anchors
            if anchor.label == value.label
        ]
        distance = min(distances, default=np.inf)
        eligible = distance <= threshold
        return (0.0 if eligible else 1.0, distance, value.lineage)

    pool = [value for value in candidates if value.lineage not in excluded]
    pool.sort(key=association)
    return keep + pool[:replace]


def greedy_match(
    candidates: list[Candidate],
    gt_labels: np.ndarray,
    gt_centers: np.ndarray,
    max_distance: float = 2.0,
) -> dict[int, int]:
    """Map candidate index to GT index, score ordered and one-to-one."""
    available = set(range(len(gt_labels)))
    output = {}
    for index in sorted(
        range(len(candidates)), key=lambda i: -candidates[i].score
    ):
        value = candidates[index]
        valid = [
            gt_index
            for gt_index in available
            if int(gt_labels[gt_index]) == value.label
            and np.linalg.norm(gt_centers[gt_index, :3] - value.center[:3])
            <= max_distance
        ]
        if valid:
            target = min(
                valid,
                key=lambda gt_index: np.linalg.norm(
                    gt_centers[gt_index, :3] - value.center[:3]
                ),
            )
            output[index] = target
            available.remove(target)
    return output


def survival_state(
    center: np.ndarray,
    matched: bool,
    projection_visible: bool,
    post_range: np.ndarray,
) -> str:
    if np.any(center[:3] < post_range[:3]) or np.any(
        center[:3] > post_range[3:]
    ):
        return "Terminated-Out-of-range"
    if matched:
        return "Present"
    if not projection_visible:
        return "Unobserved"
    return "Absent"


def delayed_promotions(
    identities: list[str | None],
    window: int,
    required: int,
) -> list[bool]:
    """Causal confirmation: only the current index can be promoted."""
    output = []
    for index, identity in enumerate(identities):
        history = identities[max(0, index - window + 1) : index + 1]
        output.append(
            identity is not None
            and sum(value == identity for value in history) >= required
        )
    return output


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels.astype(bool)]
    negatives = scores[~labels.astype(bool)]
    if not len(positives) or not len(negatives):
        return float("nan")
    return float(
        np.mean(
            positives[:, None] > negatives[None, :]
        )
        + 0.5
        * np.mean(positives[:, None] == negatives[None, :])
    )
