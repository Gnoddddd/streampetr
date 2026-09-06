"""Pure helpers for CARE-3D P1 sparse evidence routing."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


PROTOCOLS = ("blur_back", "crash_back", "dark_back")
PROTOCOL_INDEX = {name: index for index, name in enumerate(PROTOCOLS)}
MODEL_CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
SOURCE_NAMES = ("CAM_BACK_LEFT", "CAM_BACK_RIGHT", "TEMPORAL_ANCHOR")
SOURCE_CAMERA_INDICES = (
    MODEL_CAMERA_ORDER.index("CAM_BACK_LEFT"),
    MODEL_CAMERA_ORDER.index("CAM_BACK_RIGHT"),
)
QUERY_COLLISION_POLICY = "exclude_all_rows_in_shared_target_query_frame"


def assert_exact_sample_alignment(expected: Sequence[str], observed: Sequence[str]) -> None:
    if list(expected) != list(observed):
        raise RuntimeError("P1 sample order differs from frozen P0 cohort")


def assert_unique_queries(query_indices: Sequence[int]) -> None:
    values = [int(value) for value in query_indices]
    if len(values) != len(set(values)):
        raise RuntimeError("multiple P1 cohort objects map to the same target query in one frame")


def query_collision_eligibility(
    target_frame_indices: Sequence[int],
    query_indices: Sequence[int],
) -> Dict[str, object]:
    """Return the outcome-blind P1 eligibility mask for shared-query collisions.

    A detector query can receive only one routed residual/class vector at a
    frame.  If multiple frozen P0 object rows share the same
    ``(target_frame_idx, target_clean_query_index)`` pair, an object-specific
    intervention is not uniquely defined.  P1 therefore excludes *all* rows in
    that collision group rather than selecting one using labels or outcomes.
    """
    frames = np.asarray([int(value) for value in target_frame_indices], dtype=np.int64)
    queries = np.asarray([int(value) for value in query_indices], dtype=np.int64)
    if frames.shape != queries.shape or frames.ndim != 1:
        raise ValueError("target-frame and query-index vectors must be aligned 1-D arrays")
    counts: Dict[Tuple[int, int], int] = {}
    for frame, query in zip(frames.tolist(), queries.tolist()):
        key = (int(frame), int(query))
        counts[key] = counts.get(key, 0) + 1
    multiplicity = np.asarray(
        [counts[(int(frame), int(query))] for frame, query in zip(frames, queries)],
        dtype=np.int64,
    )
    eligible = multiplicity == 1
    return {
        "eligible": eligible,
        "multiplicity": multiplicity,
        "p0_rows_total": int(len(frames)),
        "p1_eligible_rows": int(eligible.sum()),
        "query_collision_excluded_rows": int((~eligible).sum()),
        "query_collision_groups": int(sum(count > 1 for count in counts.values())),
        "policy": QUERY_COLLISION_POLICY,
    }


def filter_aligned_rows(
    frame,
    arrays: Dict[str, np.ndarray],
) -> Tuple[object, Dict[str, np.ndarray], Dict[str, object]]:
    """Apply the frozen shared-query eligibility rule to a P0 scene payload."""
    if "target_frame_idx" not in frame or "target_clean_query_index" not in frame:
        raise RuntimeError("P0 metadata lacks P1 target-frame/query identity")
    audit = query_collision_eligibility(
        frame["target_frame_idx"].to_numpy(dtype=int),
        frame["target_clean_query_index"].to_numpy(dtype=int),
    )
    mask = np.asarray(audit["eligible"], dtype=bool)
    raw_n = len(frame)
    filtered = frame.loc[mask].reset_index(drop=True).copy()
    filtered["p1_query_multiplicity"] = 1
    filtered["p1_query_collision_policy"] = QUERY_COLLISION_POLICY
    output: Dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.ndim > 0 and len(array) == raw_n:
            output[key] = array[mask].copy()
        else:
            output[key] = array.copy()
    output["_p1_p0_rows_total"] = np.asarray(audit["p0_rows_total"], dtype=np.int64)
    output["_p1_eligible_rows"] = np.asarray(audit["p1_eligible_rows"], dtype=np.int64)
    output["_p1_query_collision_excluded_rows"] = np.asarray(
        audit["query_collision_excluded_rows"], dtype=np.int64
    )
    output["_p1_query_collision_groups"] = np.asarray(
        audit["query_collision_groups"], dtype=np.int64
    )
    if len(filtered):
        assert_unique_queries(
            filtered["target_clean_query_index"].to_numpy(dtype=int).tolist()
            if filtered["target_frame_idx"].nunique() == 1
            else []
        )
        for _, group in filtered.groupby("target_frame_idx", sort=False):
            assert_unique_queries(group["target_clean_query_index"].to_numpy(dtype=int).tolist())
    return filtered, output, audit


def sample_projected_camera_tokens(
    p0: Tensor,
    lidar2img: Tensor,
    centers_lidar: Tensor,
    image_hw: Tuple[int, int],
    camera_indices: Sequence[int] = SOURCE_CAMERA_INDICES,
) -> Tuple[Tensor, Tensor]:
    """Bilinearly sample FPN-P0 at predicted 3D centers.

    No GT geometry enters this function. ``centers_lidar`` are detector-predicted
    centers for the fixed paired query.  Invalid/out-of-view samples are zeroed
    and receive reliability zero.

    Args:
        p0: ``[6, C, Hf, Wf]`` image feature tensor.
        lidar2img: ``[6, 4, 4]`` transformed projection matrices.
        centers_lidar: ``[N, 3]`` predicted centers.
        image_hw: network image height/width.
    Returns:
        camera_tokens: ``[N, len(camera_indices), C]``.
        reliability: ``[N, len(camera_indices)]`` in ``{0, 1}``.
    """
    if p0.ndim != 4 or p0.shape[0] != 6:
        raise ValueError("p0 must have shape [6, C, Hf, Wf]")
    if lidar2img.shape[-2:] != (4, 4) or lidar2img.shape[0] != 6:
        raise ValueError("lidar2img must have shape [6, 4, 4]")
    if centers_lidar.ndim != 2 or centers_lidar.shape[1] != 3:
        raise ValueError("centers_lidar must have shape [N, 3]")
    if any(int(index) == MODEL_CAMERA_ORDER.index("CAM_BACK") for index in camera_indices):
        raise ValueError("failed CAM_BACK is prohibited from the P1 backup bank")

    n = centers_lidar.shape[0]
    channels = p0.shape[1]
    if n == 0:
        return (
            p0.new_empty((0, len(camera_indices), channels)),
            p0.new_empty((0, len(camera_indices))),
        )
    height, width = int(image_hw[0]), int(image_hw[1])
    if height <= 1 or width <= 1:
        raise ValueError("invalid image dimensions")

    ones = centers_lidar.new_ones((n, 1))
    homogeneous = torch.cat([centers_lidar, ones], dim=1)
    sampled, validities = [], []
    for camera_index in camera_indices:
        camera_index = int(camera_index)
        projection = lidar2img[camera_index].to(
            device=centers_lidar.device, dtype=centers_lidar.dtype
        )
        projected = homogeneous @ projection.transpose(0, 1)
        depth = projected[:, 2]
        safe_depth = depth.clamp_min(1e-6)
        u = projected[:, 0] / safe_depth
        v = projected[:, 1] / safe_depth
        valid = (
            (depth > 1e-3)
            & (u >= 0.0)
            & (u <= float(width - 1))
            & (v >= 0.0)
            & (v <= float(height - 1))
        )
        x = 2.0 * u / float(width - 1) - 1.0
        y = 2.0 * v / float(height - 1) - 1.0
        grid = torch.stack([x, y], dim=-1).view(1, n, 1, 2)
        token = F.grid_sample(
            p0[camera_index : camera_index + 1],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0, :, :, 0].transpose(0, 1).contiguous()
        token = token * valid[:, None].to(token.dtype)
        sampled.append(token)
        validities.append(valid.to(token.dtype))
    return torch.stack(sampled, dim=1), torch.stack(validities, dim=1)


def build_source_bank(
    camera_tokens: Tensor,
    camera_reliability: Tensor,
    temporal_anchor: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Append the pre-fault temporal object token as the third backup source."""
    if camera_tokens.ndim != 3 or camera_tokens.shape[1] != 2:
        raise ValueError("camera_tokens must have shape [N, 2, C]")
    if camera_reliability.shape != camera_tokens.shape[:2]:
        raise ValueError("camera_reliability shape mismatch")
    if temporal_anchor.ndim != 2 or temporal_anchor.shape[0] != camera_tokens.shape[0] \
            or temporal_anchor.shape[1] != camera_tokens.shape[2]:
        raise ValueError("temporal_anchor shape mismatch")
    temporal = temporal_anchor[:, None, :]
    temporal_reliability = camera_reliability.new_ones((camera_tokens.shape[0], 1))
    sources = torch.cat([camera_tokens, temporal], dim=1)
    reliability = torch.cat([camera_reliability, temporal_reliability], dim=1)
    if sources.shape[1] != len(SOURCE_NAMES):
        raise RuntimeError("P1 source-bank width changed")
    return sources, reliability


def assert_source_contract(source_features: np.ndarray, reliability: np.ndarray) -> None:
    if source_features.ndim != 4 or source_features.shape[2:] != (3, 256):
        raise RuntimeError("P1 source_features must be [N, P, 3, 256]")
    if reliability.shape != source_features.shape[:3]:
        raise RuntimeError("P1 source reliability shape mismatch")
    if not np.isfinite(source_features).all() or not np.isfinite(reliability).all():
        raise RuntimeError("non-finite P1 source bank")
    if np.any(reliability < 0):
        raise RuntimeError("negative source reliability")
    # Temporal anchor is source index 2 and must always be available.
    if reliability.size and not np.all(reliability[:, :, 2] == 1):
        raise RuntimeError("temporal anchor unexpectedly unavailable")


def topk_score_threshold(logits: np.ndarray, k: int = 100) -> float:
    tensor = torch.as_tensor(logits).float().sigmoid().reshape(-1)
    if tensor.numel() == 0:
        return float("nan")
    values = torch.topk(tensor, min(int(k), tensor.numel())).values
    return float(values[-1].item())


def fixed_target_metrics(logits: np.ndarray, query: int, label: int, k: int = 100) -> Dict[str, float]:
    tensor = torch.as_tensor(logits).float().sigmoid()
    query, label = int(query), int(label)
    target = float(tensor[query, label].item())
    flat = tensor.reshape(-1)
    flat_index = query * tensor.shape[1] + label
    order = torch.argsort(flat, descending=True)
    location = (order == flat_index).nonzero(as_tuple=False)
    rank = int(location[0, 0].item()) + 1 if len(location) else int(flat.numel()) + 1
    return {"score": target, "rank": rank, "topk": int(rank <= int(k))}


def cluster_bootstrap_mean(
    values: np.ndarray,
    clusters: Sequence[str],
    repetitions: int,
    seed: int,
) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    clusters = np.asarray([str(value) for value in clusters], dtype=object)
    finite = np.isfinite(values)
    values, clusters = values[finite], clusters[finite]
    if values.size == 0:
        return {"estimate": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "finite_bootstraps": 0}
    unique = np.unique(clusters)
    by_cluster = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    rng, estimates = np.random.default_rng(int(seed)), []
    for _ in range(int(repetitions)):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        indexes = np.concatenate([by_cluster[value] for value in chosen])
        estimate = float(np.mean(values[indexes]))
        if np.isfinite(estimate):
            estimates.append(estimate)
    low, high = (np.percentile(estimates, [2.5, 97.5])
                 if estimates else (np.nan, np.nan))
    return {
        "estimate": float(np.mean(values)),
        "ci_low": float(low),
        "ci_high": float(high),
        "finite_bootstraps": len(estimates),
    }


def cluster_bootstrap_fp_inflation(
    frame,
    repetitions: int,
    seed: int,
) -> Dict[str, float]:
    """Scene-cluster bootstrap of relative deployed false-positive inflation."""
    if len(frame) == 0:
        return {"estimate": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "finite_bootstraps": 0}
    scene_values = []
    for scene, group in frame.groupby("scene_token"):
        scene_values.append((str(scene), float(group.base_fp.sum()), float(group.patched_fp.sum())))
    rng = np.random.default_rng(int(seed))
    estimates = []
    for _ in range(int(repetitions)):
        indexes = rng.integers(0, len(scene_values), len(scene_values))
        base = sum(scene_values[index][1] for index in indexes)
        patched = sum(scene_values[index][2] for index in indexes)
        estimates.append((patched - base) / max(base, 1.0))
    base = sum(value[1] for value in scene_values)
    patched = sum(value[2] for value in scene_values)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return {
        "estimate": float((patched - base) / max(base, 1.0)),
        "ci_low": float(low),
        "ci_high": float(high),
        "finite_bootstraps": len(estimates),
    }


def p1_gate_flags(
    point: Dict[str, float],
    scene_ci: Dict[str, float],
    instance_ci: Dict[str, float],
    fp_ci: Dict[str, float],
    *,
    max_retained_damage_rate: float,
    max_retained_damage_ci_high: float,
    max_fp_inflation_rate: float,
    max_fp_inflation_ci_high: float,
) -> Dict[str, bool]:
    recovery_pass = bool(
        point["lost_recovery_rate"] > 0
        and scene_ci["lost_recovery_ci_low"] > 0
        and instance_ci["lost_recovery_ci_low"] > 0
    )
    net_tp_pass = bool(
        point["net_tp_delta"] > 0
        and scene_ci["net_tp_delta_ci_low"] > 0
        and instance_ci["net_tp_delta_ci_low"] > 0
    )
    crossing_pass = bool(
        point["cross_topk_recovery_rate"] > 0
        and scene_ci["cross_topk_recovery_ci_low"] > 0
        and instance_ci["cross_topk_recovery_ci_low"] > 0
    )
    score_pass = bool(
        point["target_score_delta_on_cross"] > 0
        and scene_ci["target_score_delta_ci_low"] > 0
        and instance_ci["target_score_delta_ci_low"] > 0
    )
    retained_pass = bool(
        point["retained_damage_rate"] <= float(max_retained_damage_rate)
        and scene_ci["retained_damage_ci_high"] <= float(max_retained_damage_ci_high)
        and instance_ci["retained_damage_ci_high"] <= float(max_retained_damage_ci_high)
    )
    fp_pass = bool(
        point["fp_inflation_rate"] <= float(max_fp_inflation_rate)
        and fp_ci["ci_high"] <= float(max_fp_inflation_ci_high)
    )
    clean_pass = bool(point.get("clean_identity_pass", False))
    passed = all((recovery_pass, net_tp_pass, crossing_pass, score_pass,
                  retained_pass, fp_pass, clean_pass))
    return {
        "lost_recovery_pass": recovery_pass,
        "net_tp_pass": net_tp_pass,
        "cross_topk_recovery_pass": crossing_pass,
        "target_score_pass": score_pass,
        "retained_no_harm_pass": retained_pass,
        "fp_control_pass": fp_pass,
        "clean_identity_pass": clean_pass,
        "seed_protocol_pass": passed,
    }
