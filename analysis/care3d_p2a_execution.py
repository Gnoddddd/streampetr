"""Execution-only helpers for CARE-3D P2-A formal extraction.

These helpers do not change the frozen association cost, matching rule, cohort,
labels, train-only selection, validation gate, or any P0/P1 artifact.  They only
support deterministic scene sharding, share immutable dataset metadata across
fault protocol views, and omit train-only oracle diagnostics that are not
consumed by configuration selection.
"""

from __future__ import annotations

import copy
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor


EXECUTION_POLICY = "p2a_train_parallel_shared_infos_fast_diagnostics_v2"


def shard_scene_frame(
    frame: pd.DataFrame,
    *,
    num_shards: int,
    shard_index: int,
    max_scenes: int | None = None,
) -> pd.DataFrame:
    """Return one deterministic, disjoint strided shard preserving row order."""
    num_shards = int(num_shards)
    shard_index = int(shard_index)
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    if max_scenes is not None and int(max_scenes) < 0:
        raise ValueError("max_scenes must be non-negative")

    output = frame.iloc[shard_index::num_shards].reset_index(drop=True)
    if max_scenes is not None:
        output = output.iloc[: int(max_scenes)].reset_index(drop=True)
    return output


def build_shared_protocol_dataset(clean_dataset, config, schedule):
    """Create a fault-protocol dataset view without reloading annotation infos.

    ``run_bd_temporal_support_p0.protocol_dataset`` rebuilds the entire dataset
    for every protocol.  On full nuScenes that duplicates the 599 MiB annotation
    pickle as a much larger Python object graph per protocol and per worker.

    The clean dataset already owns a fully constructed StreamPETR pipeline whose
    transforms were created through the correct OpenMMLab/plugin registries.
    Re-running a generic ``Compose`` here is both unnecessary and fragile because
    StreamPETR's custom loader registrations are not guaranteed to live in the
    registry imported by a standalone Compose call.  Instead, clone that already
    registered pipeline and reconstruct only ``ApplyPartialObservation`` directly
    from its existing class with the frozen config arguments plus the requested
    schedule.  ``data_infos`` remains shared by identity.

    This helper is execution-only.  It must be validated against the canonical
    builder on excluded engineering frames before formal extraction.
    """
    if schedule is None:
        raise ValueError("shared protocol view requires a non-clean schedule")
    if not hasattr(clean_dataset, "data_infos"):
        raise RuntimeError("clean dataset has no data_infos to share")
    if not hasattr(clean_dataset, "pipeline"):
        raise RuntimeError("clean dataset has no constructed pipeline to clone")

    value = copy.deepcopy(config.data.test)
    nodes = [
        node for node in value.pipeline
        if node.get("type") == "ApplyPartialObservation"
    ]
    if len(nodes) != 1:
        raise RuntimeError(
            f"expected one ApplyPartialObservation, got {len(nodes)}"
        )

    shared_pipeline = copy.deepcopy(clean_dataset.pipeline)
    transforms = getattr(shared_pipeline, "transforms", None)
    if transforms is None:
        raise RuntimeError("constructed clean pipeline has no transforms list")

    partial_indices = [
        index for index, transform in enumerate(transforms)
        if transform.__class__.__name__ == "ApplyPartialObservation"
    ]
    if len(partial_indices) != 1:
        raise RuntimeError(
            "expected one constructed ApplyPartialObservation transform, got "
            f"{len(partial_indices)}"
        )

    index = int(partial_indices[0])
    original_transform = transforms[index]
    kwargs = copy.deepcopy(dict(nodes[0]))
    transform_type = kwargs.pop("type", None)
    if transform_type != "ApplyPartialObservation":
        raise RuntimeError(
            f"unexpected protocol transform type in frozen config: {transform_type!r}"
        )
    kwargs["schedule_file"] = str(schedule)

    # Instantiate the already-registered concrete class directly.  This preserves
    # the canonical constructor semantics (including schedule parsing) without a
    # second registry lookup for unrelated pipeline transforms such as
    # LoadMultiViewImageFromFiles.
    replacement = original_transform.__class__(**kwargs)
    if replacement.__class__ is not original_transform.__class__:
        raise RuntimeError("protocol transform concrete class changed")
    if str(getattr(replacement, "schedule_file", "")) != str(schedule):
        raise RuntimeError("protocol transform did not bind the requested schedule")
    if getattr(replacement, "schedule", None) is None:
        raise RuntimeError("protocol transform did not load its frozen schedule")
    transforms[index] = replacement

    shared = copy.copy(clean_dataset)
    shared.pipeline = shared_pipeline
    shared.test_mode = True

    if shared is clean_dataset:
        raise RuntimeError("shared protocol dataset unexpectedly aliases base object")
    if shared.pipeline is clean_dataset.pipeline:
        raise RuntimeError("shared protocol dataset unexpectedly aliases clean pipeline")
    if shared.data_infos is not clean_dataset.data_infos:
        raise RuntimeError("shared protocol dataset duplicated data_infos")

    # Real mmdet3d datasets implement ``__len__``.  Lightweight unit-test
    # fixtures may not, so fall back to the shared immutable ``data_infos``
    # length rather than making this helper depend on a framework-only protocol.
    try:
        clean_length = len(clean_dataset)
        shared_length = len(shared)
    except TypeError:
        clean_length = len(clean_dataset.data_infos)
        shared_length = len(shared.data_infos)
    if shared_length != clean_length:
        raise RuntimeError("shared protocol dataset length changed")
    return shared


def cheap_train_oracle_diagnostics(
    cost: Tensor,
    oracle_queries: Sequence[int],
) -> Dict[str, np.ndarray]:
    """O(N) oracle diagnostics sufficient for probe-train aggregation.

    The formal probe-train selector consumes only selected query/cost plus
    exact/wrong/unmatched labels. ``assignment_rows`` requires the diagnostic
    keys structurally, but oracle rank and best-wrong margin are never used by
    train selection.  Avoiding a 900-query sort for every track and every one of
    the 15 configurations materially reduces CPU overhead while preserving all
    selection-relevant values exactly.
    """
    if cost.ndim != 2:
        raise ValueError("cost must be [N,Q]")
    oracle = np.asarray(oracle_queries, dtype=np.int64)
    if oracle.ndim != 1 or len(oracle) != int(cost.shape[0]):
        raise ValueError("oracle query vector/cost rows differ")
    if np.any(oracle < 0) or np.any(oracle >= int(cost.shape[1])):
        raise ValueError("oracle query index out of range")

    values = cost.detach().float().cpu().numpy().astype(np.float64, copy=False)
    rows = np.arange(len(oracle), dtype=np.int64)
    oracle_cost = values[rows, oracle] if len(oracle) else np.empty((0,), np.float64)
    eligible = np.isfinite(oracle_cost)

    return {
        "oracle_cost": np.where(eligible, oracle_cost, np.nan).astype(np.float64, copy=False),
        "oracle_rank": np.full(len(oracle), int(cost.shape[1]) + 1, dtype=np.int64),
        "correct_vs_best_wrong_margin": np.full(len(oracle), np.nan, dtype=np.float64),
        "oracle_geometry_eligible": eligible.astype(bool, copy=False),
    }


def assert_shards_partition(frame: pd.DataFrame, *, num_shards: int) -> None:
    """Assert deterministic shards are exhaustive and pairwise disjoint."""
    marker = np.arange(len(frame), dtype=np.int64)
    seen = []
    for shard_index in range(int(num_shards)):
        positions = marker[shard_index:: int(num_shards)]
        seen.extend(positions.tolist())
    if sorted(seen) != marker.tolist() or len(seen) != len(set(seen)):
        raise RuntimeError("P2-A execution shards are not an exact partition")
