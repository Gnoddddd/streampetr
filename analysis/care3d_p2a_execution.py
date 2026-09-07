"""Execution-only helpers for CARE-3D P2-A formal extraction.

These helpers do not change the frozen association cost, matching rule, cohort,
labels, train-only selection, validation gate, or any P0/P1 artifact.  They only
support deterministic scene sharding and omit train-only oracle diagnostics
that are not consumed by configuration selection.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor


EXECUTION_POLICY = "p2a_train_parallel_shards_fast_diagnostics_v1"


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
