#!/usr/bin/env python3
"""Execution-only wrapper for faster CARE-3D P2-A probe-train extraction.

Scientific behavior remains defined by ``export_care3d_p2a_association.py``.
This wrapper only:
1. partitions the 419 frozen probe-train scenes into deterministic disjoint
   strided shards;
2. defers shared progress-manifest writes while parallel workers are active;
3. replaces expensive train-only oracle rank/margin diagnostics with an O(N)
   structural equivalent because probe-train selection never consumes rank or
   margin.

Probe-val and probe-test are intentionally unsupported here.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

from analysis.care3d_p2a_execution import (
    EXECUTION_POLICY,
    assert_shards_partition,
    cheap_train_oracle_diagnostics,
    shard_scene_frame,
)
import scripts.export_care3d_p2a_association as base


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index out of range")
    if args.max_scenes is not None and args.max_scenes < 0:
        raise ValueError("--max-scenes must be non-negative")

    original_parse_args = base.parse_args
    original_selected_scenes = base.selected_scenes
    original_update_progress = base.update_progress
    original_oracle_diagnostics = base.oracle_diagnostics

    frozen_args = SimpleNamespace(
        engineering_scene=False,
        split="probe_train",
        max_scenes=None,
        device=args.device,
    )

    def wrapped_parse_args():
        return frozen_args

    def wrapped_selected_scenes(inner_args):
        frame = original_selected_scenes(inner_args)
        assert_shards_partition(frame, num_shards=args.num_shards)
        return shard_scene_frame(
            frame,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            max_scenes=args.max_scenes,
        )

    # Parallel workers never mutate the shared progress manifest.  The parent
    # runner performs one canonical refresh only after every worker exits 0.
    def deferred_update_progress(_validation):
        return None

    base.parse_args = wrapped_parse_args
    base.selected_scenes = wrapped_selected_scenes
    base.update_progress = deferred_update_progress
    base.oracle_diagnostics = cheap_train_oracle_diagnostics
    try:
        print(json.dumps({
            "event": "P2A_FAST_TRAIN_WORKER_START",
            "execution_policy": EXECUTION_POLICY,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "max_scenes": args.max_scenes,
            "split": "probe_train",
            "probe_val_read": False,
            "probe_test_read": False,
        }, sort_keys=True), flush=True)
        base.main()
        print(json.dumps({
            "event": "P2A_FAST_TRAIN_WORKER_COMPLETE",
            "execution_policy": EXECUTION_POLICY,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "probe_val_read": False,
            "probe_test_read": False,
        }, sort_keys=True), flush=True)
    finally:
        base.parse_args = original_parse_args
        base.selected_scenes = original_selected_scenes
        base.update_progress = original_update_progress
        base.oracle_diagnostics = original_oracle_diagnostics


if __name__ == "__main__":
    main()
