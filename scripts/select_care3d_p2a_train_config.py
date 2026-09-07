#!/usr/bin/env python3
"""Freeze the unique global P2-A association config from probe-train only."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from analysis.care3d_p2a_association import (
    P2A_QUERY_COLLISION_POLICY,
    association_grid,
    select_global_config,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/care3d/p2a_online_query_association"
SCHEMA = 1


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def main() -> None:
    validation = json.loads((REPORT / "source_validation.json").read_text())
    if validation.get("status") != "VALIDATED_BEFORE_P2A_FORWARD":
        raise RuntimeError("P2-A source validation is not frozen")
    if validation.get("probe_test_read") is not False:
        raise RuntimeError("P2-A selection cannot run after probe-test access")
    progress_path = REPORT / "progress_manifest.json"
    progress = json.loads(progress_path.read_text())
    if progress.get("probe_test_read") is not False:
        raise RuntimeError("P2-A progress indicates probe-test leakage")

    manifest = pd.read_csv(REPORT / "frozen_scene_manifest.csv")
    train_scenes = set(
        manifest[manifest.split.astype(str) == "probe_train"].scene_token.astype(str)
    )
    if len(train_scenes) != 419:
        raise RuntimeError("P2-A probe-train scene count changed")

    directory = REPORT / "incremental/probe_train"
    summaries = []
    observed = set()
    for scene in sorted(train_scenes):
        marker_path = directory / f"{scene}.complete.json"
        summary_path = directory / f"{scene}.train_summary.csv"
        if not marker_path.exists() or not summary_path.exists():
            raise RuntimeError(f"missing P2-A probe-train scene: {scene}")
        marker = json.loads(marker_path.read_text())
        valid = (
            marker.get("complete")
            and marker.get("schema_version") == SCHEMA
            and marker.get("scene_manifest_sha256") == validation["scene_manifest_sha256"]
            and marker.get("split") == "probe_train"
            and marker.get("query_collision_policy") == P2A_QUERY_COLLISION_POLICY
            and marker.get("probe_test_read") is False
        )
        if not valid:
            raise RuntimeError(f"invalid P2-A probe-train marker: {scene}")
        frame = pd.read_csv(summary_path)
        expected_rows = len(association_grid()) * 3
        if len(frame) != expected_rows:
            raise RuntimeError(
                f"P2-A train summary row count changed for {scene}: {len(frame)}"
            )
        summaries.append(frame)
        observed.add(scene)
    if observed != train_scenes:
        raise RuntimeError("P2-A probe-train coverage is incomplete")

    merged = pd.concat(summaries, ignore_index=True)
    result = select_global_config(merged)
    selection = {
        "schema_version": SCHEMA,
        "status": "P2A_GLOBAL_ASSOCIATION_CONFIG_FROZEN",
        "fit_split": "probe_train",
        "fit_scenes": 419,
        "protocols": ["blur_back", "crash_back", "dark_back"],
        "global_not_protocol_specific": True,
        "grid_size": 15,
        "selected": result["selected"],
        "selection_rule": result["selection_rule"],
        "probe_val_read": False,
        "probe_test_read": False,
        "scene_manifest_sha256": validation["scene_manifest_sha256"],
    }
    selection_path = REPORT / "selection.json"
    if selection_path.exists():
        previous = json.loads(selection_path.read_text())
        if previous != selection:
            raise RuntimeError(
                "P2-A selection was already frozen and the recomputed winner differs"
            )
    else:
        atomic_json(selection_path, selection)

    atomic_frame(REPORT / "train_config_ranking.csv", pd.DataFrame(result["ranking"]))
    atomic_frame(
        REPORT / "train_config_protocol_metrics.csv",
        pd.DataFrame(result["per_protocol"]),
    )

    progress["status"] = "P2A_CONFIG_FROZEN_VAL_EXTRACTION_ELIGIBLE"
    progress["stages"]["config_selection"] = "COMPLETE"
    progress["stages"]["probe_val_extraction"] = "ELIGIBLE"
    progress["probe_test_read"] = False
    atomic_json(progress_path, progress)
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
