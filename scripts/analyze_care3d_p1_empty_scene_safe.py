#!/usr/bin/env python3
"""Run the frozen CARE-3D P1 analysis with strict zero-object-scene I/O handling.

The formal evaluator can legitimately produce a probe-test scene with
``object_rows == 0``.  In that case ``pd.DataFrame([]).to_csv(...)`` creates a
headerless CSV, and the frozen analyzer's unconditional ``pd.read_csv`` raises
``pandas.errors.EmptyDataError`` before any metric or gate is computed.

This wrapper changes only file loading semantics:

- an empty ``*.objects.csv`` is accepted only when its completion marker says
  ``object_rows == 0``;
- any marker/file row-count disagreement remains a hard failure;
- frame CSVs are validated against ``frame_rows`` and remain part of the full
  132-scene FP analysis;
- only evaluation markers produced under the validated deployment-shape
  classifier policy are accepted;
- all frozen metrics, bootstrap seeds/repetitions, gate thresholds and decision
  logic remain in ``scripts/analyze_care3d_p1.py`` unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import scripts.analyze_care3d_p1 as _analysis


REPORT = _analysis.REPORT
SCHEMA = _analysis.SCHEMA
CLASSIFIER_EXECUTION_POLICY = "packed_deployment_shape_900_v1"


def _read_counted_csv(path: Path, expected_rows: int, *, kind: str) -> pd.DataFrame:
    """Read one evaluation CSV and enforce its marker-declared row count."""
    expected_rows = int(expected_rows)
    if expected_rows < 0:
        raise RuntimeError(f"negative {kind} row count in marker: {path}")
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        if expected_rows == 0:
            return pd.DataFrame()
        raise RuntimeError(
            f"empty {kind} CSV but marker declares {expected_rows} rows: {path}"
        ) from exc
    if len(frame) != expected_rows:
        raise RuntimeError(
            f"{kind} CSV row-count mismatch for {path}: "
            f"marker={expected_rows}, csv={len(frame)}"
        )
    return frame


def load_evaluation_zero_object_safe(validation: dict):
    """Load all 132 frozen probe-test scenes without inventing empty objects."""
    manifest = pd.read_csv(REPORT / "frozen_scene_manifest.csv")
    scenes = (
        manifest[manifest.split.astype(str) == "probe_test"]
        .scene_token.astype(str)
        .tolist()
    )
    if len(scenes) != 132:
        raise RuntimeError(f"expected 132 frozen probe-test scenes, found {len(scenes)}")

    objects, frames, clean_identity = [], [], []
    zero_object_scenes = []

    for scene in scenes:
        prefix = REPORT / "evaluation/probe_test" / scene
        marker_path = prefix.with_suffix(".complete.json")
        object_path = prefix.with_suffix(".objects.csv")
        frame_path = prefix.with_suffix(".frames.csv")
        if not marker_path.exists() or not object_path.exists() or not frame_path.exists():
            raise RuntimeError(f"missing P1 evaluation scene: {scene}")

        marker = json.loads(marker_path.read_text())
        if not marker.get("complete") or marker.get("schema_version") != SCHEMA:
            raise RuntimeError(f"invalid P1 evaluation marker: {scene}")
        if marker.get("scene_manifest_sha256") != validation["scene_manifest_sha256"]:
            raise RuntimeError(f"P1 evaluation cohort mismatch: {scene}")
        if marker.get("classifier_execution_policy") != CLASSIFIER_EXECUTION_POLICY:
            raise RuntimeError(f"P1 evaluation classifier policy mismatch: {scene}")

        if "object_rows" not in marker or "frame_rows" not in marker:
            raise RuntimeError(f"P1 evaluation marker lacks row counts: {scene}")
        expected_object_rows = int(marker["object_rows"])
        expected_frame_rows = int(marker["frame_rows"])

        object_frame = _read_counted_csv(
            object_path, expected_object_rows, kind="object"
        )
        frame_frame = _read_counted_csv(
            frame_path, expected_frame_rows, kind="frame"
        )

        if expected_object_rows == 0:
            zero_object_scenes.append(scene)
        else:
            objects.append(object_frame)
        if expected_frame_rows == 0:
            raise RuntimeError(f"P1 formal evaluation produced zero frame rows: {scene}")
        frames.append(frame_frame)
        clean_identity.append(bool(marker.get("clean_identity_pass")))

    if not objects:
        raise RuntimeError("P1 probe-test contains no object-level rows")
    if not frames:
        raise RuntimeError("P1 probe-test contains no frame-level rows")

    print(json.dumps({
        "event": "P1_ANALYSIS_ZERO_OBJECT_SCENE_IO_VALIDATED",
        "probe_test_scenes": len(scenes),
        "zero_object_scene_count": len(zero_object_scenes),
        "zero_object_scenes": zero_object_scenes,
        "classifier_execution_policy": CLASSIFIER_EXECUTION_POLICY,
        "metric_logic_changed": False,
        "gate_logic_changed": False,
    }, sort_keys=True), flush=True)

    return (
        pd.concat(objects, ignore_index=True),
        pd.concat(frames, ignore_index=True),
        bool(all(clean_identity)),
    )


def main() -> None:
    _analysis.load_evaluation = load_evaluation_zero_object_safe
    _analysis.main()


if __name__ == "__main__":
    main()
