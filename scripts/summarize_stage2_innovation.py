#!/usr/bin/env python3
"""Summarize S2.3 diagnostic JSONL traces without inventing GT metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


FIELDS = (
    "source_innovation",
    "feature_innovation",
    "geometry_innovation",
    "class_semantic_innovation",
    "ternary_semantic_innovation",
    "temporal_reacquisition",
    "combined_reliability",
    "conflict",
    "novelty_gain",
    "positive_novelty_gain",
    "negative_novelty_gain",
)
QUANTILES = (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99)


def _flatten(value) -> np.ndarray:
    array = np.asarray(value)
    return array.reshape(-1)


def _phase(experiment: str, frame_idx: int) -> str:
    lowered = experiment.lower()
    match = re.search(r"(?:_|-)(5|10|20)f(?:_|-|$)", lowered)
    if "clean" in lowered or match is None:
        return "clean"
    duration = int(match.group(1))
    if frame_idx < 3:
        return "clean"
    if frame_idx < 3 + duration:
        return "fault"
    return "recovery"


def _stats(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        result = {"count": 0, "mean": math.nan, "std": math.nan}
        result.update({f"p{int(q * 100):02d}": math.nan for q in QUANTILES})
        return result
    result = {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }
    result.update(
        {
            f"p{int(q * 100):02d}": float(np.quantile(array, q))
            for q in QUANTILES
        }
    )
    return result


def _write(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows or [{"status": "not_available"}])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    records = []
    for path in args.traces:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                diagnostics = record.get("diagnostics", {})
                candidate = (
                    path.parents[2].name
                    if len(path.parents) >= 3
                    else "unknown_candidate"
                )
                experiment = (
                    candidate
                    + "_"
                    + path.stem.replace("_diagnostic_trace", "")
                )
                frame_idx = int(record.get("frame_idx", -1))
                records.append(
                    {
                        "experiment": experiment,
                        "phase": _phase(experiment, frame_idx),
                        "frame_idx": frame_idx,
                        "diagnostics": diagnostics,
                    }
                )

    frame_rows = []
    component_values: Dict[tuple, List[float]] = {}
    for record in records:
        row = {
            "experiment": record["experiment"],
            "phase": record["phase"],
            "frame_idx": record["frame_idx"],
        }
        diagnostics = record["diagnostics"]
        for field in FIELDS:
            values = _flatten(diagnostics.get(field, []))
            finite = values[np.isfinite(values)]
            row[f"{field}_mean"] = (
                float(finite.mean()) if finite.size else math.nan
            )
            component_values.setdefault(
                (record["experiment"], record["phase"], field), []
            ).extend(finite.tolist())
        for flag in (
            "is_reacquired_query",
            "valid_feature_pair",
            "valid_geometry",
        ):
            values = _flatten(diagnostics.get(flag, []))
            row[f"{flag}_count"] = int(values.astype(bool).sum())
        frame_rows.append(row)

    component_rows = []
    for (experiment, phase, field), values in sorted(component_values.items()):
        component_rows.append(
            {
                "experiment": experiment,
                "phase": phase,
                "component": field,
                **_stats(values),
            }
        )
    phase_rows = []
    for experiment in sorted({row["experiment"] for row in frame_rows}):
        for phase in ("clean", "fault", "recovery"):
            selected = [
                row
                for row in frame_rows
                if row["experiment"] == experiment and row["phase"] == phase
            ]
            if not selected:
                continue
            phase_rows.append(
                {
                    "experiment": experiment,
                    "phase": phase,
                    "records": len(selected),
                    **{
                        f"{field}_mean": float(
                            np.nanmean([row[f"{field}_mean"] for row in selected])
                        )
                        for field in FIELDS
                    },
                }
            )
    reacquisition_rows = [
        {
            "experiment": row["experiment"],
            "phase": row["phase"],
            "frame_idx": row["frame_idx"],
            "reacquired_query_count": row["is_reacquired_query_count"],
            "temporal_reacquisition_mean": row[
                "temporal_reacquisition_mean"
            ],
        }
        for row in frame_rows
    ]
    conflict_rows = [
        {
            "experiment": row["experiment"],
            "phase": row["phase"],
            "frame_idx": row["frame_idx"],
            "conflict_mean": row["conflict_mean"],
            "reliability_mean": row["combined_reliability_mean"],
        }
        for row in frame_rows
    ]
    recovery_rows = [
        {
            "experiment": row["experiment"],
            "status": "diagnostic_only_gt_recovery_requires_matched_labels",
        }
        for row in phase_rows
        if row["phase"] == "recovery"
    ]
    calibration_rows = [
        {
            "experiment": experiment,
            "ece": "not_available",
            "brier": "not_available",
            "nll": "not_available",
            "aurc": "not_available",
            "reason": "trace_has_no_strict_query_to_gt_matching_labels",
        }
        for experiment in sorted({row["experiment"] for row in frame_rows})
    ]

    output = args.output_dir
    _write(output / "innovation_frame_summary.csv", frame_rows)
    _write(output / "innovation_component_summary.csv", component_rows)
    _write(output / "innovation_phase_summary.csv", phase_rows)
    _write(output / "innovation_reacquisition_summary.csv", reacquisition_rows)
    _write(output / "innovation_conflict_summary.csv", conflict_rows)
    _write(output / "innovation_calibration_summary.csv", calibration_rows)
    _write(output / "innovation_recovery_summary.csv", recovery_rows)


if __name__ == "__main__":
    main()
