#!/usr/bin/env python3
"""Summarize S2.2 per-query source-ledger diagnostic JSONL traces."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _write_csv(path: Path, rows: Iterable[Dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _events(protocol_file: str, scene_token: str) -> List[Dict]:
    if not protocol_file:
        return []
    payload = json.loads(Path(protocol_file).read_text(encoding="utf-8"))
    scenes = payload.get("scenes", {})
    return list(scenes.get("*", [])) + list(scenes.get(scene_token, []))


def _phase(record: Dict) -> str:
    events = _events(
        str(record.get("protocol_file") or ""),
        str(record.get("scene_token", "")),
    )
    frame = int(record.get("frame_idx", -1))
    if any(
        int(event["start_frame"]) <= frame <= int(event["end_frame"])
        for event in events
    ):
        return "fault"
    if events and frame > max(int(event["end_frame"]) for event in events):
        return "recovery"
    return "clean"


def _array(diagnostics: Dict, key: str, dimensions: int) -> np.ndarray:
    value = np.asarray(diagnostics[key], dtype=np.float64)
    while value.ndim > dimensions and value.shape[0] == 1:
        value = value[0]
    if value.ndim != dimensions:
        raise ValueError(
            f"{key} must reduce to {dimensions} dimensions, got {value.shape}"
        )
    return value


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def summarize(trace_root: Path, output_dir: Path) -> None:
    traces = sorted(trace_root.rglob("*_diagnostic_trace.jsonl"))
    if not traces:
        raise FileNotFoundError(f"No diagnostic traces under {trace_root}")

    phase_acc = defaultdict(lambda: defaultdict(list))
    camera_acc = defaultdict(lambda: defaultdict(list))
    frame_rows: List[Dict] = []
    mass_acc = defaultdict(lambda: defaultdict(list))

    for trace in traces:
        experiment = trace.stem.replace("_diagnostic_trace", "")
        for line in trace.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            diagnostics = record["diagnostics"]
            if "source_evidence" not in diagnostics:
                continue
            phase = _phase(record)
            current = _array(diagnostics, "current_source_vector", 2)
            increment = _array(diagnostics, "current_source_increment", 2)
            evidence = _array(diagnostics, "source_evidence", 2)
            provenance = _array(diagnostics, "provenance", 2)
            strength = _array(diagnostics, "source_strength", 1)
            residual = _array(diagnostics, "source_mass_residual", 1)
            violation = _array(diagnostics, "source_mass_violation", 1)
            zero_increment = _array(
                diagnostics,
                "zero_source_increment",
                1,
            )
            camera_names = list(diagnostics["source_camera_names"])
            if current.shape[1] != len(camera_names):
                raise ValueError("source_camera_names does not match source shape")

            phase_key = (experiment, phase)
            phase_acc[phase_key]["records"].append(1)
            phase_acc[phase_key]["queries"].append(int(strength.size))
            phase_acc[phase_key]["source_strength"].extend(strength.tolist())

            for index, camera_name in enumerate(camera_names):
                key = (experiment, phase, index, camera_name)
                camera_acc[key]["records"].append(1)
                camera_acc[key]["queries"].append(int(strength.size))
                camera_acc[key]["current"].extend(current[:, index].tolist())
                camera_acc[key]["increment"].extend(increment[:, index].tolist())
                camera_acc[key]["evidence"].extend(evidence[:, index].tolist())
                camera_acc[key]["provenance"].extend(
                    provenance[:, index].tolist()
                )
                camera_acc[key]["strength"].extend(strength.tolist())

            frame_rows.append(
                {
                    "experiment": experiment,
                    "scene_token": record.get("scene_token", ""),
                    "frame_idx": record.get("frame_idx", -1),
                    "phase": phase,
                    "queries": int(strength.size),
                    "source_strength_mean": float(strength.mean()),
                    "source_mass_residual_abs_max": float(
                        np.abs(residual).max()
                    ),
                    "source_mass_violation_ratio": float(violation.mean()),
                }
            )
            mass_acc[experiment]["records"].append(1)
            mass_acc[experiment]["queries"].append(int(strength.size))
            mass_acc[experiment]["residual"].extend(residual.tolist())
            mass_acc[experiment]["violation"].extend(violation.tolist())
            mass_acc[experiment]["zero_increment"].extend(
                zero_increment.tolist()
            )

    phase_rows = [
        {
            "experiment": experiment,
            "phase": phase,
            "records": sum(values["records"]),
            "queries": sum(values["queries"]),
            "source_strength_mean": _mean(values["source_strength"]),
        }
        for (experiment, phase), values in sorted(phase_acc.items())
    ]
    camera_rows = [
        {
            "experiment": experiment,
            "phase": phase,
            "camera_index": index,
            "camera_name": name,
            "records": sum(values["records"]),
            "queries": sum(values["queries"]),
            "current_source_mean": _mean(values["current"]),
            "current_increment_mean": _mean(values["increment"]),
            "source_evidence_mean": _mean(values["evidence"]),
            "provenance_mean": _mean(values["provenance"]),
            "source_strength_mean": _mean(values["strength"]),
        }
        for (experiment, phase, index, name), values in sorted(
            camera_acc.items()
        )
    ]
    mass_rows = []
    for experiment, values in sorted(mass_acc.items()):
        residual = np.asarray(values["residual"], dtype=np.float64)
        violation = np.asarray(values["violation"], dtype=np.float64)
        zero_increment = np.asarray(
            values["zero_increment"], dtype=np.float64
        )
        mass_rows.append(
            {
                "experiment": experiment,
                "records": sum(values["records"]),
                "queries": sum(values["queries"]),
                "residual_mean": float(residual.mean()),
                "residual_abs_max": float(np.abs(residual).max()),
                "violation_count": int(violation.sum()),
                "violation_ratio": float(violation.mean()),
                "zero_source_increment_count": int(zero_increment.sum()),
                "zero_source_increment_ratio": float(zero_increment.mean()),
            }
        )

    _write_csv(
        output_dir / "source_phase_summary.csv",
        phase_rows,
        list(phase_rows[0]),
    )
    _write_csv(
        output_dir / "source_camera_summary.csv",
        camera_rows,
        list(camera_rows[0]),
    )
    _write_csv(
        output_dir / "source_frame_summary.csv",
        frame_rows,
        list(frame_rows[0]),
    )
    _write_csv(
        output_dir / "source_mass_conservation_summary.csv",
        mass_rows,
        list(mass_rows[0]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summarize(args.trace_root, args.output_dir)


if __name__ == "__main__":
    main()
