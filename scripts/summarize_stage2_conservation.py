#!/usr/bin/env python3
"""Summarize S2.1 per-query conservation diagnostics into CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List, Tuple


def _flatten(values) -> Iterable:
    if isinstance(values, list):
        for value in values:
            yield from _flatten(value)
    else:
        yield values


def _read_trace(experiment: str, path: Path) -> dict:
    records = 0
    queries = 0
    residual_sum = 0.0
    residual_abs_max = 0.0
    violation_count = 0
    unsupported_growth_count = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            diagnostics = record.get("diagnostics", {})
            required = (
                "conservation_residual",
                "conservation_violation_mask",
                "unsupported_growth",
            )
            missing = [key for key in required if key not in diagnostics]
            if missing:
                raise ValueError(
                    f"{path}:{line_number} missing diagnostics {missing}"
                )
            residuals = [
                float(value)
                for value in _flatten(
                    diagnostics["conservation_residual"]
                )
            ]
            violations = [
                bool(value)
                for value in _flatten(
                    diagnostics["conservation_violation_mask"]
                )
            ]
            unsupported = [
                bool(value)
                for value in _flatten(
                    diagnostics["unsupported_growth"]
                )
            ]
            if not (
                len(residuals) == len(violations) == len(unsupported)
            ):
                raise ValueError(
                    f"{path}:{line_number} diagnostic lengths do not match"
                )
            records += 1
            queries += len(residuals)
            residual_sum += sum(residuals)
            residual_abs_max = max(
                residual_abs_max,
                max((abs(value) for value in residuals), default=0.0),
            )
            violation_count += sum(violations)
            unsupported_growth_count += sum(unsupported)

    if records == 0 or queries == 0:
        raise ValueError(f"{path} contains no usable diagnostic records")
    return {
        "experiment": experiment,
        "records": records,
        "queries": queries,
        "residual_mean": residual_sum / queries,
        "residual_abs_max": residual_abs_max,
        "violation_count": violation_count,
        "violation_ratio": violation_count / queries,
        "unsupported_growth_count": unsupported_growth_count,
        "unsupported_growth_ratio": unsupported_growth_count / queries,
    }


def _parse_trace(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("trace must be EXPERIMENT=PATH")
    experiment, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not experiment or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid trace: {value}")
    return experiment, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace",
        action="append",
        required=True,
        type=_parse_trace,
        help="EXPERIMENT=PATH; may be supplied more than once",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows: List[dict] = [
        _read_trace(experiment, path)
        for experiment, path in args.trace
    ]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "records",
        "queries",
        "residual_mean",
        "residual_abs_max",
        "violation_count",
        "violation_ratio",
        "unsupported_growth_count",
        "unsupported_growth_ratio",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
