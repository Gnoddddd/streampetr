#!/usr/bin/env python3
"""Constraint-first ranking for S2.3 candidate metric rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _number(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.metrics.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        hard_pass = (
            _number(row, "clean_map_delta") >= -0.003
            and _number(row, "clean_nds_delta") >= -0.003
            and _number(row, "conservation_violation_ratio") <= 1e-6
            and _number(row, "unsupported_growth_ratio") <= 1e-6
            and _number(row, "source_mass_violation_ratio") <= 1e-6
            and _number(row, "recovery_max_delta") <= 1.0
        )
        row["hard_constraints_pass"] = int(hard_pass)
        row["score_j"] = (
            0.30 * _number(row, "mean_fault_nds_improvement")
            + 0.20 * _number(row, "mean_fault_map_improvement")
            + 0.20 * _number(row, "recovery_mean_reduction")
            + 0.10 * _number(row, "recovery_max_reduction")
            + 0.10 * _number(row, "calibration_improvement")
            + 0.05 * _number(row, "action_response_improvement")
            + 0.05 * _number(row, "clean_performance_improvement")
        )
    rows.sort(
        key=lambda row: (
            int(row["hard_constraints_pass"]),
            float(row["score_j"]),
            _number(row, "worst_protocol_nds"),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows or [{"status": "not_available"}])


if __name__ == "__main__":
    main()
