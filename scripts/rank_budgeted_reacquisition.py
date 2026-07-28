#!/usr/bin/env python3
"""Apply S2.3 rescue hard gates before ranking B0--B6."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _f(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.metrics.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        hard_pass = (
            _f(row, "clean_map_delta") >= -0.003
            and _f(row, "clean_nds_delta") >= -0.003
            and _f(row, "mean_fault_nds_delta") >= -0.002
            and _f(row, "recovery_mean_delta") <= 0.0
            and _f(row, "recovery_max_delta") <= 1.0
            and _f(row, "clean_bonus_trigger_ratio") <= 1e-3
            and _f(row, "conservation_violation_ratio") <= 1e-6
            and _f(row, "unsupported_growth_ratio") <= 1e-6
        )
        row["hard_constraints_pass"] = int(hard_pass)
        row["rank_score"] = (
            0.45 * _f(row, "mean_fault_nds_delta")
            + 0.25 * _f(row, "mean_fault_map_delta")
            - 0.15 * _f(row, "recovery_mean_delta")
            + 0.10 * _f(row, "clean_nds_delta")
            + 0.05 * _f(row, "clean_map_delta")
        )
    rows.sort(
        key=lambda row: (
            int(row["hard_constraints_pass"]),
            float(row["rank_score"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows or [{"status": "not_available"}])


if __name__ == "__main__":
    main()
