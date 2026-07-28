#!/usr/bin/env python3
"""Fail S2.3 candidates whose innovation diagnostics collapse."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("component_summary", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    failures = []
    with args.component_summary.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            component = row.get("component", "")
            mean = float(row.get("mean", "nan"))
            std = float(row.get("std", "nan"))
            p01 = float(row.get("p01", "nan"))
            p99 = float(row.get("p99", "nan"))
            reason = None
            if component == "source_innovation" and p99 <= 1e-6:
                reason = "source_innovation_always_zero"
            elif component == "feature_innovation" and std <= 1e-6:
                reason = "feature_innovation_variance_zero"
            elif component == "temporal_reacquisition" and mean > 0.95:
                reason = "temporal_reacquisition_never_falls"
            elif component == "conflict" and mean > 0.75:
                reason = "conflict_persistently_high"
            elif component == "combined_reliability" and p99 < 0.05:
                reason = "reliability_persistently_low"
            elif component in (
                "positive_novelty_gain",
                "negative_novelty_gain",
            ) and p99 <= 1e-6:
                reason = f"{component}_always_zero"
            elif component == "novelty_gain" and (
                p99 - p01 <= 1e-6 or p01 > 0.99
            ):
                reason = "effective_innovation_collapsed"
            if reason:
                failures.append({**row, "failure_reason": reason})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        sorted({key for row in failures for key in row})
        if failures
        else ["status"]
    )
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures or [{"status": "pass"}])
    if failures:
        raise SystemExit(f"{len(failures)} innovation collapse checks failed")


if __name__ == "__main__":
    main()
