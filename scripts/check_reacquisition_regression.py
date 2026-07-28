#!/usr/bin/env python3
"""Fail when rescue metrics violate numerical or checkpoint invariants."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch


RUNTIME_KEYS = (
    "pre_gap_strength",
    "pre_gap_presence",
    "pre_gap_uncertainty",
    "pre_gap_source_evidence",
    "gap_active",
    "gap_age",
    "reacquisition_consumed",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tolerance", type=float, default=2e-5)
    args = parser.parse_args()
    failures = []
    with args.summary.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if float(row["residual_abs_max"]) > args.tolerance:
                failures.append(
                    f"{row['experiment']}: conservation residual exceeded"
                )
            if float(row["unsupported_growth_ratio"]) > 0.0:
                failures.append(
                    f"{row['experiment']}: unsupported growth detected"
                )
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        leaked = [
            key
            for key in state
            if any(key.endswith(name) for name in RUNTIME_KEYS)
        ]
        if leaked:
            failures.append(
                "runtime ledger keys leaked into checkpoint: "
                + ", ".join(leaked)
            )
    if failures:
        print("\n".join(failures), file=sys.stderr)
        raise SystemExit(1)
    print("reacquisition regression checks passed")


if __name__ == "__main__":
    main()
