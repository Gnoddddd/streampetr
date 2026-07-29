#!/usr/bin/env python3
"""Recursively compare paired StreamPETR prediction pickles."""

from __future__ import annotations

import csv
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch


ROOT = Path("outputs/stage3/raydn_screening/disabled_invariance")
OUTPUT = Path("reports/stage3/raydn_screening/disabled_invariance.csv")
PROTOCOLS = (
    "clean_no_corruption",
    "camera_crash_back_5f",
    "camera_crash_back_10f",
    "compound_fog_crash_10f",
)


def compare(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        a = torch.as_tensor(left).detach().cpu()
        b = torch.as_tensor(right).detach().cpu()
        if a.shape != b.shape or a.dtype != b.dtype:
            return 1, float("inf")
        if a.numel() == 0:
            return 0, 0.0
        difference = (a.to(torch.float64) - b.to(torch.float64)).abs()
        return int(torch.count_nonzero(difference).item()), float(difference.max())
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        a = np.asarray(left)
        b = np.asarray(right)
        if a.shape != b.shape or a.dtype != b.dtype:
            return 1, float("inf")
        if a.size == 0:
            return 0, 0.0
        if np.issubdtype(a.dtype, np.number):
            difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
            return int(np.count_nonzero(difference)), float(difference.max())
        return (0, 0.0) if np.array_equal(a, b) else (1, float("inf"))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return 1, float("inf")
        results = [compare(left[key], right[key]) for key in sorted(left)]
    elif (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, (str, bytes))
        and not isinstance(right, (str, bytes))
    ):
        if len(left) != len(right):
            return 1, float("inf")
        results = [compare(a, b) for a, b in zip(left, right)]
    elif hasattr(left, "tensor") and hasattr(right, "tensor"):
        return compare(left.tensor, right.tensor)
    else:
        return (0, 0.0) if left == right else (1, float("inf"))
    return (
        sum(item[0] for item in results),
        max((item[1] for item in results), default=0.0),
    )


def main():
    rows = []
    for pair in ("B0", "M1"):
        for protocol in PROTOCOLS:
            directory = ROOT / pair / protocol
            with (directory / "baseline/predictions.pkl").open("rb") as handle:
                baseline = pickle.load(handle)
            with (directory / "disabled/predictions.pkl").open("rb") as handle:
                disabled = pickle.load(handle)
            mismatch_count, max_difference = compare(baseline, disabled)
            rows.append(
                {
                    "baseline": pair,
                    "protocol": protocol,
                    "mismatch_count": mismatch_count,
                    "max_abs_diff": max_difference,
                    "pass": mismatch_count == 0 and max_difference == 0.0,
                }
            )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    if not all(row["pass"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

