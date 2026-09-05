#!/usr/bin/env python3
"""Compare disabled objective inference with frozen B0 protocol outputs."""

import csv
from pathlib import Path

import mmcv
import torch


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/stage4/gt_query_survival_audit"
CURRENT = ROOT / "outputs/stage4/hard_positive_boundary_objective_audit/disabled"
REPORT = ROOT / "reports/stage4/hard_positive_boundary_objective_audit"


def compare(left, right):
    if hasattr(left, "tensor") or hasattr(right, "tensor"):
        return compare(left.tensor, right.tensor)
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or left.shape != right.shape:
            return float("inf"), 0
        return (float((left.cpu() - right.cpu()).abs().max()) if left.numel() else 0.0), 1
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right): return float("inf"), 0
        values = [compare(left[k], right[k]) for k in left]
    elif isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right): return float("inf"), 0
        values = [compare(a, b) for a, b in zip(left, right)]
    else:
        return (0.0 if left == right else float("inf")), 1
    return max((v[0] for v in values), default=0.0), sum(v[1] for v in values)


def main():
    rows = []
    for group in ("clean", "dark_back", "blur_back", "crash_back"):
        difference, leaves = compare(
            mmcv.load(str(BASE / group / "predictions.pkl")),
            mmcv.load(str(CURRENT / group / "predictions.pkl")),
        )
        rows.append({"protocol": group, "leaves": leaves,
                     "max_abs_diff": difference, "exact": difference == 0.0})
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / "disabled_invariance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    if any(not row["exact"] for row in rows):
        raise RuntimeError(f"disabled objective diverged: {rows}")


if __name__ == "__main__": main()

