#!/usr/bin/env python3
"""Recursively compare serialized MMDetection predictions."""

import argparse
import csv
from pathlib import Path

import mmcv
import numpy as np
import torch


def leaves(value):
    if torch.is_tensor(value):
        yield value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        yield value
    elif hasattr(value, "tensor") and torch.is_tensor(value.tensor):
        yield value.tensor.detach().cpu().numpy()
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from leaves(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from leaves(item)
    elif isinstance(value, (int, float, bool, np.number)):
        yield np.asarray(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for protocol in (
        "clean_no_corruption",
        "camera_crash_back_5f",
        "camera_crash_back_10f",
        "compound_fog_crash_10f",
    ):
        baseline = mmcv.load(Path(args.root) / "b0" / protocol / "predictions.pkl")
        disabled = mmcv.load(Path(args.root) / "disabled" / protocol / "predictions.pkl")
        baseline_leaves = list(leaves(baseline))
        disabled_leaves = list(leaves(disabled))
        if len(baseline_leaves) != len(disabled_leaves):
            raise RuntimeError(f"{protocol}: prediction structure differs")
        max_diff = 0.0
        tensor_count = 0
        for left, right in zip(baseline_leaves, disabled_leaves):
            if left.shape != right.shape:
                raise RuntimeError(f"{protocol}: prediction shape differs")
            if np.issubdtype(left.dtype, np.number):
                tensor_count += 1
                if left.size:
                    max_diff = max(
                        max_diff,
                        float(np.max(np.abs(left.astype(float) - right.astype(float)))),
                    )
            elif not np.array_equal(left, right):
                raise RuntimeError(f"{protocol}: nonnumeric value differs")
        rows.append(
            dict(
                protocol=protocol,
                records=len(baseline),
                compared_tensors=tensor_count,
                max_abs_diff=max_diff,
                exact=max_diff == 0.0,
            )
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    print("max_abs_diff", max(row["max_abs_diff"] for row in rows))


if __name__ == "__main__":
    main()
