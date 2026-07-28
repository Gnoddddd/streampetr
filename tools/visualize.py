#!/usr/bin/env python3
"""Visualize evidence trace summaries produced during training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", help="Path to evidence_trace.jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    path = Path(args.trace)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No trace records in {path}")
    x = [record["iter"] for record in records]
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in ("mean_observability", "mean_uncertainty", "mean_existence", "mean_novelty"):
        if key in records[0]:
            ax.plot(x, [record[key] for record in records], label=key)
    ax.set_xlabel("iteration")
    ax.set_ylabel("value")
    ax.set_title("Evidence3D training trace")
    ax.legend()
    ax.grid(True, alpha=0.3)
    output = Path(args.output) if args.output else path.with_suffix(".png")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
