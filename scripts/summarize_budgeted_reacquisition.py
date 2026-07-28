#!/usr/bin/env python3
"""Summarize S2.3 rescue diagnostics from one or more JSONL traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


FIELDS = (
    "restoration_bonus",
    "lost_strength",
    "motion_consistency",
    "source_recovery",
    "reacquisition_gate",
    "conservation_residual",
)
DETECTION_METRICS = ("mAP", "NDS", "mATE", "mASE", "mAOE", "mAVE", "mAAE")


def _flat(value):
    return np.asarray(value).reshape(-1)


def _number(values, statistic):
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return math.nan
    if statistic == "mean":
        return float(array.mean())
    return float(np.quantile(array, statistic))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.traces:
        values = {field: [] for field in FIELDS}
        counts = {
            "queries": 0,
            "reacquired": 0,
            "bonus": 0,
            "violation": 0,
            "unsupported": 0,
        }
        records = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                diagnostics = record.get("diagnostics", {})
                records += 1
                strength = _flat(diagnostics.get("strength", []))
                counts["queries"] += strength.size
                counts["reacquired"] += int(
                    _flat(diagnostics.get("is_reacquired", [])).astype(bool).sum()
                )
                bonus = _flat(diagnostics.get("restoration_bonus", []))
                counts["bonus"] += int((bonus > 1e-8).sum())
                counts["violation"] += int(
                    _flat(
                        diagnostics.get("conservation_violation_mask", [])
                    ).astype(bool).sum()
                )
                counts["unsupported"] += int(
                    _flat(
                        diagnostics.get("unsupported_growth", [])
                    ).astype(bool).sum()
                )
                for field in FIELDS:
                    values[field].extend(
                        _flat(diagnostics.get(field, [])).tolist()
                    )
        queries = max(counts["queries"], 1)
        residual = np.abs(np.asarray(values["conservation_residual"]))
        positive_bonus = [
            value for value in values["restoration_bonus"] if value > 1e-8
        ]
        log_path = path.parents[1] / "evaluation.log"
        log_text = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.is_file()
            else ""
        )
        detection = {}
        for metric in DETECTION_METRICS:
            matches = re.findall(
                rf"{re.escape(metric)}:\s*([-+]?(?:\d+\.\d+|\d+))",
                log_text,
            )
            detection[metric] = (
                float(matches[-1]) if matches else math.nan
            )
        row = {
            "candidate": path.parents[2].name,
            "protocol": path.parents[1].name,
            "experiment": (
                path.parents[2].name + "_" + path.parents[1].name
            ),
            **detection,
            "records": records,
            "queries": counts["queries"],
            "restoration_trigger_count": counts["bonus"],
            "restoration_trigger_ratio": counts["bonus"] / queries,
            "reacquired_count": counts["reacquired"],
            "reacquired_ratio": counts["reacquired"] / queries,
            "bonus_mean": _number(values["restoration_bonus"], "mean"),
            "bonus_p90": _number(values["restoration_bonus"], 0.90),
            "bonus_p99": _number(values["restoration_bonus"], 0.99),
            "positive_bonus_mean": _number(positive_bonus, "mean"),
            "positive_bonus_p90": _number(positive_bonus, 0.90),
            "positive_bonus_p99": _number(positive_bonus, 0.99),
            "lost_strength_mean": _number(values["lost_strength"], "mean"),
            "motion_consistency_mean": _number(
                values["motion_consistency"], "mean"
            ),
            "source_recovery_mean": _number(
                values["source_recovery"], "mean"
            ),
            "residual_mean": _number(
                values["conservation_residual"], "mean"
            ),
            "residual_abs_max": (
                float(residual.max()) if residual.size else math.nan
            ),
            "violation_count": counts["violation"],
            "violation_ratio": counts["violation"] / queries,
            "unsupported_growth_count": counts["unsupported"],
            "unsupported_growth_ratio": counts["unsupported"] / queries,
        }
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows or [{"status": "not_available"}])


if __name__ == "__main__":
    main()
