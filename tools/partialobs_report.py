#!/usr/bin/env python3
"""Build a PartialObs-3D metric report from exported diagnostic arrays.

Input JSON fields are optional. See ``evaluation/example_metric_input.json``.
This tool intentionally stays independent of the nuScenes evaluator so the same
failure diagnostics can be reused across query-based and dense-BEV models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.metrics import (
    evidence_inflation_ratio,
    reacquisition_delay,
    risk_coverage_curve,
    stale_object_persistence,
    unsupported_false_positive_rate,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Diagnostic JSON input")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    input_path = Path(args.input).expanduser().resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    report = {}

    if "is_false_positive" in payload and "supported" in payload:
        report["UFPR"] = unsupported_false_positive_rate(
            payload["is_false_positive"], payload["supported"]
        )
    if "stale_lengths" in payload:
        report["SOP"] = stale_object_persistence(payload["stale_lengths"])
    if "evidence_strength" in payload and "no_new_observation" in payload:
        report["EIR"] = evidence_inflation_ratio(
            payload["evidence_strength"], payload["no_new_observation"]
        )
    if "correct_after_recovery" in payload and "recovery_index" in payload:
        report["RD"] = reacquisition_delay(
            payload["correct_after_recovery"], payload["recovery_index"]
        )
    if "errors" in payload and "uncertainties" in payload:
        curve = risk_coverage_curve(
            payload["errors"],
            payload["uncertainties"],
            steps=int(payload.get("risk_coverage_steps", 20)),
        )
        report["risk_coverage"] = {
            key: np.asarray(value).tolist() for key, value in curve.items()
        }

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_name(input_path.stem + "_report.json")
    )
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
