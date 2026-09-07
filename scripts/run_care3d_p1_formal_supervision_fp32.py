#!/usr/bin/env python3
"""Run formal P1 train/val supervision with the FP32 cache policy."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/care3d/p1_sparse_evidence_router"


def main() -> None:
    gate_path = REPORT / "engineering_smoke_gate.json"
    if not gate_path.exists():
        raise RuntimeError("run scripts/run_care3d_p1_smoke_fp32.sh first")
    gate = json.loads(gate_path.read_text())
    if gate.get("passed") is not True:
        raise RuntimeError("P1 FP32 engineering smoke gate did not pass")

    command = [
        sys.executable,
        str(ROOT / "scripts/export_care3d_p1_supervision_fp32.py"),
        "--formal-train-val",
        "--device",
        "cuda:0",
    ]
    subprocess.run(command, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
