#!/usr/bin/env python3
"""Run formal P1 train/val supervision only after the full smoke gate passes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/care3d/p1_sparse_evidence_router"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-scenes", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gate_path = REPORT / "engineering_smoke_gate.json"
    progress_path = REPORT / "progress_manifest.json"
    if not gate_path.exists() or not progress_path.exists():
        raise RuntimeError("P1 engineering smoke gate has not been completed")
    gate = json.loads(gate_path.read_text())
    progress = json.loads(progress_path.read_text())
    if gate.get("passed") is not True:
        raise RuntimeError("P1 formal supervision locked: engineering smoke failed")
    if progress.get("stages", {}).get("engineering_smoke") != "PASSED":
        raise RuntimeError("P1 formal supervision locked: progress smoke is not PASSED")
    command = [
        sys.executable,
        str(ROOT / "scripts/export_care3d_p1_supervision.py"),
        "--formal-train-val",
        "--device",
        str(args.device),
    ]
    if args.max_scenes is not None:
        command += ["--max-scenes", str(int(args.max_scenes))]
    print("executing:", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
