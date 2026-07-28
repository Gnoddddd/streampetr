#!/usr/bin/env python3
"""Project-root training entrypoint that delegates to pinned StreamPETR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import PROJECT_ROOT, run_upstream


def parse_args():
    parser = argparse.ArgumentParser(description="Train Evidence3D on StreamPETR")
    parser.add_argument(
        "--config",
        default="configs/evidence_conserving/mini_debug.py",
        help="Config path relative to the Evidence3D project root",
    )
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument(
        "upstream_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are passed to StreamPETR tools/train.py",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = (PROJECT_ROOT / args.config).resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir
        else PROJECT_ROOT / "outputs" / config.stem
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    upstream = [str(config), "--work-dir", str(work_dir), "--seed", str(args.seed)]
    if args.resume_from:
        upstream += ["--resume-from", str(Path(args.resume_from).expanduser().resolve())]
    if args.no_validate:
        upstream.append("--no-validate")
    extra = list(args.upstream_args)
    if extra and extra[0] == "--":
        extra = extra[1:]
    upstream.extend(extra)
    return run_upstream("tools/train.py", upstream)


if __name__ == "__main__":
    sys.exit(main())
