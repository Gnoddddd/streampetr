#!/usr/bin/env python3
"""Format frozen StreamPETR predictions without invoking dataset evaluation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from mmcv import Config

try:
    from .common import PROJECT_ROOT, STREAM_REPO, ensure_wsl_project, runtime_env
    from .evaluate import resolve_protocol_cfg_path
except ImportError:
    from common import PROJECT_ROOT, STREAM_REPO, ensure_wsl_project, runtime_env
    from evaluate import resolve_protocol_cfg_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--protocol")
    parser.add_argument("--trace-dir")
    parser.add_argument(
        "--camera-attention-dir",
        help="Export prediction-invariant Camera-to-Query attention traces.",
    )
    parser.add_argument(
        "--gt-query-survival-dir",
        help="Export prediction-invariant per-layer GT-query survival traces.",
    )
    parser.add_argument(
        "--trace-layers",
        action="store_true",
        help="Export prediction-invariant per-layer candidates for audit.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--json-prefix", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_wsl_project()
    config_path = (PROJECT_ROOT / args.config).resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    trace_dir = (
        Path(args.trace_dir).expanduser().resolve()
        if args.trace_dir
        else None
    )
    camera_attention_dir = (
        Path(args.camera_attention_dir).expanduser().resolve()
        if args.camera_attention_dir
        else None
    )
    gt_query_survival_dir = (
        Path(args.gt_query_survival_dir).expanduser().resolve()
        if args.gt_query_survival_dir
        else None
    )
    output = Path(args.out).expanduser().resolve()
    json_prefix = Path(args.json_prefix).expanduser().resolve()
    info = (
        PROJECT_ROOT
        / "data/nuscenes-mini"
        / f"nuscenes2d_temporal_infos_{args.split}.pkl"
    ).resolve()
    for path in (config_path, checkpoint, info):
        if not path.is_file():
            raise FileNotFoundError(path)
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
    if camera_attention_dir is not None:
        camera_attention_dir.mkdir(parents=True, exist_ok=True)
    if gt_query_survival_dir is not None:
        gt_query_survival_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_prefix.parent.mkdir(parents=True, exist_ok=True)

    config = Config.fromfile(str(config_path))
    options = [f"data.test.ann_file={info}"]
    env = {}
    if trace_dir is not None:
        env["COUNTERFACTUAL_TRACE_DIR"] = str(trace_dir)
        if args.trace_layers:
            env["COUNTERFACTUAL_TRACE_LAYERS"] = "1"
    if camera_attention_dir is not None:
        env["CAMERA_ATTENTION_TRACE_DIR"] = str(camera_attention_dir)
    if gt_query_survival_dir is not None:
        env["GT_QUERY_SURVIVAL_TRACE_DIR"] = str(gt_query_survival_dir)
    if args.protocol:
        protocol = Path(args.protocol).expanduser().resolve()
        if not protocol.is_file():
            raise FileNotFoundError(protocol)
        options.append(f"{resolve_protocol_cfg_path(config)}={protocol}")
        env["EVIDENCE3D_PROTOCOL"] = str(protocol)

    command = [
        sys.executable,
        str(STREAM_REPO / "tools/test.py"),
        str(config_path),
        str(checkpoint),
        "--launcher",
        "none",
        "--seed",
        "2026",
        "--deterministic",
        "--out",
        str(output),
        "--format-only",
        "--eval-options",
        f"jsonfile_prefix={json_prefix}",
        "--cfg-options",
        *options,
    ]
    print(" ".join(command), flush=True)
    return subprocess.run(
        command,
        cwd=str(STREAM_REPO),
        env=runtime_env(env),
        check=False,
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
