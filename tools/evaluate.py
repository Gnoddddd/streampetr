#!/usr/bin/env python3
"""Single-GPU non-distributed evaluation wrapper for StreamPETR."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, List, Sequence, Tuple

from mmcv import Config

try:
    from .common import PROJECT_ROOT, STREAM_REPO, ensure_wsl_project, runtime_env
except ImportError:  # Direct ``python tools/evaluate.py`` execution.
    from common import PROJECT_ROOT, STREAM_REPO, ensure_wsl_project, runtime_env


def find_transform_paths(
    value: Any,
    transform_type: str,
    path: Sequence[str] = (),
) -> List[Tuple[str, ...]]:
    """Return every recursive config path matching a transform type."""
    matches: List[Tuple[str, ...]] = []
    if isinstance(value, Mapping):
        if value.get("type") == transform_type:
            matches.append(tuple(path))
        for key, child in value.items():
            matches.extend(
                find_transform_paths(child, transform_type, (*path, str(key)))
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            matches.extend(
                find_transform_paths(child, transform_type, (*path, str(index)))
            )
    return matches


def resolve_protocol_cfg_path(config: Config) -> str:
    """Resolve one unambiguous test-pipeline protocol injection target."""
    try:
        test_config = config.data.test
    except AttributeError as error:
        raise ValueError("Config has no data.test pipeline to inject") from error
    matches = find_transform_paths(
        test_config,
        "ApplyPartialObservation",
        ("data", "test"),
    )
    rendered = [".".join(path) for path in matches]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one ApplyPartialObservation under data.test, "
            f"found {len(matches)}: {rendered}"
        )
    return ".".join((*matches[0], "schedule_file"))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Evidence3D")
    parser.add_argument("--config", default="configs/evidence_conserving/mini_train.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval", default="bbox")
    parser.add_argument("--protocol", default=None)
    parser.add_argument(
        "--reacquisition-diagnostics",
        action="store_true",
        help="Enable the read-only S2.3 reacquisition observer.",
    )
    parser.add_argument("--master-port", type=int, default=29517)
    parser.add_argument("upstream_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_wsl_project()
    config = (PROJECT_ROOT / args.config).resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    env_extra = {}
    if args.protocol:
        protocol = Path(args.protocol).expanduser().resolve()
        if not protocol.is_file():
            raise FileNotFoundError(protocol)
        env_extra["EVIDENCE3D_PROTOCOL"] = str(protocol)
    command = [
        sys.executable,
        str(STREAM_REPO / "tools" / "test.py"),
        str(config),
        str(checkpoint),
        "--launcher",
        "none",
        "--eval",
        args.eval,
    ]
    cfg_options = []
    if args.protocol:
        protocol_cfg_path = resolve_protocol_cfg_path(Config.fromfile(str(config)))
        print(
            "ApplyPartialObservation protocol injection path: "
            f"{protocol_cfg_path}",
            flush=True,
        )
        # The environment variable is retained for trace naming/provenance.
        cfg_options.append(f"{protocol_cfg_path}={protocol}")
    if args.reacquisition_diagnostics:
        cfg_options.append(
            "model.pts_bbox_head.enable_reacquisition_diagnostics=True"
        )
    if cfg_options:
        command.extend(["--cfg-options", *cfg_options])
    extra = list(args.upstream_args)
    if extra and extra[0] == "--":
        extra = extra[1:]
    command.extend(extra)
    return subprocess.run(
        command,
        cwd=str(STREAM_REPO),
        env=runtime_env(env_extra),
        check=False,
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
