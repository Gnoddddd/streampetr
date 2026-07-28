"""Shared wrapper utilities for running the pinned StreamPETR checkout."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAM_REPO = PROJECT_ROOT / "repos" / "StreamPETR"
MMDET3D_REPO = STREAM_REPO / "mmdetection3d"


def ensure_wsl_project() -> None:
    if not STREAM_REPO.is_dir():
        raise FileNotFoundError(
            f"StreamPETR not found at {STREAM_REPO}. Run scripts/clone_streampetr.sh first."
        )
    if not MMDET3D_REPO.is_dir():
        raise FileNotFoundError(
            f"MMDetection3D not found at {MMDET3D_REPO}. Run scripts/install_streampetr_env.sh."
        )


def runtime_env(extra: Optional[Mapping[str, str]] = None) -> dict:
    env = os.environ.copy()
    paths = [str(PROJECT_ROOT), str(STREAM_REPO), str(MMDET3D_REPO)]
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["EVIDENCE3D_ROOT"] = str(PROJECT_ROOT)
    if extra:
        env.update({str(key): str(value) for key, value in extra.items()})
    return env


def run_upstream(
    relative_tool: str,
    args: Iterable[str],
    extra_env: Optional[Mapping[str, str]] = None,
) -> int:
    ensure_wsl_project()
    command = [sys.executable, str(STREAM_REPO / relative_tool), *map(str, args)]
    completed = subprocess.run(
        command,
        cwd=str(STREAM_REPO),
        env=runtime_env(extra_env),
        check=False,
    )
    return int(completed.returncode)
