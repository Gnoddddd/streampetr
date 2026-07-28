#!/usr/bin/env python3
"""Check WSL layout, legacy OpenMMLab imports, data, and evidence invariants."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.nuscenes_wrapper import validate_nuscenes_mini_layout
from models.temporal_update import EvidenceConservingTemporalUpdate


def module_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
        return str(getattr(module, "__version__", "imported"))
    except Exception as error:  # diagnosis must report rather than hide failure
        return f"ERROR: {type(error).__name__}: {error}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    versions: Dict[str, str] = {
        "torch": str(torch.__version__),
        "mmcv": module_version("mmcv"),
        "mmdet": module_version("mmdet"),
        "mmseg": module_version("mmseg"),
        "mmdet3d": module_version("mmdet3d"),
        "nuscenes": module_version("nuscenes"),
    }
    plugin_status = "not attempted"
    if (ROOT / "repos/StreamPETR").is_dir():
        try:
            importlib.import_module("evidence3d_plugin")
            plugin_status = "imported"
        except Exception as error:
            plugin_status = f"ERROR: {type(error).__name__}: {error}"

    status = {
        "project_root": str(ROOT),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "wsl_distro": os.environ.get("WSL_DISTRO_NAME"),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python": sys.executable,
        "versions": versions,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "stream_repo": (ROOT / "repos/StreamPETR").is_dir(),
        "mmdet3d_repo": (ROOT / "repos/StreamPETR/mmdetection3d").is_dir(),
        "plugin": plugin_status,
        "data": validate_nuscenes_mini_layout(ROOT / "data/nuscenes-mini"),
    }

    updater = EvidenceConservingTemporalUpdate(gamma=0.9)
    prior_alpha = torch.tensor([5.0])
    prior_beta = torch.tensor([3.0])
    result = updater(
        prior_alpha,
        prior_beta,
        torch.tensor([0.9]),
        torch.tensor([0.1]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
    )
    expected_strength = 0.9 * (prior_alpha + prior_beta - 2.0)
    status["no_observation_conservation"] = bool(
        torch.allclose(result["strength"], expected_strength, atol=1e-6)
        and result["conservation_violation"].max().item() == 0.0
    )
    print(json.dumps(status, indent=2, ensure_ascii=False))

    if args.strict:
        expected_versions = {
            "torch": "1.9.0",
            "mmcv": "1.6.0",
            "mmdet": "2.28.2",
            "mmseg": "0.30.0",
            "mmdet3d": "1.0.0rc6",
        }
        version_ok = all(
            versions[name].startswith(expected)
            for name, expected in expected_versions.items()
        )
        required = [
            status["wsl_distro"] == "Ubuntu-22.04",
            status["conda_env"] == "streampetr",
            status["python_version"].startswith("3.8."),
            status["cuda_available"],
            status["stream_repo"],
            status["mmdet3d_repo"],
            status["plugin"] == "imported",
            version_ok,
            all(status["data"].values()),
            status["no_observation_conservation"],
        ]
        return 0 if all(required) else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
