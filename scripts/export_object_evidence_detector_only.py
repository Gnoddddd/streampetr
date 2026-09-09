#!/usr/bin/env python3
"""Strip every OE-PG training component from a checkpoint."""

from __future__ import annotations

import argparse
import importlib
import runpy
import sys
from pathlib import Path
from typing import Dict, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAINING_COMPONENTS = {"object_evidence", "pg_projector", "teacher", "adapter"}


def detector_only_state_dict(
    state_dict: Mapping[str, torch.Tensor], detector_prefixes=("detector.",)
) -> Dict[str, torch.Tensor]:
    """Unwrap a training wrapper and remove all training-only parameter namespaces."""
    normalized = {}
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith("module.") else key
        normalized[clean_key] = value
    wrapped_prefix = next(
        (prefix for prefix in detector_prefixes if any(key.startswith(prefix) for key in normalized)),
        None,
    )
    result = {}
    for key, value in normalized.items():
        if wrapped_prefix is not None:
            if not key.startswith(wrapped_prefix):
                continue
            key = key[len(wrapped_prefix):]
        if any(part in TRAINING_COMPONENTS for part in key.split(".")):
            continue
        if key in result:
            raise ValueError(f"checkpoint key collision after stripping: {key}")
        result[key] = value
    if not result:
        raise ValueError("no detector parameters found")
    return result


def _build_streampetr(config_path: str):
    stream_root = ROOT / "repos/StreamPETR"
    sys.path.insert(0, str(stream_root))
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings
    from mmdet3d.models import build_model

    config = Config.fromfile(config_path)
    if config.get("custom_imports"):
        import_modules_from_strings(**config.custom_imports)
    return build_model(
        config.model, train_cfg=config.get("train_cfg"), test_cfg=config.get("test_cfg")
    )


def _build_from_factory(specification: str):
    module_name, function_name = specification.split(":", 1)
    return getattr(importlib.import_module(module_name), function_name)()


def _build_bevdepth(config_path: str):
    sys.path.insert(0, str(ROOT / "repos/BEVDepth"))
    values = runpy.run_path(config_path)
    if "build_model" in values:
        return values["build_model"]()
    from bevdepth.models.base_bev_depth import BaseBEVDepth

    try:
        return BaseBEVDepth(values["backbone_conf"], values["head_conf"], is_train_depth=True)
    except KeyError as error:
        raise ValueError(
            "BEVDepth config must define build_model() or backbone_conf and head_conf"
        ) from error


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--architecture", required=True, choices=("streampetr", "bevdepth"))
    parser.add_argument("--config", help="StreamPETR vanilla config used for strict validation")
    parser.add_argument(
        "--model-factory", help="BEVDepth vanilla model factory as importable.module:function"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.input, map_location="cpu")
    source = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    prefixes = ("detector.", "model.") if args.architecture == "bevdepth" else ("detector.",)
    stripped = detector_only_state_dict(source, detector_prefixes=prefixes)
    if args.architecture == "streampetr" and args.config:
        _build_streampetr(args.config).load_state_dict(stripped, strict=True)
    elif args.architecture == "bevdepth" and (args.model_factory or args.config):
        vanilla = (
            _build_from_factory(args.model_factory)
            if args.model_factory else _build_bevdepth(args.config)
        )
        vanilla.load_state_dict(stripped, strict=True)
    elif args.model_factory:
        raise ValueError("validation option does not match architecture")
    output = {
        "state_dict": stripped,
        "meta": {
            "export": "detector_only_v1",
            "architecture": args.architecture,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"exported {len(stripped)} vanilla detector tensors to {args.output}")


if __name__ == "__main__":
    main()
