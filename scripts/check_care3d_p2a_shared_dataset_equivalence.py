#!/usr/bin/env python3
"""Validate P2-A shared-info dataset views against the canonical builder.

This is an engineering-only check on one discovery scene excluded from every
formal CARE split.  It does not run association efficacy, probe-val, or
probe-test.  For frames 3 and 12 it verifies that the canonical independently
built fault dataset and the memory-saving shared-info view produce exactly the
same model inputs for every frozen fault protocol.
"""

from __future__ import annotations

import gc
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STREAM = ROOT / "repos/StreamPETR"
sys.dont_write_bytecode = True
sys.path.insert(0, str(STREAM))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.utils import import_modules_from_strings  # noqa: E402

from analysis.care3d_p2a_execution import (  # noqa: E402
    EXECUTION_POLICY,
    build_shared_protocol_dataset,
)
from scripts.audit_dark_target_recoverability import unpack  # noqa: E402
from scripts.run_bd_temporal_support_p0 import CONFIG, protocol_dataset  # noqa: E402


REPORT = ROOT / "reports/care3d/p2a_online_query_association"
PROTOCOL_PATHS = {
    "blur_back": ROOT / "protocols/presets/motion_blur_back_10f_s09.json",
    "crash_back": ROOT / "protocols/presets/camera_crash_back_10f.json",
    "dark_back": ROOT / "protocols/presets/dark_back_10f_s09.json",
}
FRAMES = (3, 12)
SEED = 20260907


def reset_rng() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def exact_equal(left, right, path="root") -> None:
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            raise RuntimeError(f"type mismatch at {path}")
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(f"tensor layout mismatch at {path}")
        if not torch.equal(left.cpu(), right.cpu()):
            diff = (
                float((left.detach().float().cpu() - right.detach().float().cpu()).abs().max())
                if left.numel() else 0.0
            )
            raise RuntimeError(f"tensor mismatch at {path}: max_abs_diff={diff}")
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            raise RuntimeError(f"array type mismatch at {path}")
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(f"array layout mismatch at {path}")
        if not np.array_equal(left, right, equal_nan=True):
            raise RuntimeError(f"array mismatch at {path}")
        return
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise RuntimeError(f"dict type mismatch at {path}")
        if set(left) != set(right):
            raise RuntimeError(f"dict keys mismatch at {path}")
        for key in left:
            exact_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, type(right)) or len(left) != len(right):
            raise RuntimeError(f"sequence mismatch at {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            exact_equal(a, b, f"{path}[{index}]")
        return
    try:
        equal = left == right
        if isinstance(equal, (bool, np.bool_)):
            if not bool(equal):
                raise RuntimeError(f"value mismatch at {path}: {left!r} != {right!r}")
            return
    except Exception:
        pass
    if repr(left) != repr(right):
        raise RuntimeError(f"repr mismatch at {path}: {left!r} != {right!r}")


def unpack_cpu(dataset, index):
    reset_rng()
    sample = dataset[index]
    meta, image, data = unpack(sample, torch.device("cpu"))
    return meta, image, data


def main() -> None:
    validation = json.loads((REPORT / "source_validation.json").read_text())
    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    if validation.get("status") != "VALIDATED_BEFORE_P2A_FORWARD":
        raise RuntimeError("P2-A source validation is not frozen")
    if validation.get("probe_test_read") is not False or progress.get("probe_test_read") is not False:
        raise RuntimeError("probe-test must remain locked during shared-dataset validation")

    engineering = pd.read_csv(REPORT / "engineering_scene_manifest.csv")
    if len(engineering) != 1:
        raise RuntimeError("expected one excluded engineering scene")
    tokens = json.loads(str(engineering.iloc[0].sample_tokens_0_12))

    cfg = Config.fromfile(str(CONFIG))
    import_modules_from_strings(**cfg.custom_imports)
    cfg.model.pretrained = None
    clean = protocol_dataset(cfg, None)
    token_index = {str(info["token"]): index for index, info in enumerate(clean.data_infos)}

    checks = []
    for protocol, schedule in PROTOCOL_PATHS.items():
        canonical = protocol_dataset(cfg, schedule)
        shared = build_shared_protocol_dataset(clean, cfg, schedule)
        if shared.data_infos is not clean.data_infos:
            raise RuntimeError("shared dataset did not reuse clean data_infos")
        if len(canonical) != len(shared):
            raise RuntimeError(f"dataset length mismatch: {protocol}")

        for frame_idx in FRAMES:
            token = str(tokens[frame_idx])
            index = token_index[token]
            if str(canonical.data_infos[index]["token"]) != token:
                raise RuntimeError(f"canonical token mismatch: {protocol} frame={frame_idx}")
            if str(shared.data_infos[index]["token"]) != token:
                raise RuntimeError(f"shared token mismatch: {protocol} frame={frame_idx}")

            canonical_meta, canonical_image, canonical_data = unpack_cpu(canonical, index)
            shared_meta, shared_image, shared_data = unpack_cpu(shared, index)
            exact_equal(canonical_image, shared_image, "image")
            exact_equal(canonical_data, shared_data, "data")
            exact_equal(canonical_meta, shared_meta, "meta")
            checks.append({
                "protocol": protocol,
                "frame_idx": int(frame_idx),
                "sample_token": token,
                "image_shape": list(canonical_image.shape),
                "input_exact": True,
            })

        del canonical, shared
        gc.collect()

    result = {
        "status": "P2A_SHARED_DATASET_EQUIVALENCE_PASSED",
        "execution_policy": EXECUTION_POLICY,
        "engineering_scene_token": str(engineering.iloc[0].scene_token),
        "frames": list(FRAMES),
        "protocols": list(PROTOCOL_PATHS),
        "checks": checks,
        "all_model_inputs_exact": True,
        "shared_data_infos": True,
        "probe_val_read": False,
        "probe_test_read": False,
    }
    out = REPORT / "engineering_smoke/shared_dataset_equivalence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
