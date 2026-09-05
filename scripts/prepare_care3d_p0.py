#!/usr/bin/env python3
"""Freeze CARE-3D P0 sources, split, schemas, and engineering smoke scene."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import mmcv
import pandas as pd

from analysis.care3d_counterfactual import PROTOCOLS, assert_disjoint_splits


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/care3d/p0_counterfactual_vulnerability"
STAGE5 = ROOT / "reports/full_nuscenes/prospective_failure_decodability"
STAGE5_SCENES = STAGE5 / "frozen_scene_manifest.csv"
DISCOVERY = ROOT / "reports/full_nuscenes/ctep_method_activation/scene_list.csv"
PREREG = ROOT / "docs/CARE3D_P0_PREREGISTRATION.md"
CONFIG = ROOT / "configs/full_nuscenes/stream_petr_r50_90e_ctep_train_audit.py"
CHECKPOINT = ROOT / "checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth"
INFO = ROOT / "data/nuscenes/nuscenes2d_temporal_infos_train.pkl"
PROTOCOL_PATHS = {
    "blur_back": ROOT / "protocols/presets/motion_blur_back_10f_s09.json",
    "crash_back": ROOT / "protocols/presets/camera_crash_back_10f.json",
    "dark_back": ROOT / "protocols/presets/dark_back_10f_s09.json",
}
EXPECTED_HASHES = {
    CONFIG: "927ba2518a4ca460d2f7f6b3ba74dab620ac8e2995ee7e9aadbcbebf2d7c64a6",
    CHECKPOINT: "e6323ae5c31adf1eedd46d6dd4fd3c73d95aa26f18cc8aa23c196494b7de3451",
    INFO: "dc5e5e611badbdb1c0270a3583e022cf14a9af7b3ff8f02370434b8ec50b493d",
    DISCOVERY: "7bbd389f8ec1f02e75d3d8a7feb773f109fdf24d5824b65db89fc7544e425fb3",
    STAGE5_SCENES: "83637205c930611ccdc6879eb233f72a9b0a5997248f4b5b5edf3242182d6da1",
}
EXPECTED_PROTOCOL_HASHES = {
    "blur_back": "d6245b78b8961715c030b2ddd7908d84d8358ca8939e95efc158da5d33093fe4",
    "crash_back": "6e3c5714d934d0b4991b4858eb2be0519e404f80c181b2ea9b5a5941fb66cdc9",
    "dark_back": "46a46855f7f6db1126dcfa9e14e6469c31b1266959cbdaa5505d511bff2b16b5",
}
STREAM_PETR_COMMIT = "95f64702306ccdb7a78889578b2a55b5deb35b2a"
SCHEMA = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def engineering_scene() -> pd.DataFrame:
    discovery = pd.read_csv(DISCOVERY)
    if len(discovery) != 16:
        raise RuntimeError("expected exactly 16 discovery scenes")
    selected = str(discovery.iloc[0].scene_token)
    info_object = mmcv.load(str(INFO))
    infos = info_object["infos"] if isinstance(info_object, dict) else info_object
    by_scene = defaultdict(list)
    for index, info in enumerate(infos):
        by_scene[str(info["scene_token"])].append((int(info["frame_idx"]), index, str(info["token"])))
    ordered = sorted(by_scene[selected])
    if len(ordered) < 13 or [item[0] for item in ordered[:13]] != list(range(13)):
        raise RuntimeError("engineering scene lacks frames 0..12")
    return pd.DataFrame([{
        "scene_token": selected,
        "scene_name": "engineering_excluded_discovery_scene",
        "split": "engineering_smoke",
        "split_sha256": "excluded_from_formal_split",
        "split_bucket": -1,
        "num_scene_frames": len(ordered),
        "dataset_indices_0_12": json.dumps([item[1] for item in ordered[:13]], separators=(",", ":")),
        "sample_tokens_0_12": json.dumps([item[2] for item in ordered[:13]], separators=(",", ":")),
    }])


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in list(EXPECTED_HASHES) + list(PROTOCOL_PATHS.values()) + [PREREG]
               if not path.exists()]
    if missing:
        raise FileNotFoundError("required CARE-3D P0 sources are missing:\n" + "\n".join(missing))

    observed = {str(path.relative_to(ROOT)): sha256(path) for path in EXPECTED_HASHES}
    expected = {str(path.relative_to(ROOT)): value for path, value in EXPECTED_HASHES.items()}
    if observed != expected:
        raise RuntimeError(f"immutable CARE-3D source mismatch: {observed}")
    protocol_hashes = {key: sha256(path) for key, path in PROTOCOL_PATHS.items()}
    if protocol_hashes != EXPECTED_PROTOCOL_HASHES:
        raise RuntimeError(f"fault protocol hash mismatch: {protocol_hashes}")

    source_validation_path = STAGE5 / "source_validation.json"
    if source_validation_path.exists():
        source_validation = json.loads(source_validation_path.read_text())
        if source_validation.get("status") != "VALIDATED_BEFORE_FORWARD":
            raise RuntimeError("Stage5 source validation is not frozen/valid")
        if source_validation.get("scene_manifest_sha256") != EXPECTED_HASHES[STAGE5_SCENES]:
            raise RuntimeError("Stage5 scene validation hash changed")

    formal = pd.read_csv(STAGE5_SCENES)
    required = {"scene_token", "split", "sample_tokens_0_12", "dataset_indices_0_12"}
    if not required <= set(formal.columns):
        raise RuntimeError(f"Stage5 scene manifest missing columns: {sorted(required - set(formal.columns))}")
    assert_disjoint_splits(formal.to_dict("records"))
    counts = Counter(formal.split.astype(str))
    if len(formal) != 684 or counts != Counter({"probe_train": 419, "probe_val": 133, "probe_test": 132}):
        raise RuntimeError(f"formal scene split changed: {counts}")

    discovery_scenes = set(pd.read_csv(DISCOVERY).scene_token.astype(str))
    if discovery_scenes & set(formal.scene_token.astype(str)):
        raise RuntimeError("discovery scene leaked into formal CARE split")

    frozen_path = REPORT / "frozen_scene_manifest.csv"
    if frozen_path.exists():
        existing = pd.read_csv(frozen_path)
        if not existing.equals(formal):
            raise RuntimeError("existing CARE scene manifest differs from frozen Stage5 split")
    else:
        atomic_csv(frozen_path, formal)

    smoke = engineering_scene()
    smoke_path = REPORT / "engineering_scene_manifest.csv"
    if smoke_path.exists():
        if not pd.read_csv(smoke_path).equals(smoke):
            raise RuntimeError("engineering smoke scene manifest changed")
    else:
        atomic_csv(smoke_path, smoke)

    validation = {
        "schema_version": SCHEMA,
        "status": "VALIDATED_BEFORE_FORWARD",
        "source_hashes": observed,
        "protocol_sha256": protocol_hashes,
        "preregistration_sha256": sha256(PREREG),
        "scene_manifest_sha256": sha256(frozen_path),
        "engineering_manifest_sha256": sha256(smoke_path),
        "scene_counts": dict(counts),
        "excluded_discovery_scenes": 16,
        "stream_petr_commit": STREAM_PETR_COMMIT,
        "detector_training": "PROHIBITED",
        "routing": "DISABLED_P0",
    }
    atomic_json(REPORT / "source_validation.json", validation)

    progress_path = REPORT / "progress_manifest.json"
    initial = {
        "schema_version": SCHEMA,
        "status": "PREPARED_P0_COUNTERFACTUAL_EXTRACTION_PENDING",
        "scene_manifest_sha256": validation["scene_manifest_sha256"],
        "stages": {
            "engineering_smoke": "PENDING",
            "formal_extraction": "LOCKED_PENDING_ENGINEERING_SMOKE",
            "training": "LOCKED_PENDING_EXTRACTION",
            "analysis": "LOCKED_PENDING_TRAINING",
            "P1": "LOCKED_PENDING_P0",
        },
    }
    if progress_path.exists():
        current = json.loads(progress_path.read_text())
        if current.get("scene_manifest_sha256") != initial["scene_manifest_sha256"]:
            raise RuntimeError("CARE resume scene-manifest mismatch")
    else:
        atomic_json(progress_path, initial)

    print(json.dumps(validation, indent=2, sort_keys=True))
    print(f"engineering_scene_token={smoke.iloc[0].scene_token}")


if __name__ == "__main__":
    main()
