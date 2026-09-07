#!/usr/bin/env python3
"""Validate and freeze CARE-3D P2-A0 sources before association forward."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "reports/care3d/p0_counterfactual_vulnerability"
P1 = ROOT / "reports/care3d/p1_sparse_evidence_router"
REPORT = ROOT / "reports/care3d/p2a_online_query_association"
PREREG = ROOT / "docs/CARE3D_P2A_PREREGISTRATION.md"
CONFIG = ROOT / "configs/care3d/p2a_online_query_association.py"
P1_FREEZE = ROOT / "docs/results/CARE3D_P1_FINAL_SHA256.txt"
DETECTOR_CONFIG = ROOT / "configs/full_nuscenes/stream_petr_r50_90e_ctep_train_audit.py"
CHECKPOINT = ROOT / "checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth"
TRAIN_INFO = ROOT / "data/nuscenes/nuscenes2d_temporal_infos_train.pkl"
VAL_INFO = ROOT / "data/nuscenes/nuscenes2d_temporal_infos_val.pkl"
PROTOCOL_PATHS = {
    "blur_back": ROOT / "protocols/presets/motion_blur_back_10f_s09.json",
    "crash_back": ROOT / "protocols/presets/camera_crash_back_10f.json",
    "dark_back": ROOT / "protocols/presets/dark_back_10f_s09.json",
}
EXPECTED = {
    "scene_manifest_sha256": "83637205c930611ccdc6879eb233f72a9b0a5997248f4b5b5edf3242182d6da1",
    "p1_decision_sha256": "d0172c9de8d17c4359224a90cee11312d260b45f595fcf662d5d8ccbd2405da3",
    "detector_config_sha256": "927ba2518a4ca460d2f7f6b3ba74dab620ac8e2995ee7e9aadbcbebf2d7c64a6",
    "detector_checkpoint_sha256": "e6323ae5c31adf1eedd46d6dd4fd3c73d95aa26f18cc8aa23c196494b7de3451",
    "train_info_sha256": "dc5e5e611badbdb1c0270a3583e022cf14a9af7b3ff8f02370434b8ec50b493d",
}
EXPECTED_PROTOCOL_HASHES = {
    "blur_back": "d6245b78b8961715c030b2ddd7908d84d8358ca8939e95efc158da5d33093fe4",
    "crash_back": "6e3c5714d934d0b4991b4858eb2be0519e404f80c181b2ea9b5a5941fb66cdc9",
    "dark_back": "46a46855f7f6db1126dcfa9e14e6469c31b1266959cbdaa5505d511bff2b16b5",
}
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


def main() -> None:
    required = [
        P0 / "decision.json",
        P0 / "source_validation.json",
        P0 / "frozen_scene_manifest.csv",
        P0 / "engineering_scene_manifest.csv",
        P1 / "decision.json",
        P1 / "progress_manifest.json",
        PREREG,
        CONFIG,
        P1_FREEZE,
        DETECTOR_CONFIG,
        CHECKPOINT,
        TRAIN_INFO,
        VAL_INFO,
        *PROTOCOL_PATHS.values(),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing P2-A prerequisite(s):\n" + "\n".join(missing))

    p0_decision = json.loads((P0 / "decision.json").read_text())
    if p0_decision.get("decision") != "GO_CARE3D_COUNTERFACTUAL_P0":
        raise RuntimeError("P2-A locked: CARE P0 is not GO")
    p1_decision_path = P1 / "decision.json"
    p1_decision = json.loads(p1_decision_path.read_text())
    if p1_decision.get("decision") != "GO_CARE3D_P1_SPARSE_EVIDENCE_ROUTER":
        raise RuntimeError("P2-A locked: CARE P1 is not GO")
    if p1_decision.get("P2_status") != "ELIGIBLE":
        raise RuntimeError("P2-A locked: P1 did not unlock P2")
    p1_progress = json.loads((P1 / "progress_manifest.json").read_text())
    if p1_progress.get("status") != "GO_CARE3D_P1_SPARSE_EVIDENCE_ROUTER":
        raise RuntimeError("P2-A locked: P1 progress is not frozen GO")

    observed = {
        "scene_manifest_sha256": sha256(P0 / "frozen_scene_manifest.csv"),
        "p1_decision_sha256": sha256(p1_decision_path),
        "detector_config_sha256": sha256(DETECTOR_CONFIG),
        "detector_checkpoint_sha256": sha256(CHECKPOINT),
        "train_info_sha256": sha256(TRAIN_INFO),
    }
    if observed != EXPECTED:
        raise RuntimeError(f"immutable P2-A source mismatch: {observed}")
    freeze_text = P1_FREEZE.read_text(encoding="utf-8")
    if EXPECTED["p1_decision_sha256"] not in freeze_text:
        raise RuntimeError("P1 final SHA freeze does not contain the P1 decision hash")

    protocol_hashes = {name: sha256(path) for name, path in PROTOCOL_PATHS.items()}
    if protocol_hashes != EXPECTED_PROTOCOL_HASHES:
        raise RuntimeError(f"P2-A protocol hash mismatch: {protocol_hashes}")

    manifest = pd.read_csv(P0 / "frozen_scene_manifest.csv")
    counts = manifest.groupby("split").size().to_dict()
    expected_counts = {"probe_train": 419, "probe_val": 133, "probe_test": 132}
    if counts != expected_counts:
        raise RuntimeError(f"P2-A split identity changed: {counts}")
    if manifest.scene_token.astype(str).duplicated().any():
        raise RuntimeError("P2-A frozen scene manifest contains duplicate scenes")
    engineering = pd.read_csv(P0 / "engineering_scene_manifest.csv")
    if len(engineering) != 1 or str(engineering.iloc[0].split) != "engineering_smoke":
        raise RuntimeError("P2-A engineering manifest changed")

    REPORT.mkdir(parents=True, exist_ok=True)
    frozen_manifest = REPORT / "frozen_scene_manifest.csv"
    engineering_manifest = REPORT / "engineering_scene_manifest.csv"
    if frozen_manifest.exists() and not pd.read_csv(frozen_manifest).equals(manifest):
        raise RuntimeError("existing P2-A frozen scene manifest changed")
    if engineering_manifest.exists() and not pd.read_csv(engineering_manifest).equals(engineering):
        raise RuntimeError("existing P2-A engineering manifest changed")
    if not frozen_manifest.exists():
        atomic_csv(frozen_manifest, manifest)
    if not engineering_manifest.exists():
        atomic_csv(engineering_manifest, engineering)

    probe_test_dir = REPORT / "probe_test"
    if probe_test_dir.exists() and any(path.is_file() for path in probe_test_dir.rglob("*")):
        raise RuntimeError("P2-A probe_test artifacts exist before the P2-A0 gate")

    validation = {
        "schema_version": SCHEMA,
        "status": "VALIDATED_BEFORE_P2A_FORWARD",
        "scene_manifest_sha256": observed["scene_manifest_sha256"],
        "p1_decision": p1_decision["decision"],
        "p1_decision_sha256": observed["p1_decision_sha256"],
        "detector_config_sha256": observed["detector_config_sha256"],
        "detector_checkpoint_sha256": observed["detector_checkpoint_sha256"],
        "train_info_sha256": observed["train_info_sha256"],
        "official_val_info_sha256_record_only": sha256(VAL_INFO),
        "protocol_sha256": protocol_hashes,
        "p2a_preregistration_sha256": sha256(PREREG),
        "p2a_config_sha256": sha256(CONFIG),
        "split_counts": expected_counts,
        "stream_petr_frozen": True,
        "p0_frozen": True,
        "p1_frozen": True,
        "probe_test_read": False,
        "probe_test_locked": True,
    }
    validation_path = REPORT / "source_validation.json"
    if validation_path.exists():
        previous = json.loads(validation_path.read_text())
        frozen_keys = (
            "scene_manifest_sha256",
            "p1_decision_sha256",
            "detector_config_sha256",
            "detector_checkpoint_sha256",
            "train_info_sha256",
            "official_val_info_sha256_record_only",
            "protocol_sha256",
            "p2a_preregistration_sha256",
            "p2a_config_sha256",
        )
        changed = [key for key in frozen_keys if previous.get(key) != validation.get(key)]
        if changed:
            raise RuntimeError(f"P2-A frozen source identity changed: {changed}")
    else:
        atomic_json(validation_path, validation)

    progress_path = REPORT / "progress_manifest.json"
    if not progress_path.exists():
        atomic_json(progress_path, {
            "schema_version": SCHEMA,
            "scene_manifest_sha256": observed["scene_manifest_sha256"],
            "status": "P2A_PREPARED_ENGINEERING_SMOKE_PENDING",
            "probe_test_read": False,
            "stages": {
                "engineering_smoke": "PENDING",
                "probe_train_extraction": "LOCKED_PENDING_ENGINEERING_SMOKE",
                "config_selection": "LOCKED_PENDING_TRAIN_EXTRACTION",
                "probe_val_extraction": "LOCKED_PENDING_SELECTION",
                "probe_val_analysis": "LOCKED_PENDING_VAL_EXTRACTION",
                "probe_test": "LOCKED_P2A0",
                "P2A1": "LOCKED_PENDING_P2A0",
            },
        })
    else:
        progress = json.loads(progress_path.read_text())
        if progress.get("scene_manifest_sha256") != observed["scene_manifest_sha256"]:
            raise RuntimeError("P2-A progress belongs to another cohort")
        if progress.get("probe_test_read") is not False:
            raise RuntimeError("P2-A probe-test lock was violated")

    print(json.dumps(validation, indent=2, sort_keys=True))
    print(json.dumps(json.loads(progress_path.read_text()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
