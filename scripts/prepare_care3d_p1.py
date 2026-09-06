#!/usr/bin/env python3
"""Validate and freeze CARE-3D P1 sources before any P1 forward pass."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "reports/care3d/p0_counterfactual_vulnerability"
TRANSFER = ROOT / "reports/care3d/p0_cross_severity"
REPORT = ROOT / "reports/care3d/p1_sparse_evidence_router"
PREREG = ROOT / "docs/CARE3D_P1_PREREGISTRATION.md"
CONFIG = ROOT / "configs/care3d/p1_sparse_evidence_router.py"
PROTOCOL_PATHS = {
    "blur_back": ROOT / "protocols/presets/motion_blur_back_10f_s09.json",
    "crash_back": ROOT / "protocols/presets/camera_crash_back_10f.json",
    "dark_back": ROOT / "protocols/presets/dark_back_10f_s09.json",
}
EXPECTED_SCENE_MANIFEST_SHA256 = "83637205c930611ccdc6879eb233f72a9b0a5997248f4b5b5edf3242182d6da1"
SEEDS = (42, 2027, 2028)
SCHEMA = 1
QUERY_COLLISION_POLICY = "exclude_all_rows_in_shared_target_query_frame"
AMENDMENT_HEADING = "## Implementation amendment: shared target-query collisions"


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


def any_files(path: Path) -> bool:
    return path.exists() and any(item.is_file() for item in path.rglob("*"))


def amendment_is_explicitly_preregistered() -> bool:
    text = PREREG.read_text(encoding="utf-8")
    return AMENDMENT_HEADING in text and QUERY_COLLISION_POLICY in text


def downstream_p1_training_or_test_started() -> dict:
    training_files = any_files(REPORT / "training")
    probe_test_files = any_files(REPORT / "evaluation/probe_test")
    progress_path = REPORT / "progress_manifest.json"
    probe_test_progress_opened = False
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        test_stage = progress.get("stages", {}).get("probe_test_evaluation")
        probe_test_progress_opened = isinstance(test_stage, dict) or test_stage not in (
            None,
            "LOCKED_PENDING_TRAINING",
        )
    return {
        "training_files_exist": bool(training_files),
        "probe_test_files_exist": bool(probe_test_files),
        "probe_test_progress_opened": bool(probe_test_progress_opened),
        "started": bool(training_files or probe_test_files or probe_test_progress_opened),
    }


def main() -> None:
    required = [
        P0 / "decision.json",
        P0 / "progress_manifest.json",
        P0 / "source_validation.json",
        P0 / "frozen_scene_manifest.csv",
        P0 / "engineering_scene_manifest.csv",
        TRANSFER / "decision.json",
        TRANSFER / "progress_manifest.json",
        PREREG,
        CONFIG,
        *PROTOCOL_PATHS.values(),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen P1 prerequisite(s): {missing}")

    p0_decision = json.loads((P0 / "decision.json").read_text())
    if p0_decision.get("decision") != "GO_CARE3D_COUNTERFACTUAL_P0":
        raise RuntimeError("P1 locked: main CARE P0 is not GO")
    transfer_decision = json.loads((TRANSFER / "decision.json").read_text())
    if transfer_decision.get("decision") != "PASS_CARE3D_P0_CROSS_SEVERITY":
        raise RuntimeError("P1 locked: cross-severity transfer did not pass")
    if transfer_decision.get("retrained") is not False or transfer_decision.get("recalibrated") is not False:
        raise RuntimeError("P1 locked: cross-severity provenance changed")

    p0_validation = json.loads((P0 / "source_validation.json").read_text())
    manifest_path = P0 / "frozen_scene_manifest.csv"
    manifest_sha = sha256(manifest_path)
    if manifest_sha != EXPECTED_SCENE_MANIFEST_SHA256:
        raise RuntimeError(f"frozen P0 scene manifest hash changed: {manifest_sha}")
    if p0_validation.get("scene_manifest_sha256") != manifest_sha:
        raise RuntimeError("P0 source-validation scene hash no longer matches")

    manifest = pd.read_csv(manifest_path)
    counts = manifest.groupby("split").size().to_dict()
    expected = {"probe_train": 419, "probe_val": 133, "probe_test": 132}
    if counts != expected:
        raise RuntimeError(f"P1 split identity changed: {counts} != {expected}")
    if manifest.scene_token.astype(str).duplicated().any():
        raise RuntimeError("duplicate scenes in frozen P0 manifest")

    checkpoint_hashes = {}
    training_manifest_hashes = {}
    for seed in SEEDS:
        directory = P0 / "training" / f"seed_{seed}"
        checkpoint = directory / "best.pth"
        training_manifest = directory / "training_manifest.json"
        if not checkpoint.exists() or not training_manifest.exists():
            raise RuntimeError(f"missing frozen P0 seed {seed}")
        meta = json.loads(training_manifest.read_text())
        if meta.get("status") != "TRAINING_COMPLETE_TEST_UNSEEN":
            raise RuntimeError(f"P0 seed {seed} status changed")
        if meta.get("probe_test_read") is not False:
            raise RuntimeError(f"P0 seed {seed} was not test-blind")
        if meta.get("routing_enabled") is not False:
            raise RuntimeError(f"P0 seed {seed} unexpectedly enabled routing")
        if int(meta.get("detector_parameters_in_optimizer", -1)) != 0:
            raise RuntimeError(f"P0 seed {seed} optimized detector parameters")
        checkpoint_hashes[str(seed)] = sha256(checkpoint)
        training_manifest_hashes[str(seed)] = sha256(training_manifest)

    source = {
        "schema_version": SCHEMA,
        "status": "VALIDATED_BEFORE_P1_FORWARD",
        "main_p0_decision": p0_decision["decision"],
        "main_p0_decision_sha256": sha256(P0 / "decision.json"),
        "cross_severity_decision": transfer_decision["decision"],
        "cross_severity_decision_sha256": sha256(TRANSFER / "decision.json"),
        "scene_manifest_sha256": manifest_sha,
        "p0_checkpoint_sha256": checkpoint_hashes,
        "p0_training_manifest_sha256": training_manifest_hashes,
        "p1_preregistration_sha256": sha256(PREREG),
        "p1_config_sha256": sha256(CONFIG),
        "protocol_sha256": {name: sha256(path) for name, path in PROTOCOL_PATHS.items()},
        "split_counts": expected,
        "probe_test_locked_until_training_complete": True,
        "stream_petr_frozen": True,
        "p0_frozen": True,
    }

    REPORT.mkdir(parents=True, exist_ok=True)
    validation_path = REPORT / "source_validation.json"
    if validation_path.exists():
        previous = json.loads(validation_path.read_text())
        frozen_keys = (
            "main_p0_decision_sha256",
            "cross_severity_decision_sha256",
            "scene_manifest_sha256",
            "p0_checkpoint_sha256",
            "p0_training_manifest_sha256",
            "p1_preregistration_sha256",
            "p1_config_sha256",
            "protocol_sha256",
        )
        changed = [key for key in frozen_keys if previous.get(key) != source.get(key)]
        if changed:
            downstream = downstream_p1_training_or_test_started()
            amendment_allowed = bool(
                changed == ["p1_preregistration_sha256"]
                and amendment_is_explicitly_preregistered()
                and not downstream["started"]
            )
            if not amendment_allowed:
                raise RuntimeError(
                    "P1 frozen source identity changed outside the allowed pre-training "
                    f"preregistration amendment: changed={changed}, downstream={downstream}"
                )
            source["p1_preregistration_amendment"] = {
                "type": "shared_target_query_collision_eligibility",
                "policy": QUERY_COLLISION_POLICY,
                "declared_date": "2026-09-06",
                "previous_sha256": previous.get("p1_preregistration_sha256"),
                "current_sha256": source["p1_preregistration_sha256"],
                "only_changed_frozen_key": "p1_preregistration_sha256",
                "upstream_identity_unchanged": True,
                "p1_router_training_started": False,
                "probe_test_opened": False,
            }
        elif "p1_preregistration_amendment" in previous:
            source["p1_preregistration_amendment"] = previous[
                "p1_preregistration_amendment"
            ]
    atomic_json(validation_path, source)

    manifest.to_csv(REPORT / "frozen_scene_manifest.csv", index=False)
    engineering = pd.read_csv(P0 / "engineering_scene_manifest.csv")
    if len(engineering) != 1:
        raise RuntimeError("expected exactly one frozen P0 engineering scene")
    engineering.to_csv(REPORT / "engineering_scene_manifest.csv", index=False)

    progress_path = REPORT / "progress_manifest.json"
    if not progress_path.exists():
        progress = {
            "schema_version": SCHEMA,
            "scene_manifest_sha256": manifest_sha,
            "status": "P1_ENGINEERING_SMOKE_PENDING",
            "stages": {
                "engineering_smoke": "PENDING",
                "supervision_extraction": "LOCKED_PENDING_ENGINEERING_SMOKE",
                "training": "LOCKED_PENDING_SUPERVISION",
                "probe_test_evaluation": "LOCKED_PENDING_TRAINING",
                "analysis": "LOCKED_PENDING_TEST_EVALUATION",
                "P2": "LOCKED_PENDING_P1",
            },
        }
        atomic_json(progress_path, progress)
    else:
        progress = json.loads(progress_path.read_text())
        if progress.get("scene_manifest_sha256") != manifest_sha:
            raise RuntimeError("P1 progress manifest belongs to another cohort")

    print(json.dumps(source, indent=2, sort_keys=True))
    print(json.dumps(json.loads(progress_path.read_text()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
