#!/usr/bin/env python3
"""Freeze CARE-3D P2-A2-R0 inputs without opening probe-val or probe-test data."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from analysis.care3d_p2a2_lineage import FROZEN_P2A0_CONFIG


ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "reports/care3d/p0_counterfactual_vulnerability"
P2A0 = ROOT / "reports/care3d/p2a_online_query_association"
REPORT = ROOT / "reports/care3d/p2a2_memory_lineage_r0"
P2A0_FREEZE = P2A0 / "final_freeze/P2A0_FINAL_SHA256.txt"
EXPECTED_SELECTION_SHA256 = "493901c6b15f8233b804d38c592a4a9abd9d0810a131eee1b87ebd6a999ebe12"
EXPECTED_DECISION_SHA256 = "8800d3012379e87df01f11a83b4e92fef079c3d6583f064d6e6ac98d5966439c"
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
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def main() -> None:
    selection_path = P2A0 / "selection.json"
    decision_path = P2A0 / "decision.json"
    source_manifest_path = P0 / "frozen_scene_manifest.csv"
    engineering_path = P0 / "engineering_scene_manifest.csv"
    required = (
        selection_path,
        decision_path,
        P2A0_FREEZE,
        source_manifest_path,
        engineering_path,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing P2-A2-R0 prerequisite(s):\n" + "\n".join(missing))
    observed_selection_hash = sha256(selection_path)
    observed_decision_hash = sha256(decision_path)
    if observed_selection_hash != EXPECTED_SELECTION_SHA256:
        raise RuntimeError("frozen P2-A0 selection checksum changed")
    if observed_decision_hash != EXPECTED_DECISION_SHA256:
        raise RuntimeError("frozen P2-A0 decision checksum changed")
    freeze_text = P2A0_FREEZE.read_text(encoding="utf-8")
    if EXPECTED_SELECTION_SHA256 not in freeze_text or EXPECTED_DECISION_SHA256 not in freeze_text:
        raise RuntimeError("P2-A0 final freeze does not contain required checksums")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    selected = selection.get("selected", {})
    if selection.get("status") != "P2A_GLOBAL_ASSOCIATION_CONFIG_FROZEN":
        raise RuntimeError("P2-A0 selected configuration is not frozen")
    if selected.get("config_id") != FROZEN_P2A0_CONFIG.config_id:
        raise RuntimeError("P2-A0 selected config id changed")
    for key, expected in FROZEN_P2A0_CONFIG.as_dict().items():
        if key == "config_id":
            continue
        if float(selected.get(key, float("nan"))) != float(expected):
            raise RuntimeError(f"P2-A0 selected {key} changed")
    if decision.get("decision") != "NO_GO_CARE3D_P2A_ONLINE_QUERY_ASSOCIATION":
        raise RuntimeError("P2-A0 decision is not the frozen No-Go")
    if decision.get("P2A1_status") != "LOCKED_P2A0_NO_GO":
        raise RuntimeError("P2-A1 lock changed")
    if selection.get("probe_val_read") is not False or selection.get("probe_test_read") is not False:
        raise RuntimeError("P2-A0 selection input discipline changed")
    if decision.get("probe_test_read") is not False:
        raise RuntimeError("P2-A0 probe-test lock changed")

    source_manifest = pd.read_csv(source_manifest_path)
    train = source_manifest[
        source_manifest.split.astype(str) == "probe_train"
    ].reset_index(drop=True)
    if len(train) != 419 or train.scene_token.astype(str).duplicated().any():
        raise RuntimeError("P2-A2-R0 requires exactly 419 unique probe-train scenes")
    engineering = pd.read_csv(engineering_path)
    if len(engineering) != 1 or str(engineering.iloc[0].split) != "engineering_smoke":
        raise RuntimeError("excluded engineering scene changed")

    train_path = REPORT / "probe_train_manifest.csv"
    smoke_path = REPORT / "engineering_scene_manifest.csv"
    if train_path.exists() and not pd.read_csv(train_path).equals(train):
        raise RuntimeError("existing P2-A2-R0 probe-train manifest changed")
    if smoke_path.exists() and not pd.read_csv(smoke_path).equals(engineering):
        raise RuntimeError("existing P2-A2-R0 engineering manifest changed")
    if not train_path.exists():
        atomic_csv(train_path, train)
    if not smoke_path.exists():
        atomic_csv(smoke_path, engineering)

    validation = {
        "schema_version": SCHEMA,
        "status": "P2A2_R0_PREPARED_ENGINEERING_SMOKE_PENDING",
        "development_pre_gate": True,
        "p2a0_decision": decision["decision"],
        "P2A1_status": decision["P2A1_status"],
        "p2a0_selection_sha256": observed_selection_hash,
        "p2a0_decision_sha256": observed_decision_hash,
        "selected_config": FROZEN_P2A0_CONFIG.as_dict(),
        "probe_train_scenes": 419,
        "engineering_scenes": 1,
        "probe_val_read": False,
        "probe_test_read": False,
        "probe_test_locked": True,
        "gt_used_as_association_input": False,
        "clean_future_used_as_association_input": False,
        "oracle_query_used_as_association_input": False,
    }
    validation_path = REPORT / "source_validation.json"
    if validation_path.exists():
        previous = json.loads(validation_path.read_text(encoding="utf-8"))
        for key in (
            "p2a0_selection_sha256",
            "p2a0_decision_sha256",
            "selected_config",
            "probe_train_scenes",
        ):
            if previous.get(key) != validation.get(key):
                raise RuntimeError(f"existing P2-A2-R0 source validation changed: {key}")
    else:
        atomic_json(validation_path, validation)
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
