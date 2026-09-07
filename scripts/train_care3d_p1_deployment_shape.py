#!/usr/bin/env python3
"""Run formal CARE-3D P1 training with deployed classifier execution shape.

This entrypoint leaves the frozen P1 dataset, loss, optimizer, seeds and replay
tolerance unchanged. It only patches the frozen StreamPETR final classifier so
standalone 2-D object batches are evaluated through the deployed
``[1, 900, 256]`` execution shape validated by the pre-test replay diagnosis.
Native 3-D detector calls are untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import scripts.train_care3d_p1 as _trainer
from models.care3d_classifier_execution import (
    CLASSIFIER_EXECUTION_POLICY,
    install_deployment_shape_forward,
)


REPORT = _trainer.REPORT
DIAGNOSIS = REPORT / "classifier_replay_diagnosis.json"
REPLAY_TOLERANCE = 5e-4


def require_diagnosis() -> dict:
    if not DIAGNOSIS.exists():
        raise RuntimeError("run scripts/diagnose_care3d_p1_classifier_replay.py first")
    value = json.loads(DIAGNOSIS.read_text())
    if value.get("status") != "P1_CLASSIFIER_REPLAY_DIAGNOSIS_COMPLETE":
        raise RuntimeError("P1 classifier replay diagnosis is incomplete")
    if value.get("probe_test_read") is not False:
        raise RuntimeError("probe-test was opened before classifier execution repair")
    if value.get("storage_precision_policy") != "fp32_router_supervision_v1":
        raise RuntimeError("deployment-shape training requires FP32 query supervision")
    packed = float(value.get("packed_900_replay", {}).get("max_abs_diff", float("inf")))
    matched = float(value.get("shape_matched_900_replay", {}).get("max_abs_diff", float("inf")))
    if packed > REPLAY_TOLERANCE or matched > REPLAY_TOLERANCE:
        raise RuntimeError(
            f"deployment-shape replay is not validated: packed={packed}, matched={matched}"
        )
    return value


_ORIGINAL_BUILD_CLASSIFIER = _trainer.build_classifier
_ORIGINAL_ATOMIC_JSON = _trainer.atomic_json


def _build_classifier(device, config):
    classifier = _ORIGINAL_BUILD_CLASSIFIER(device, config)
    return install_deployment_shape_forward(classifier)


def _atomic_json_with_execution_policy(path: Path, value: object) -> None:
    if isinstance(value, dict) and path.name == "training_manifest.json":
        value = dict(value)
        value["classifier_execution_policy"] = CLASSIFIER_EXECUTION_POLICY
        value["classifier_replay_diagnosis_sha256"] = _trainer.sha256(DIAGNOSIS)
        value["classifier_replay_tolerance"] = REPLAY_TOLERANCE
    _ORIGINAL_ATOMIC_JSON(path, value)


def main() -> None:
    diagnosis = require_diagnosis()
    _trainer.build_classifier = _build_classifier
    _trainer.atomic_json = _atomic_json_with_execution_policy
    print(json.dumps({
        "event": "P1_DEPLOYMENT_SHAPE_CLASSIFIER_ENABLED",
        "policy": CLASSIFIER_EXECUTION_POLICY,
        "packed_900_max_abs_diff": diagnosis["packed_900_replay"]["max_abs_diff"],
        "shape_matched_900_max_abs_diff": diagnosis["shape_matched_900_replay"]["max_abs_diff"],
        "probe_test_read": False,
    }, sort_keys=True), flush=True)
    _trainer.main()


if __name__ == "__main__":
    main()
