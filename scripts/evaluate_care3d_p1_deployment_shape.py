#!/usr/bin/env python3
"""Run locked CARE-3D P1 probe-test with deployed classifier execution shape.

The detector itself keeps its native 3-D classifier calls unchanged. Only the
standalone routed-query replay inside P1 evaluation is redirected through the
validated packed ``[1, 900, 256]`` classifier execution shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import scripts.evaluate_care3d_p1 as _evaluator
from models.care3d_classifier_execution import (
    CLASSIFIER_EXECUTION_POLICY,
    install_deployment_shape_forward,
)


REPORT = _evaluator.REPORT
_ORIGINAL_LOAD_CHECKPOINT = _evaluator.load_checkpoint
_ORIGINAL_REQUIRE_TRAINING_COMPLETE = _evaluator.require_training_complete
_ORIGINAL_ATOMIC_JSON = _evaluator.atomic_json


def _load_checkpoint_and_install(model, *args, **kwargs):
    result = _ORIGINAL_LOAD_CHECKPOINT(model, *args, **kwargs)
    if hasattr(model, "pts_bbox_head") and hasattr(model.pts_bbox_head, "cls_branches"):
        install_deployment_shape_forward(model.pts_bbox_head.cls_branches[-1])
    return result


def _require_training_complete_with_policy():
    validation, progress = _ORIGINAL_REQUIRE_TRAINING_COMPLETE()
    for seed in _evaluator.SEEDS:
        path = REPORT / "training" / f"seed_{seed}" / "training_manifest.json"
        value = json.loads(path.read_text())
        if value.get("classifier_execution_policy") != CLASSIFIER_EXECUTION_POLICY:
            raise RuntimeError(
                f"P1 seed {seed} was not frozen with {CLASSIFIER_EXECUTION_POLICY}"
            )
    return validation, progress


def _atomic_json_with_execution_policy(path: Path, value: object) -> None:
    if isinstance(value, dict) and path.name.endswith(".complete.json") \
            and "evaluation/probe_test" in str(path):
        value = dict(value)
        value["classifier_execution_policy"] = CLASSIFIER_EXECUTION_POLICY
    _ORIGINAL_ATOMIC_JSON(path, value)


def main() -> None:
    _evaluator.load_checkpoint = _load_checkpoint_and_install
    _evaluator.require_training_complete = _require_training_complete_with_policy
    _evaluator.atomic_json = _atomic_json_with_execution_policy
    _evaluator.main()


if __name__ == "__main__":
    main()
