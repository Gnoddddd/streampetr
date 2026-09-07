#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pytest \
  tests/test_care3d.py \
  tests/test_care3d_p0_pipeline.py \
  tests/test_care3d_cross_severity.py \
  tests/test_care3d_p1.py \
  tests/test_care3d_p1_fp32_export.py \
  tests/test_care3d_classifier_execution.py \
  -v

python scripts/diagnose_care3d_p1_classifier_replay.py

python - <<'PY'
import json
from pathlib import Path

path = Path("reports/care3d/p1_sparse_evidence_router/classifier_replay_diagnosis.json")
value = json.loads(path.read_text())
tol = float(value["frozen_replay_tolerance"])
packed = float(value["packed_900_replay"]["max_abs_diff"])
matched = float(value["shape_matched_900_replay"]["max_abs_diff"])
standalone = float(value["trainer_style_replay"]["max_abs_diff"])
assert value["probe_test_read"] is False
assert packed <= tol
assert matched <= tol
assert standalone > tol
print(json.dumps({
    "status": "P1_DEPLOYMENT_SHAPE_CLASSIFIER_SMOKE_PASSED",
    "probe_test_read": False,
    "trainer_style_max_abs_diff": standalone,
    "packed_900_max_abs_diff": packed,
    "shape_matched_900_max_abs_diff": matched,
    "frozen_replay_tolerance": tol,
    "classifier_execution_policy": "packed_deployment_shape_900_v1",
}, indent=2, sort_keys=True))
PY
