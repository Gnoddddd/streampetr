#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-cuda:0}"

pytest -q tests/test_care3d_p2a2_r1_features.py
python scripts/export_care3d_p2a2_r1_features.py \
  --engineering-scene \
  --device "$DEVICE"

python - <<'PY'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.care3d_p2a2_r1_features import finite_model_matrix

report = Path("reports/care3d/p2a2_r1_relative_evidence")
smoke = json.loads((report / "engineering_smoke.json").read_text())
if smoke.get("status") != "P2A2_R1_F0_ENGINEERING_SMOKE_PASSED":
    raise RuntimeError(f"R1-F0 engineering smoke failed: {smoke}")
for key in (
    "r0_assignment_exact", "p2a0_frozen_cost_exact", "feature_non_mutating",
    "fixed_model_matrix_finite",
):
    if smoke.get(key) is not True:
        raise RuntimeError(f"R1-F0 smoke invariant failed: {key}")
for key in (
    "gt_used_as_feature_input", "clean_future_used_as_feature_input",
    "oracle_query_used_as_feature_input", "probe_val_read", "probe_test_read",
):
    if smoke.get(key) is not False:
        raise RuntimeError(f"R1-F0 forbidden input/read flag changed: {key}")
scene = str(smoke["scene_token"])
rows = pd.read_csv(report / "engineering_smoke" / f"{scene}.rows.csv")
if len(rows) != int(smoke.get("disagreement_rows", -1)) or len(rows) <= 0:
    raise RuntimeError("R1-F0 smoke did not export disagreement rows")
if (rows.p2a0_selected_query == rows.lineage_child_query).any():
    raise RuntimeError("R1-F0 smoke exported agreement rows")
if not (rows[["p2a0_wins", "lineage_wins", "both_wrong"]].sum(axis=1) == 1).all():
    raise RuntimeError("R1-F0 smoke labels are not exhaustive")
if not np.isfinite(finite_model_matrix(rows)).all():
    raise RuntimeError("R1-F0 encoded model matrix is not finite")
print(json.dumps(smoke, indent=2, sort_keys=True))
PY
