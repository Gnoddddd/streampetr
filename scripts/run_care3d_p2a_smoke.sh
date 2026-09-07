#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda:0}"

pytest -q tests/test_care3d_p2a_association.py
python scripts/prepare_care3d_p2a.py
python scripts/export_care3d_p2a_association.py \
  --engineering-scene \
  --device "$DEVICE"

python - <<'PY'
import json
from pathlib import Path

report = Path("reports/care3d/p2a_online_query_association")
validation = json.loads((report / "source_validation.json").read_text())
progress = json.loads((report / "progress_manifest.json").read_text())
engineering = __import__("pandas").read_csv(report / "engineering_scene_manifest.csv")
if len(engineering) != 1:
    raise RuntimeError("expected exactly one P2-A engineering scene")
scene = str(engineering.iloc[0].scene_token)
marker = json.loads((report / "engineering_smoke" / f"{scene}.complete.json").read_text())
required = (
    "equivalence_pass",
    "branch_state_pass",
    "query_layout_pass",
    "association_input_contract_pass",
    "hungarian_contract_pass",
)
if not marker.get("complete"):
    raise RuntimeError("P2-A engineering marker is incomplete")
if not all(bool(marker.get(key)) for key in required):
    raise RuntimeError(f"P2-A engineering invariant failed: {marker}")
if marker.get("probe_test_read") is not False:
    raise RuntimeError("P2-A engineering smoke read probe-test")
if validation.get("probe_test_read") is not False:
    raise RuntimeError("P2-A source validation indicates probe-test access")
if progress.get("probe_test_read") is not False:
    raise RuntimeError("P2-A progress indicates probe-test access")
probe_test = report / "probe_test"
if probe_test.exists() and any(path.is_file() for path in probe_test.rglob("*")):
    raise RuntimeError("P2-A probe-test artifacts exist during engineering smoke")
if progress.get("status") != "P2A_ENGINEERING_SMOKE_PASSED":
    raise RuntimeError(f"unexpected P2-A smoke progress: {progress.get('status')}")

print(json.dumps({
    "status": "P2A_ENGINEERING_SMOKE_PASSED",
    "scene_token": scene,
    "eligible_rows": marker.get("eligible_rows"),
    "p2a_collision_excluded_rows": marker.get("p2a_collision_excluded_rows"),
    "query_layout_pass": marker.get("query_layout_pass"),
    "branch_state_pass": marker.get("branch_state_pass"),
    "equivalence_pass": marker.get("equivalence_pass"),
    "association_input_contract_pass": marker.get("association_input_contract_pass"),
    "hungarian_contract_pass": marker.get("hungarian_contract_pass"),
    "probe_test_read": False,
}, indent=2, sort_keys=True))
PY
