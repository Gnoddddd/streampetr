#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-cuda:0}"

pytest -q tests/test_care3d_p2a2_lineage.py
python scripts/prepare_care3d_p2a2_lineage_r0.py
python scripts/export_care3d_p2a2_lineage_r0.py \
  --engineering-scene \
  --device "$DEVICE"

python - <<'PY'
import json
from pathlib import Path

path = Path("reports/care3d/p2a2_memory_lineage_r0/engineering_smoke.json")
value = json.loads(path.read_text())
if value.get("status") != "P2A2_R0_LINEAGE_SMOKE_PASSED":
    raise RuntimeError(f"P2-A2-R0 lineage smoke failed: {value}")
for key in (
    "gt_used_as_input",
    "clean_future_used_as_input",
    "oracle_query_used_as_input",
    "probe_val_read",
    "probe_test_read",
):
    if value.get(key) is not False:
        raise RuntimeError(f"P2-A2-R0 forbidden input/read flag changed: {key}")
if value.get("num_query") != 644 or value.get("num_propagated") != 256:
    raise RuntimeError("P2-A2-R0 query layout changed")
if value.get("topk_proposals") != 256 or value.get("query_count") != 900:
    raise RuntimeError("P2-A2-R0 Top-K layout changed")
if value.get("memory_lineage_torch_equal") is not True:
    raise RuntimeError("recomputed rec_memory is not torch.equal to memory prefix")
if float(value.get("memory_lineage_max_abs_diff", float("inf"))) != 0.0:
    raise RuntimeError("recomputed rec_memory differs from memory prefix")
if int(value.get("rows", 0)) <= 0:
    raise RuntimeError("engineering smoke did not exercise real association rows")
print(json.dumps(value, indent=2, sort_keys=True))
PY
