#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-cuda:0}"
WORKERS="${P2A2_R1_WORKERS:-2}"
MAX_SCENES_PER_SHARD="${P2A2_R1_MAX_SCENES_PER_SHARD:-}"
LOG_DIR="outputs/care3d/p2a2_r1_relative_evidence"
mkdir -p "$LOG_DIR"

if [[ "$WORKERS" != "2" ]]; then
  echo "R1-F0 formal extraction is frozen to exactly 2 workers" >&2
  exit 2
fi

EXTRA=()
if [[ -n "$MAX_SCENES_PER_SHARD" ]]; then
  if ! [[ "$MAX_SCENES_PER_SHARD" =~ ^[0-9]+$ ]]; then
    echo "P2A2_R1_MAX_SCENES_PER_SHARD must be a non-negative integer" >&2
    exit 2
  fi
  EXTRA+=(--max-scenes "$MAX_SCENES_PER_SHARD")
fi

python - <<'PY'
import json
from pathlib import Path
path = Path("reports/care3d/p2a2_r1_relative_evidence/engineering_smoke.json")
if not path.exists():
    raise RuntimeError("run scripts/run_care3d_p2a2_r1_feature_smoke.sh first")
value = json.loads(path.read_text())
if value.get("status") != "P2A2_R1_F0_ENGINEERING_SMOKE_PASSED":
    raise RuntimeError("R1-F0 engineering smoke has not passed")
if value.get("probe_val_read") is not False or value.get("probe_test_read") is not False:
    raise RuntimeError("R1-F0 held-out split lock changed")
print("PASS: R1-F0 probe-train feature extraction is eligible")
PY

pids=()
for ((i=0; i<WORKERS; i++)); do
  log="$LOG_DIR/worker_${i}_of_${WORKERS}.log"
  CUDA_VISIBLE_DEVICES=0 python scripts/export_care3d_p2a2_r1_features.py \
    --split probe_train \
    --num-shards "$WORKERS" \
    --shard-index "$i" \
    --defer-progress \
    --device "$DEVICE" \
    "${EXTRA[@]}" \
    > "$log" 2>&1 &
  pids+=("$!")
done

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "worker $i complete"
  else
    echo "worker $i FAILED; inspect $LOG_DIR/worker_${i}_of_${WORKERS}.log" >&2
    failed=1
  fi
done
[[ "$failed" -eq 0 ]]

if [[ -n "$MAX_SCENES_PER_SHARD" ]]; then
  echo "P2A2_R1_F0_PARALLEL_PILOT_COMPLETE"
  exit 0
fi

python scripts/export_care3d_p2a2_r1_features.py \
  --split probe_train \
  --max-scenes 0 \
  --device "$DEVICE"
python scripts/analyze_care3d_p2a2_r1_features.py
