#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

WORKERS="${P2A_WORKERS:-2}"
DEVICE="${DEVICE:-cuda:0}"
MAX_SCENES_PER_SHARD="${P2A_MAX_SCENES_PER_SHARD:-}"
LOG_DIR="outputs/care3d/p2a_parallel_train"
mkdir -p "$LOG_DIR"

if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "P2A_WORKERS must be a positive integer" >&2
  exit 2
fi

EXTRA=()
if [[ -n "$MAX_SCENES_PER_SHARD" ]]; then
  if ! [[ "$MAX_SCENES_PER_SHARD" =~ ^[0-9]+$ ]]; then
    echo "P2A_MAX_SCENES_PER_SHARD must be a non-negative integer" >&2
    exit 2
  fi
  EXTRA+=(--max-scenes "$MAX_SCENES_PER_SHARD")
fi

python - <<'PY'
import json
from pathlib import Path

R = Path("reports/care3d/p2a_online_query_association")
progress = json.loads((R / "progress_manifest.json").read_text())
validation = json.loads((R / "source_validation.json").read_text())
equivalence_path = R / "engineering_smoke/shared_dataset_equivalence.json"
if progress.get("status") not in {
    "P2A_ENGINEERING_SMOKE_PASSED",
    "P2A_TRAIN_EXTRACTION_RUNNING",
}:
    raise RuntimeError(f"unexpected P2-A state before train extraction: {progress.get('status')}")
if progress.get("probe_test_read") is not False:
    raise RuntimeError("P2-A progress indicates probe-test access")
if validation.get("probe_test_read") is not False:
    raise RuntimeError("P2-A source validation indicates probe-test access")
if not equivalence_path.exists():
    raise RuntimeError(
        "run scripts/check_care3d_p2a_shared_dataset_equivalence.py before v2 train extraction"
    )
equivalence = json.loads(equivalence_path.read_text())
if equivalence.get("status") != "P2A_SHARED_DATASET_EQUIVALENCE_PASSED":
    raise RuntimeError("P2-A shared-dataset equivalence did not pass")
if equivalence.get("all_model_inputs_exact") is not True:
    raise RuntimeError("P2-A shared-dataset model inputs are not exact")
if equivalence.get("probe_test_read") is not False:
    raise RuntimeError("P2-A shared-dataset validation indicates probe-test access")
print("PASS: P2-A probe-train extraction is eligible and probe-test is locked")
print("PASS: shared-info dataset execution equivalence is frozen")
PY

echo "===== host memory before worker launch ====="
free -h || true
echo "===== swap ====="
swapon --show || true

pids=()
for ((i=0; i<WORKERS; i++)); do
  log="$LOG_DIR/worker_${i}_of_${WORKERS}.log"
  echo "starting shard $i/$WORKERS -> $log"
  CUDA_VISIBLE_DEVICES=0 python scripts/export_care3d_p2a_train_fast.py \
    --num-shards "$WORKERS" \
    --shard-index "$i" \
    --device "$DEVICE" \
    "${EXTRA[@]}" \
    > "$log" 2>&1 &
  pids+=("$!")
done

failed=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  if wait "$pid"; then
    echo "worker $i complete"
  else
    echo "worker $i FAILED; inspect $LOG_DIR/worker_${i}_of_${WORKERS}.log" >&2
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

# A pilot run intentionally leaves formal extraction incomplete.  Do not
# pretend it is complete and do not run configuration selection.
if [[ -n "$MAX_SCENES_PER_SHARD" ]]; then
  echo "P2A_PARALLEL_PILOT_COMPLETE"
  echo "No shared progress update or config selection was performed."
  exit 0
fi

# Canonical single-process refresh: with max-scenes=0 no detector forward is
# performed; the base exporter only aggregates all already-written markers.
python scripts/export_care3d_p2a_association.py \
  --split probe_train \
  --max-scenes 0 \
  --device "$DEVICE"

python - <<'PY'
import json
from pathlib import Path
R = Path("reports/care3d/p2a_online_query_association")
p = json.loads((R / "progress_manifest.json").read_text())
stage = p.get("stages", {}).get("probe_train_extraction", {})
print(json.dumps({
    "status": p.get("status"),
    "completed_scenes": stage.get("completed_scenes"),
    "expected_scenes": stage.get("expected_scenes"),
    "eligible_rows": stage.get("eligible_rows"),
    "probe_test_read": p.get("probe_test_read"),
}, indent=2, sort_keys=True))
if int(stage.get("completed_scenes", -1)) != 419:
    raise RuntimeError("P2-A parallel train extraction did not complete 419 scenes")
if p.get("status") != "P2A_TRAIN_EXTRACTION_COMPLETE_SELECTION_ELIGIBLE":
    raise RuntimeError(f"unexpected final train-extraction status: {p.get('status')}")
if p.get("probe_test_read") is not False:
    raise RuntimeError("probe-test was accessed during P2-A train extraction")
print("P2A_PARALLEL_TRAIN_EXTRACTION_COMPLETE")
PY
