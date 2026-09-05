#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_care3d_full_once.sh \
    --data-root /path/to/full/nuscenes \
    --checkpoint /path/to/checkpoint.pth \
    [--gpu 0] [--skip-prepare]

What it does:
  1. checks the full nuScenes v1.0-trainval mount;
  2. prepares StreamPETR temporal info files when missing;
  3. runs unit tests;
  4. evaluates Clean + CAM_BACK Crash + Dark + Motion Blur on full nuScenes val;
  5. saves logs, nuScenes result JSON files, metrics.csv, manifest.txt and git_state.txt.

The script does not enable CARE-3D routing. It is the reproducible full-data
baseline/P0 data-generation entry point for the current gated branch.
EOF
}

PROJECT_ROOT="${EVIDENCE3D_ROOT:-$HOME/research/evidence3d}"
DATA_ROOT=""
CHECKPOINT=""
GPU="0"
SKIP_PREPARE="0"
CONFIG="configs/care3d/full_official_r50_900q_eval.py"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --skip-prepare)
      SKIP_PREPARE="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$DATA_ROOT" || -z "$CHECKPOINT" ]]; then
  usage
  exit 2
fi

cd "$PROJECT_ROOT"
DATA_ROOT="$(readlink -f "$DATA_ROOT")"
CHECKPOINT="$(readlink -f "$CHECKPOINT")"

if [[ "${CONDA_DEFAULT_ENV:-}" != "streampetr" ]]; then
  echo "[ERROR] Activate the streampetr environment first: conda activate streampetr" >&2
  exit 2
fi

for required in maps samples sweeps v1.0-trainval; do
  if [[ ! -e "$DATA_ROOT/$required" ]]; then
    echo "[ERROR] Missing full nuScenes path: $DATA_ROOT/$required" >&2
    exit 2
  fi
done

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] Checkpoint not found: $CHECKPOINT" >&2
  exit 2
fi

export EVIDENCE3D_DATA_ROOT="$DATA_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"

if [[ "$SKIP_PREPARE" != "1" ]]; then
  bash scripts/prepare_nuscenes_full.sh "$DATA_ROOT"
else
  test -f "$DATA_ROOT/nuscenes2d_temporal_infos_train.pkl"
  test -f "$DATA_ROOT/nuscenes2d_temporal_infos_val.pkl"
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="reports/care3d/full_nuscenes/$RUN_ID"
mkdir -p "$OUT_DIR/results" "$OUT_DIR/logs"

{
  echo "run_id=$RUN_ID"
  echo "project_root=$PROJECT_ROOT"
  echo "data_root=$DATA_ROOT"
  echo "checkpoint=$CHECKPOINT"
  echo "config=$CONFIG"
  echo "gpu=$GPU"
  echo "conda_env=${CONDA_DEFAULT_ENV:-}"
  echo "python=$(command -v python)"
  python --version 2>&1 | sed 's/^/python_version=/'
} > "$OUT_DIR/manifest.txt"

{
  git rev-parse HEAD
  git branch --show-current
  git status --short
} > "$OUT_DIR/git_state.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$OUT_DIR/nvidia_smi.txt" || true
fi

printf '%s\n' \
  "scenario,protocol" \
  "clean," \
  "crash_back,protocols/presets/camera_crash_back_10f.json" \
  "dark_back,protocols/presets/dark_back_10f_s09.json" \
  "motion_blur_back,protocols/presets/motion_blur_back_10f_s09.json" \
  > "$OUT_DIR/scenarios.csv"

echo "[1/5] Running unit tests..."
pytest -q 2>&1 | tee "$OUT_DIR/logs/pytest.log"

evaluate_one() {
  local name="$1"
  local protocol="$2"
  local log="$OUT_DIR/logs/${name}.log"
  local saved="$OUT_DIR/results/${name}_results_nusc.json"

  echo "=================================================="
  echo "[EVAL] $name"
  if [[ -n "$protocol" ]]; then
    echo "[PROTOCOL] $protocol"
  else
    echo "[PROTOCOL] clean"
  fi
  echo "=================================================="

  if [[ -n "$protocol" ]]; then
    python tools/evaluate.py \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --eval bbox \
      --protocol "$protocol" \
      2>&1 | tee "$log"
  else
    python tools/evaluate.py \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --eval bbox \
      2>&1 | tee "$log"
  fi

  if grep -niE "Traceback|AssertionError|RuntimeError:|CUDA error|out of memory" "$log"; then
    echo "[ERROR] Evaluation log contains an exception: $log" >&2
    exit 1
  fi

  local relative_result
  relative_result="$(
    grep -oE 'Results writes to .*/results_nusc\.json' "$log" \
      | tail -n 1 \
      | sed 's/^Results writes to //'
  )"

  if [[ -z "$relative_result" ]]; then
    echo "[ERROR] Could not locate results_nusc.json from log: $log" >&2
    exit 1
  fi

  local result_path
  if [[ "$relative_result" = /* ]]; then
    result_path="$relative_result"
  else
    result_path="$PROJECT_ROOT/repos/StreamPETR/$relative_result"
  fi

  if [[ ! -f "$result_path" ]]; then
    echo "[ERROR] Result file not found: $result_path" >&2
    exit 1
  fi

  cp "$result_path" "$saved"
  echo "[OK] Saved: $saved"
}

echo "[2/5] Clean evaluation..."
evaluate_one "clean" ""

echo "[3/5] Crash evaluation..."
evaluate_one "crash_back" "protocols/presets/camera_crash_back_10f.json"

echo "[4/5] Dark evaluation..."
evaluate_one "dark_back" "protocols/presets/dark_back_10f_s09.json"

echo "[5/5] Motion Blur evaluation..."
evaluate_one "motion_blur_back" "protocols/presets/motion_blur_back_10f_s09.json"

python - "$OUT_DIR" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
scenarios = ["clean", "crash_back", "dark_back", "motion_blur_back"]

metric_names = ["mAP", "NDS", "mATE", "mASE", "mAOE", "mAVE", "mAAE"]


def parse_metrics(log_path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    output = {}
    for name in metric_names:
        matches = re.findall(rf"(?m)^{name}:\s*([0-9.]+)", text)
        output[name] = float(matches[-1]) if matches else None
    return output


def count_boxes(path):
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload.get("results", payload)
    return sum(len(v) for v in results.values() if isinstance(v, list))

rows = []
clean = None
for scenario in scenarios:
    log_path = root / "logs" / f"{scenario}.log"
    result_path = root / "results" / f"{scenario}_results_nusc.json"
    metrics = parse_metrics(log_path)
    metrics["scenario"] = scenario
    metrics["total_boxes"] = count_boxes(result_path)
    if scenario == "clean":
        clean = dict(metrics)
    rows.append(metrics)

for row in rows:
    if clean is not None:
        for name in ("mAP", "NDS"):
            value = row.get(name)
            clean_value = clean.get(name)
            row[f"delta_{name}"] = (
                None if value is None or clean_value is None else value - clean_value
            )

fieldnames = [
    "scenario", "mAP", "NDS", "delta_mAP", "delta_NDS",
    "mATE", "mASE", "mAOE", "mAVE", "mAAE", "total_boxes",
]
with (root / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

with (root / "summary.json").open("w", encoding="utf-8") as handle:
    json.dump(rows, handle, ensure_ascii=False, indent=2)

print("\n=== CARE-3D full nuScenes summary ===")
for row in rows:
    print(
        f"{row['scenario']:18s} "
        f"mAP={row.get('mAP')} NDS={row.get('NDS')} "
        f"d_mAP={row.get('delta_mAP')} d_NDS={row.get('delta_NDS')}"
    )
PY

cp "$CONFIG" "$OUT_DIR/config_snapshot.py"

cat <<EOF

==================================================
CARE-3D full nuScenes run completed.
Results directory:
  $PROJECT_ROOT/$OUT_DIR

Key files:
  $OUT_DIR/metrics.csv
  $OUT_DIR/summary.json
  $OUT_DIR/results/*.json
  $OUT_DIR/logs/*.log
  $OUT_DIR/manifest.txt
  $OUT_DIR/git_state.txt
==================================================
EOF
