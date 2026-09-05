#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${EVIDENCE3D_ROOT:-$HOME/research/evidence3d}"
DATA_ROOT="${1:-${EVIDENCE3D_DATA_ROOT:-$PROJECT_ROOT/data/nuscenes}}"
STREAM_DIR="$PROJECT_ROOT/repos/StreamPETR"

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

if [[ ! -d "$STREAM_DIR" ]]; then
  echo "[ERROR] StreamPETR repository not found: $STREAM_DIR" >&2
  exit 2
fi

TRAIN_INFO="$DATA_ROOT/nuscenes2d_temporal_infos_train.pkl"
VAL_INFO="$DATA_ROOT/nuscenes2d_temporal_infos_val.pkl"

if [[ -f "$TRAIN_INFO" && -f "$VAL_INFO" && "${CARE3D_FORCE_PREPARE:-0}" != "1" ]]; then
  echo "[OK] Full nuScenes temporal infos already exist."
  echo "     $TRAIN_INFO"
  echo "     $VAL_INFO"
  exit 0
fi

export EVIDENCE3D_DATA_ROOT="$DATA_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$STREAM_DIR:$STREAM_DIR/mmdetection3d:${PYTHONPATH:-}"

cd "$STREAM_DIR"
echo "[INFO] Preparing full nuScenes v1.0-trainval under: $DATA_ROOT"
python tools/create_data_nusc.py \
  --root-path "$DATA_ROOT" \
  --out-dir "$DATA_ROOT" \
  --extra-tag nuscenes2d \
  --version v1.0-trainval \
  --max-sweeps 10

test -f "$TRAIN_INFO"
test -f "$VAL_INFO"

echo "[OK] Full nuScenes temporal infos created:"
echo "     $TRAIN_INFO"
echo "     $VAL_INFO"
