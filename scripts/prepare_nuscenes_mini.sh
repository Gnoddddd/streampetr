#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${EVIDENCE3D_ROOT:-$HOME/research/evidence3d}"
DATA_ROOT="${EVIDENCE3D_DATA_ROOT:-$PROJECT_ROOT/data/nuscenes-mini}"
STREAM_DIR="$PROJECT_ROOT/repos/StreamPETR"

for required in maps samples sweeps v1.0-mini; do
  if [[ ! -e "$DATA_ROOT/$required" ]]; then
    echo "Missing nuScenes-mini path: $DATA_ROOT/$required" >&2
    exit 2
  fi
done
if [[ "${CONDA_DEFAULT_ENV:-}" != "streampetr" ]]; then
  echo "Activate the streampetr environment first: conda activate streampetr" >&2
  exit 2
fi

export PYTHONPATH="$PROJECT_ROOT:$STREAM_DIR:$STREAM_DIR/mmdetection3d:${PYTHONPATH:-}"
cd "$STREAM_DIR"
python tools/create_data_nusc.py \
  --root-path "$DATA_ROOT" \
  --out-dir "$DATA_ROOT" \
  --extra-tag nuscenes2d \
  --version v1.0-mini \
  --max-sweeps 10

test -f "$DATA_ROOT/nuscenes2d_temporal_infos_train.pkl"
test -f "$DATA_ROOT/nuscenes2d_temporal_infos_val.pkl"
echo "nuScenes-mini temporal infos created under $DATA_ROOT"
