#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${EVIDENCE3D_ROOT:-$HOME/research/evidence3d}"
cd "$PROJECT_ROOT"
if [[ "${CONDA_DEFAULT_ENV:-}" != "streampetr" ]]; then
  echo "Activate the model environment first: conda activate streampetr" >&2
  exit 2
fi
unset EVIDENCE3D_DISABLE_RANDOM_CORRUPTION || true
python tools/train.py --config configs/streampetr/baseline_corrupted_mini.py "$@"
