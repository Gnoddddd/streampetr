#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${EVIDENCE3D_ROOT:-$HOME/research/evidence3d}"
cd "$PROJECT_ROOT"
python -m pytest -q tests
python tools/diagnose.py
if [[ "${RUN_MODEL_SMOKE:-0}" == "1" ]]; then
  python tools/train.py --config configs/evidence_conserving/mini_smoke.py --no-validate
fi
