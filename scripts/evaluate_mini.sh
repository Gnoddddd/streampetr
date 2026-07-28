#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${EVIDENCE3D_ROOT:-$HOME/research/evidence3d}"
CHECKPOINT="${1:?Usage: scripts/evaluate_mini.sh CHECKPOINT [PROTOCOL_JSON]}"
PROTOCOL="${2:-}"
cd "$PROJECT_ROOT"
args=(--config configs/evidence_conserving/mini_train.py --checkpoint "$CHECKPOINT" --eval bbox)
if [[ -n "$PROTOCOL" ]]; then
  args+=(--protocol "$PROTOCOL")
fi
python tools/evaluate.py "${args[@]}"
