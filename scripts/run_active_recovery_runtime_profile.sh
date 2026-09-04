#!/usr/bin/env bash
set -uo pipefail

root="$(git rev-parse --show-toplevel)"
checkpoint="$root/outputs/stage3/observability_distillation/b0/iter_969.pth"
target_root="$root/outputs/stage3/active_recovery_query_injection/runtime_profile"
status=0
for group in Q1 Q2 Q3; do
  target="$target_root/$group"
  mkdir -p "$target"
  ACTIVE_RECOVERY_MODE="$group" \
  ACTIVE_RECOVERY_TRACE_DIR="$target/trace" \
    python tools/infer_frozen.py \
      --config configs/stage3/active_recovery_b0_audit.py \
      --checkpoint "$checkpoint" \
      --split val \
      --protocol protocols/presets/camera_crash_back_5f.json \
      --out "$target/predictions.pkl" \
      --json-prefix "$target/formatted" \
      >"$target/stdout_stderr.log" 2>&1
  value=$?
  printf '%s\n' "$value" >"$target/exit_code.txt"
  if [[ "$value" -ne 0 ]]; then
    status="$value"
  fi
done
exit "$status"
