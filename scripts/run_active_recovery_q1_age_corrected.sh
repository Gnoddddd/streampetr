#!/usr/bin/env bash
set -uo pipefail

root="$(git rev-parse --show-toplevel)"
checkpoint="$root/outputs/stage3/observability_distillation/b0/iter_969.pth"
run_root="$root/outputs/stage3/active_recovery_query_injection/Q1_age_corrected"
status=0
for item in \
  "Clean:" \
  "Crash5:protocols/presets/camera_crash_back_5f.json" \
  "Crash10:protocols/presets/camera_crash_back_10f.json" \
  "Compound:protocols/presets/compound_fog_crash_10f.json" \
  "NaturalRecovery:protocols/counterfactual_view_deficit/val_natural_recovery.json"
do
  name="${item%%:*}"
  protocol="${item#*:}"
  target="$run_root/$name"
  mkdir -p "$target"
  command=(
    python tools/infer_frozen.py
    --config configs/stage3/active_recovery_b0_audit.py
    --checkpoint "$checkpoint"
    --split val
    --out "$target/predictions.pkl"
    --json-prefix "$target/formatted"
  )
  if [[ -n "$protocol" ]]; then
    command+=(--protocol "$protocol")
  fi
  ACTIVE_RECOVERY_MODE=Q1 \
  ACTIVE_RECOVERY_TRACE_DIR="$target/trace" \
    "${command[@]}" >"$target/stdout_stderr.log" 2>&1
  value=$?
  printf '%s\n' "$value" >"$target/exit_code.txt"
  if [[ "$value" -ne 0 ]]; then
    status="$value"
  fi
done
exit "$status"
