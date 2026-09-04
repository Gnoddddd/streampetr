#!/usr/bin/env bash
set -uo pipefail

root="$(git rev-parse --show-toplevel)"
checkpoint="$root/outputs/stage3/observability_distillation/b0/iter_969.pth"
run_root="$root/outputs/stage3/active_recovery_query_injection"
base_config="configs/stage3/mini_convergence_b0.py"
audit_config="configs/stage3/active_recovery_b0_audit.py"
overall_status=0

declare -A protocols=(
  [Clean]=""
  [Crash5]="protocols/presets/camera_crash_back_5f.json"
  [Crash10]="protocols/presets/camera_crash_back_10f.json"
  [Compound]="protocols/presets/compound_fog_crash_10f.json"
  [NaturalRecovery]="protocols/counterfactual_view_deficit/val_natural_recovery.json"
)

run_one() {
  local group="$1"
  local protocol_name="$2"
  local config="$3"
  local protocol="${protocols[$protocol_name]}"
  local target="$run_root/$group/$protocol_name"
  local -a command=(
    python tools/infer_frozen.py
    --config "$config"
    --checkpoint "$checkpoint"
    --split val
    --out "$target/predictions.pkl"
    --json-prefix "$target/formatted"
  )
  if [[ -n "$protocol" ]]; then
    command+=(--protocol "$protocol")
  fi
  mkdir -p "$target"
  if [[ -f "$target/exit_code.txt" ]]; then
    printf 'Refusing duplicate inference: %s/%s\n' "$group" "$protocol_name" >&2
    overall_status=20
    return
  fi
  printf '%q ' "${command[@]}" > "$target/command.txt"
  printf '\n' >> "$target/command.txt"
  git rev-parse HEAD > "$target/git_head.txt"
  sha256sum "$checkpoint" > "$target/checkpoint_sha256.txt"
  date --iso-8601=seconds > "$target/started_at.txt"
  if [[ "$group" == "Q0" ]]; then
    "${command[@]}" >"$target/stdout_stderr.log" 2>&1
  else
    ACTIVE_RECOVERY_MODE="$group" \
    ACTIVE_RECOVERY_TRACE_DIR="$target/trace" \
      "${command[@]}" >"$target/stdout_stderr.log" 2>&1
  fi
  local status=$?
  printf '%s\n' "$status" > "$target/exit_code.txt"
  date --iso-8601=seconds > "$target/finished_at.txt"
  if [[ "$status" -ne 0 ]]; then
    overall_status="$status"
  fi
}

# Explicit disabled-path run.
run_one InjectionOff Clean "$audit_config"

for protocol_name in Clean Crash5 Crash10 Compound NaturalRecovery; do
  run_one Q0 "$protocol_name" "$base_config"
done

for group in Q1 Q2 Q3; do
  for protocol_name in Clean Crash5 Crash10 Compound NaturalRecovery; do
    run_one "$group" "$protocol_name" "$audit_config"
  done
done

exit "$overall_status"
