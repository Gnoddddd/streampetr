#!/usr/bin/env bash
set -uo pipefail

root="$(git rev-parse --show-toplevel)"
checkpoint="$root/outputs/stage3/observability_distillation/b0/iter_969.pth"
run_root="$root/outputs/stage3/reviewer_proof_recovery_audit"
config="configs/stage3/counterfactual_b0_audit.py"
overall_status=0

run_one() {
  local name="$1"
  local protocol="${2:-}"
  local layers="${3:-}"
  local target="$run_root/$name"
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
  if [[ -n "$layers" ]]; then
    command+=(--trace-dir "$target/trace" --trace-layers)
  fi
  mkdir -p "$target"
  if [[ -f "$target/exit_code.txt" ]]; then
    printf 'Refusing duplicate frozen inference: %s\n' "$name" >&2
    overall_status=20
    return
  fi
  printf '%q ' "${command[@]}" > "$target/command.txt"
  printf '\n' >> "$target/command.txt"
  git rev-parse HEAD > "$target/git_head.txt"
  sha256sum "$checkpoint" > "$target/checkpoint_sha256.txt"
  date --iso-8601=seconds > "$target/started_at.txt"
  "${command[@]}" >"$target/stdout_stderr.log" 2>&1
  local status=$?
  printf '%s\n' "$status" > "$target/exit_code.txt"
  date --iso-8601=seconds > "$target/finished_at.txt"
  if [[ "$status" -ne 0 ]]; then
    overall_status="$status"
  fi
}

run_one invariance_a
run_one invariance_b
run_one clean "" layers
run_one crash5 protocols/presets/camera_crash_back_5f.json layers
run_one crash10 protocols/presets/camera_crash_back_10f.json layers
run_one compound protocols/presets/compound_fog_crash_10f.json layers

exit "$overall_status"
