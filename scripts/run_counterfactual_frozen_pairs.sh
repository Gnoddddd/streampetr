#!/usr/bin/env bash
set -uo pipefail

root="$(git rev-parse --show-toplevel)"
checkpoint="$root/outputs/stage3/observability_distillation/b0/iter_969.pth"
run_root="$root/outputs/stage3/counterfactual_view_deficit_audit"
audit_config="configs/stage3/counterfactual_b0_audit.py"
original_config="configs/stage3/mini_convergence_b0.py"
overall_status=0

run_one() {
  local name="$1"
  local split="$2"
  local config="$3"
  local protocol="${4:-}"
  local trace="${5:-}"
  local target="$run_root/$name"
  local -a command=(
    python tools/infer_frozen.py
    --config "$config"
    --checkpoint "$checkpoint"
    --split "$split"
    --out "$target/predictions.pkl"
    --json-prefix "$target/formatted"
  )
  if [[ -n "$protocol" ]]; then
    command+=(--protocol "$protocol")
  fi
  if [[ -n "$trace" ]]; then
    command+=(--trace-dir "$target/trace")
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

# Three prediction-invariance runs. No trace is enabled.
run_one invariance_original val "$original_config"
run_one invariance_audit val "$audit_config"
run_one invariance_repeat val "$audit_config"

# Frozen paired train inference.
run_one train_full train "$audit_config" "" trace
run_one train_available train "$audit_config" \
  protocols/counterfactual_view_deficit/train_seen.json trace

# Frozen Full val pass and all preregistered Available conditions.
run_one val_full val "$audit_config" "" trace
run_one val_seen val "$audit_config" \
  protocols/counterfactual_view_deficit/val_seen.json trace
run_one val_nonadjacent_double val "$audit_config" \
  protocols/counterfactual_view_deficit/val_nonadjacent_double.json trace
run_one val_three_camera val "$audit_config" \
  protocols/counterfactual_view_deficit/val_three_camera.json trace
run_one val_duration_10 val "$audit_config" \
  protocols/counterfactual_view_deficit/val_duration_10.json trace
run_one val_duration_20 val "$audit_config" \
  protocols/counterfactual_view_deficit/val_duration_20.json trace
run_one val_natural_recovery val "$audit_config" \
  protocols/counterfactual_view_deficit/val_natural_recovery.json trace
run_one val_crash5 val "$audit_config" \
  protocols/presets/camera_crash_back_5f.json trace
run_one val_crash10 val "$audit_config" \
  protocols/presets/camera_crash_back_10f.json trace
run_one val_compound val "$audit_config" \
  protocols/presets/compound_fog_crash_10f.json trace

exit "$overall_status"
