#!/usr/bin/env bash
set -uo pipefail

root="$(git rev-parse --show-toplevel)"
checkpoint="$root/outputs/stage3/observability_distillation/b0/iter_969.pth"
config="configs/stage4/hard_positive_boundary_disabled.py"
run_root="$root/outputs/stage4/hard_positive_boundary_objective_audit/disabled"
overall=0

run_one() {
  local name="$1" protocol="${2:-}" target="$run_root/$1"
  local -a command=(python tools/infer_frozen.py --config "$config"
    --checkpoint "$checkpoint" --split val --out "$target/predictions.pkl"
    --json-prefix "$target/formatted")
  if [[ -n "$protocol" ]]; then command+=(--protocol "$protocol"); fi
  mkdir -p "$target"
  if [[ -f "$target/exit_code.txt" ]]; then
    printf 'Refusing duplicate disabled inference: %s\n' "$name" >&2; overall=20; return
  fi
  printf '%q ' "${command[@]}" > "$target/command.txt"; printf '\n' >> "$target/command.txt"
  git rev-parse HEAD > "$target/git_head.txt"
  sha256sum "$checkpoint" > "$target/checkpoint_sha256.txt"
  date --iso-8601=seconds > "$target/started_at.txt"
  "${command[@]}" >"$target/stdout_stderr.log" 2>&1
  local status=$?
  printf '%s\n' "$status" > "$target/exit_code.txt"
  date --iso-8601=seconds > "$target/finished_at.txt"
  if [[ "$status" -ne 0 ]]; then overall="$status"; fi
}

run_one clean
run_one dark_back protocols/presets/dark_back_10f_s09.json
run_one blur_back protocols/presets/motion_blur_back_10f_s09.json
run_one crash_back protocols/presets/camera_crash_back_10f.json
if [[ "$overall" -eq 0 ]]; then python scripts/check_hard_positive_disabled.py; overall=$?; fi
exit "$overall"

