#!/usr/bin/env bash
set -euo pipefail

project_root="$(git rev-parse --show-toplevel)"
output_root="$project_root/outputs/stage2/s2_4_isolation_audit/inference"
checkpoint="$project_root/outputs/stage2/s2_2_source_ledger_debug_50/iter_50.pth"

if [[ -e "$output_root" ]]; then
  printf 'Refusing to overwrite existing inference output: %s\n' \
    "$output_root" >&2
  exit 1
fi
if [[ ! -f "$checkpoint" ]]; then
  printf 'Missing S2.2 checkpoint: %s\n' "$checkpoint" >&2
  exit 1
fi

mkdir -p "$output_root"
git rev-parse HEAD >"$output_root/git_commit.txt"
printf '%s\n' "$checkpoint" >"$output_root/checkpoint.txt"

modes=(legacy disabled)
configs=(
  "$project_root/configs/evidence_conserving/mini_stage2_correlation_discount_legacy.py"
  "$project_root/configs/evidence_conserving/mini_stage2_correlation_discount_disabled.py"
)
protocol_names=(clean crash5 crash10 compound)
protocols=(
  "$project_root/protocols/presets/clean_no_corruption.json"
  "$project_root/protocols/presets/camera_crash_back_5f.json"
  "$project_root/protocols/presets/camera_crash_back_10f.json"
  "$project_root/protocols/presets/compound_fog_crash_10f.json"
)

for mode_index in "${!modes[@]}"; do
  mode="${modes[$mode_index]}"
  config="${configs[$mode_index]}"
  if [[ ! -f "$config" ]]; then
    printf 'Missing audit config: %s\n' "$config" >&2
    exit 1
  fi
  for protocol_index in "${!protocol_names[@]}"; do
    protocol_name="${protocol_names[$protocol_index]}"
    protocol="${protocols[$protocol_index]}"
    run_dir="$output_root/$mode/$protocol_name"
    mkdir -p "$run_dir"
    printf '%s\n' "$config" >"$run_dir/config.txt"
    printf '%s\n' "$protocol" >"$run_dir/protocol.txt"

    EVIDENCE3D_S24_ISOLATION_AUDIT=1 \
    EVIDENCE3D_EVAL_TRACE="$run_dir/trace.jsonl" \
    EVIDENCE3D_PROTOCOL_DEBUG=1 \
    CUDA_VISIBLE_DEVICES=0 \
      python "$project_root/tools/evaluate.py" \
        --config "$config" \
        --checkpoint "$checkpoint" \
        --protocol "$protocol" \
        --eval bbox \
        -- \
        --out "$run_dir/predictions.pkl" \
        --eval-options "jsonfile_prefix=$run_dir/nuscenes_results"

    if [[ ! -s "$run_dir/predictions.pkl" ||
          ! -s "$run_dir/trace.jsonl" ]]; then
      printf 'Incomplete inference output: %s/%s\n' \
        "$mode" "$protocol_name" >&2
      exit 1
    fi
  done
done

printf 'S2.4 isolation inference completed: %s\n' "$output_root"

