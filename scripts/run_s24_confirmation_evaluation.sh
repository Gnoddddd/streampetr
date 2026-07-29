#!/usr/bin/env bash
set -euo pipefail

project_root="$(git rev-parse --show-toplevel)"
output_root="$project_root/outputs/stage2/s2_4_baseline_confirmation/eval"

if [[ -e "$output_root" ]]; then
  printf 'Refusing to overwrite existing evaluation output: %s\n' \
    "$output_root" >&2
  exit 1
fi

modes=(c0 c1)
configs=(
  "$project_root/configs/evidence_conserving/mini_stage2_canonical_no_discount_200.py"
  "$project_root/configs/evidence_conserving/mini_stage2_legacy_fixed_discount_200.py"
)
checkpoints=(
  "$project_root/outputs/stage2/s2_4_baseline_confirmation/c0_200/iter_200.pth"
  "$project_root/outputs/stage2/s2_4_baseline_confirmation/c1_200/iter_200.pth"
)
protocol_names=(clean crash5 crash10 compound)
protocols=(
  "$project_root/protocols/presets/clean_no_corruption.json"
  "$project_root/protocols/presets/camera_crash_back_5f.json"
  "$project_root/protocols/presets/camera_crash_back_10f.json"
  "$project_root/protocols/presets/compound_fog_crash_10f.json"
)

mkdir -p "$output_root"
git rev-parse HEAD >"$output_root/git_commit.txt"

for mode_index in "${!modes[@]}"; do
  mode="${modes[$mode_index]}"
  config="${configs[$mode_index]}"
  checkpoint="${checkpoints[$mode_index]}"
  if [[ ! -s "$checkpoint" ]]; then
    printf 'Missing checkpoint for %s: %s\n' "$mode" "$checkpoint" >&2
    exit 1
  fi
  for protocol_index in "${!protocol_names[@]}"; do
    protocol_name="${protocol_names[$protocol_index]}"
    protocol="${protocols[$protocol_index]}"
    run_dir="$output_root/$mode/$protocol_name"
    mkdir -p "$run_dir"
    printf '%s\n' "$config" >"$run_dir/config.txt"
    printf '%s\n' "$checkpoint" >"$run_dir/checkpoint.txt"
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

    metrics="$run_dir/nuscenes_results/pts_bbox/metrics_summary.json"
    if [[ ! -s "$run_dir/predictions.pkl" ||
          ! -s "$run_dir/trace.jsonl" ||
          ! -s "$metrics" ]]; then
      printf 'Incomplete evaluation output: %s/%s\n' \
        "$mode" "$protocol_name" >&2
      exit 1
    fi
  done
done

printf 'S2.4 baseline confirmation evaluation completed: %s\n' \
  "$output_root"
