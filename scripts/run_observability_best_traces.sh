#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
trace_root="$root/outputs/stage3/observability_distillation/best_traces"
experiments=(
  "B0:configs/stage3/mini_observability_b0.py:b0:969"
  "R0:configs/stage3/mini_observability_r0.py:r0:323"
  "R1:configs/stage3/mini_observability_r1.py:r1:323"
)
for entry in "${experiments[@]}"; do
  IFS=: read -r experiment config directory iteration <<<"$entry"
  checkpoint="$root/outputs/stage3/observability_distillation/$directory/iter_$iteration.pth"
  for protocol in clean_no_corruption camera_crash_back_5f camera_crash_back_10f compound_fog_crash_10f; do
    target="$trace_root/$experiment/$protocol"
    mkdir -p "$target"
    rm -f "$target/memory_writes.jsonl"
    S3_R1_MEMORY_TRACE="$target/memory_writes.jsonl" \
      python tools/evaluate.py \
        --config "$config" \
        --checkpoint "$checkpoint" \
        --protocol "protocols/presets/$protocol.json" \
        --eval bbox -- \
        --out "$target/predictions.pkl" \
        --eval-options "jsonfile_prefix=$target/nuscenes_results" \
        >"$target/evaluation.log" 2>&1
  done
done
