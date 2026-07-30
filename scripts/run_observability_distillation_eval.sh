#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
eval_root="$root/outputs/stage3/observability_distillation/eval"
experiments=(
  "B0:configs/stage3/mini_observability_b0.py:b0"
  "R0:configs/stage3/mini_observability_r0.py:r0"
  "R1:configs/stage3/mini_observability_r1.py:r1"
)
milestones=("1:323" "3:969" "6:1938")
protocols=(
  clean_no_corruption
  camera_crash_back_5f
  camera_crash_back_10f
  compound_fog_crash_10f
)
for entry in "${experiments[@]}"; do
  IFS=: read -r experiment config directory <<<"$entry"
  for milestone in "${milestones[@]}"; do
    IFS=: read -r epoch iteration <<<"$milestone"
    checkpoint="$root/outputs/stage3/observability_distillation/$directory/iter_$iteration.pth"
    test -f "$checkpoint"
    for protocol in "${protocols[@]}"; do
      target="$eval_root/$experiment/epoch_$epoch/$protocol"
      mkdir -p "$target"
      /usr/bin/time -f 'wall_seconds=%e max_rss_kb=%M' \
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
done
