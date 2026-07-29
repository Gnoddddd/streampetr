#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"
out_root="outputs/stage3/mini_convergence_loss_balance/eval"
protocols=(
  clean_no_corruption
  camera_crash_back_5f
  camera_crash_back_10f
  compound_fog_crash_10f
)
experiments=(
  "B0:configs/stage3/mini_convergence_b0.py:b0"
  "M1:configs/stage3/mini_convergence_m1.py:m1"
  "M1-Ramp:configs/stage3/mini_convergence_m1_ramp.py:m1_ramp"
)
milestones=(
  "1:323"
  "3:969"
  "6:1938"
  "12:3876"
)

for experiment_entry in "${experiments[@]}"; do
  IFS=: read -r experiment config directory <<<"$experiment_entry"
  for milestone_entry in "${milestones[@]}"; do
    IFS=: read -r epoch iteration <<<"$milestone_entry"
    checkpoint="outputs/stage3/mini_convergence_loss_balance/$directory/iter_$iteration.pth"
    test -f "$checkpoint"
    for protocol in "${protocols[@]}"; do
      target="$root/$out_root/$experiment/epoch_$epoch/$protocol"
      mkdir -p "$target"
      EVIDENCE3D_EVAL_TRACE="$target/traces" \
      python tools/evaluate.py \
        --config "$config" \
        --checkpoint "$checkpoint" \
        --protocol "protocols/presets/${protocol}.json" \
        --eval bbox \
        -- \
        --out "$target/predictions.pkl" \
        --eval-options "jsonfile_prefix=$target/nuscenes_results" \
        2>&1 | tee "$target/evaluation.log"
    done
  done
done
