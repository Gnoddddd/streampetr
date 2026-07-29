#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"
out_root="outputs/stage3/raydn_screening/eval"
protocols=(
  clean_no_corruption
  camera_crash_back_5f
  camera_crash_back_10f
  compound_fog_crash_10f
)
experiments=(
  "B0:configs/stage3/mini_raydn_b0_50.py:outputs/stage3/raydn_screening/b0_50/iter_50.pth"
  "B0_RayDN:configs/stage3/mini_raydn_b0_raydn_50.py:outputs/stage3/raydn_screening/b0_raydn_50/iter_50.pth"
  "M1:configs/stage3/mini_raydn_m1_50.py:outputs/stage3/raydn_screening/m1_50/iter_50.pth"
  "M1_RayDN:configs/stage3/mini_raydn_m1_raydn_50.py:outputs/stage3/raydn_screening/m1_raydn_50/iter_50.pth"
)

for entry in "${experiments[@]}"; do
  IFS=: read -r experiment config checkpoint <<<"$entry"
  for protocol in "${protocols[@]}"; do
    target="$root/$out_root/$experiment/$protocol"
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

