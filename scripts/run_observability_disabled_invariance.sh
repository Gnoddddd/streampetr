#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
checkpoint="$root/checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth"
output_root="$root/outputs/stage3/observability_distillation/disabled_invariance"
for model in b0 disabled; do
  if [[ "$model" == "b0" ]]; then
    config="configs/stage3/mini_observability_b0.py"
  else
    config="configs/stage3/mini_observability_disabled.py"
  fi
  for protocol in clean_no_corruption camera_crash_back_5f camera_crash_back_10f compound_fog_crash_10f; do
    target="$output_root/$model/$protocol"
    mkdir -p "$target"
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
python scripts/check_prediction_invariance.py \
  --root "$output_root" \
  --output reports/stage3/observability_distillation/disabled_invariance.csv
