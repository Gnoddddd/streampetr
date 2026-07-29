#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"
checkpoint="outputs/stage2/s2_2_source_ledger_debug_50/iter_50.pth"
out_root="outputs/stage3/raydn_screening/disabled_invariance"

protocols=(
  clean_no_corruption
  camera_crash_back_5f
  camera_crash_back_10f
  compound_fog_crash_10f
)
pairs=(
  "B0:configs/stage3/mini_raydn_b0_50.py:configs/stage3/mini_raydn_b0_disabled_eval.py"
  "M1:configs/stage3/mini_raydn_m1_50.py:configs/stage3/mini_raydn_m1_raydn_50.py"
)

for entry in "${pairs[@]}"; do
  IFS=: read -r pair baseline_config raydn_config <<<"$entry"
  for protocol in "${protocols[@]}"; do
    protocol_file="protocols/presets/${protocol}.json"
    for variant in baseline disabled; do
      if [[ "$variant" == baseline ]]; then
        config="$baseline_config"
      else
        config="$raydn_config"
      fi
      target="$out_root/$pair/$protocol/$variant"
      mkdir -p "$target"
      python tools/evaluate.py \
        --config "$config" \
        --checkpoint "$checkpoint" \
        --protocol "$protocol_file" \
        --eval bbox \
        -- \
        --out "$target/predictions.pkl" \
        2>&1 | tee "$target/evaluation.log"
    done
  done
done

