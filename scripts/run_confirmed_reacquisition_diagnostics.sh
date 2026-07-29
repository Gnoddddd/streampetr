#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/research/research/evidence3d"
OUTPUT_ROOT="$ROOT/outputs/stage2/s2_3_confirmed_reacquisition_diagnosis"
S22_CHECKPOINT="$ROOT/outputs/stage2/s2_2_source_ledger_debug_50/iter_50.pth"
B4_CHECKPOINT="$ROOT/outputs/stage2/s2_3_rescue/debug_50/b4/iter_50.pth"
B6_CHECKPOINT="$ROOT/outputs/stage2/s2_3_rescue/debug_50/b6/iter_50.pth"

# This matrix is intentionally limited to the already-public development
# protocols. It contains no holdout, additional seed, training, or S2.4 entry.
CANDIDATES=(
  "b0:configs/evidence_conserving/mini_stage2_reacquisition_b0.py:$S22_CHECKPOINT"
  "b4_zero:configs/evidence_conserving/mini_stage2_reacquisition_b4.py:$S22_CHECKPOINT"
  "b6_zero:configs/evidence_conserving/mini_stage2_reacquisition_b6.py:$S22_CHECKPOINT"
  "b4_50iter:configs/evidence_conserving/mini_stage2_reacquisition_b4_debug50.py:$B4_CHECKPOINT"
  "b6_50iter:configs/evidence_conserving/mini_stage2_reacquisition_b6_debug50.py:$B6_CHECKPOINT"
)
PROTOCOLS=(
  "clean:protocols/presets/clean_no_corruption.json"
  "camera_crash_5:protocols/presets/camera_crash_back_5f.json"
  "camera_crash_10:protocols/presets/camera_crash_back_10f.json"
  "compound:protocols/presets/compound_fog_crash_10f.json"
)

for candidate_spec in "${CANDIDATES[@]}"; do
  IFS=: read -r candidate config checkpoint <<< "$candidate_spec"
  for protocol_spec in "${PROTOCOLS[@]}"; do
    IFS=: read -r protocol_name protocol <<< "$protocol_spec"
    run_dir="$OUTPUT_ROOT/$candidate/$protocol_name"
    mkdir -p "$run_dir/traces"
    export EVIDENCE3D_EVAL_TRACE="$run_dir/traces"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      python "$ROOT/tools/evaluate.py" \
        --config "$config" \
        --checkpoint "$checkpoint" \
        --protocol "$protocol" \
        --reacquisition-diagnostics \
        --eval bbox \
        -- \
        --out "$run_dir/predictions.pkl" \
        2>&1 | tee "$run_dir/evaluation.log"
  done
done

unset EVIDENCE3D_EVAL_TRACE
