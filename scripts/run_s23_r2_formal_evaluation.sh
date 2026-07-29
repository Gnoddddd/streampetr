#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/research/research/evidence3d"
OUTPUT_ROOT="$ROOT/outputs/stage2/s2_3_r2_formal/formal_evaluation"
S22_CHECKPOINT="$ROOT/outputs/stage2/s2_2_source_ledger_debug_50/iter_50.pth"
R2A_CHECKPOINT="$ROOT/outputs/stage2/s2_3_r2_formal/debug_50/r2_a/iter_50.pth"
R2B_CHECKPOINT="$ROOT/outputs/stage2/s2_3_r2_formal/debug_50/r2_b/iter_50.pth"

# This is the complete pre-declared R2 matrix. It deliberately contains no
# holdout, additional seed, 200-iteration run, teacher, or S2.4 entry.
CANDIDATES=(
  "r2_a_zero:configs/evidence_conserving/mini_stage2_r2_a_isolation.py:$S22_CHECKPOINT"
  "r2_b_zero:configs/evidence_conserving/mini_stage2_r2_b_confirmed.py:$S22_CHECKPOINT"
  "r2_a_50iter:configs/evidence_conserving/mini_stage2_r2_a_debug50.py:$R2A_CHECKPOINT"
  "r2_b_50iter:configs/evidence_conserving/mini_stage2_r2_b_debug50.py:$R2B_CHECKPOINT"
)
PROTOCOLS=(
  "clean_no_corruption:protocols/presets/clean_no_corruption.json"
  "camera_crash_back_5f:protocols/presets/camera_crash_back_5f.json"
  "camera_crash_back_10f:protocols/presets/camera_crash_back_10f.json"
  "compound_fog_crash_10f:protocols/presets/compound_fog_crash_10f.json"
)

cd "$ROOT"
for candidate_spec in "${CANDIDATES[@]}"; do
  IFS=: read -r candidate config checkpoint <<< "$candidate_spec"
  if [[ -n "${CANDIDATE_FILTER:-}" && "$candidate" != "$CANDIDATE_FILTER" ]]; then
    continue
  fi
  for protocol_spec in "${PROTOCOLS[@]}"; do
    IFS=: read -r protocol_name protocol <<< "$protocol_spec"
    run_dir="$OUTPUT_ROOT/$candidate/$protocol_name"
    if [[ -e "$run_dir/predictions.pkl" || -e "$run_dir/results_nusc.json" ]]; then
      echo "Refusing to overwrite completed evaluation: $run_dir" >&2
      exit 2
    fi
    mkdir -p "$run_dir/traces" "$run_dir/nuscenes_results"
    cp "$protocol" "$run_dir/protocol_used.json"
    export EVIDENCE3D_EVAL_TRACE="$run_dir/traces"
    echo "FORMAL_EVAL_START candidate=$candidate protocol=$protocol_name"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      python "$ROOT/tools/evaluate.py" \
        --config "$config" \
        --checkpoint "$checkpoint" \
        --protocol "$protocol" \
        --reacquisition-diagnostics \
        --eval bbox \
        -- \
        --out "$run_dir/predictions.pkl" \
        --eval bbox \
        --eval-options "jsonfile_prefix=$run_dir/nuscenes_results" \
        > "$run_dir/evaluation.log" 2>&1
    result_json="$(
      find "$run_dir/nuscenes_results" -type f -name results_nusc.json |
        sort | head -n 1
    )"
    if [[ -z "$result_json" ]]; then
      echo "Missing results_nusc.json: $run_dir" >&2
      exit 3
    fi
    cp "$result_json" "$run_dir/results_nusc.json"
    echo "FORMAL_EVAL_DONE candidate=$candidate protocol=$protocol_name"
  done
done
unset EVIDENCE3D_EVAL_TRACE
