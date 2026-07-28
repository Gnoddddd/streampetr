#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/research/evidence3d"
cd "$ROOT"

B0_ROOT="$ROOT/outputs/gt_recovery_predictions/official_r50_900q_baseline"

CLEAN_PROTOCOL="$(
  cat "$B0_ROOT/clean_no_corruption/protocol_source_path.txt"
)"

CRASH5_PROTOCOL="$(
  cat "$B0_ROOT/camera_crash_back_5f/protocol_source_path.txt"
)"

CRASH10_PROTOCOL="$(
  cat "$B0_ROOT/camera_crash_back_10f/protocol_source_path.txt"
)"

COMPOUND_PROTOCOL="$(
  cat "$B0_ROOT/compound_fog_crash_10f/protocol_source_path.txt"
)"

for protocol in \
  "$CLEAN_PROTOCOL" \
  "$CRASH5_PROTOCOL" \
  "$CRASH10_PROTOCOL" \
  "$COMPOUND_PROTOCOL"
do
  if [ ! -f "$protocol" ]; then
    echo "[错误] 协议不存在：$protocol"
    exit 1
  fi
done

run_one() {
  local script="$1"
  local output_root="$2"
  local experiment="$3"
  local protocol="$4"

  local result="$output_root/$experiment/results_nusc.json"

  echo
  echo "================================================================"
  echo "脚本：$script"
  echo "实验：$experiment"
  echo "协议：$protocol"
  echo "================================================================"

  if [ -s "$result" ]; then
    echo "[跳过] 已存在有效结果：$result"
    return
  fi

  bash "$script" "$experiment" "$protocol"

  if [ ! -s "$result" ]; then
    echo "[错误] 结果没有生成：$result"
    exit 1
  fi

  echo "[完成] $result"
}

B0_OUTPUT="$ROOT/outputs/gt_recovery_predictions/official_r50_900q_baseline"
B1_OUTPUT="$ROOT/outputs/gt_recovery_predictions/classification_active_r50_900q"
T1_OUTPUT="$ROOT/outputs/gt_recovery_predictions/stage1_active_r50_900q"

B0_SCRIPT="$ROOT/scripts/eval_gt_recovery_predictions_official_r50.sh"
B1_SCRIPT="$ROOT/scripts/eval_gt_recovery_predictions_classification_active_r50.sh"
T1_SCRIPT="$ROOT/scripts/eval_gt_recovery_predictions_stage1_active.sh"

for script in "$B0_SCRIPT" "$B1_SCRIPT" "$T1_SCRIPT"
do
  if [ ! -f "$script" ]; then
    echo "[错误] 评测脚本不存在：$script"
    exit 1
  fi
done

run_one \
  "$B0_SCRIPT" \
  "$B0_OUTPUT" \
  "fixed_v2_clean_no_corruption" \
  "$CLEAN_PROTOCOL"

run_one \
  "$B0_SCRIPT" \
  "$B0_OUTPUT" \
  "fixed_v2_camera_crash_back_5f" \
  "$CRASH5_PROTOCOL"

run_one \
  "$B0_SCRIPT" \
  "$B0_OUTPUT" \
  "fixed_v2_camera_crash_back_10f" \
  "$CRASH10_PROTOCOL"

run_one \
  "$B0_SCRIPT" \
  "$B0_OUTPUT" \
  "fixed_v2_compound_fog_crash_10f" \
  "$COMPOUND_PROTOCOL"

run_one \
  "$B1_SCRIPT" \
  "$B1_OUTPUT" \
  "fixed_v2_clean_no_corruption" \
  "$CLEAN_PROTOCOL"

run_one \
  "$B1_SCRIPT" \
  "$B1_OUTPUT" \
  "fixed_v2_camera_crash_back_5f" \
  "$CRASH5_PROTOCOL"

run_one \
  "$B1_SCRIPT" \
  "$B1_OUTPUT" \
  "fixed_v2_camera_crash_back_10f" \
  "$CRASH10_PROTOCOL"

run_one \
  "$B1_SCRIPT" \
  "$B1_OUTPUT" \
  "fixed_v2_compound_fog_crash_10f" \
  "$COMPOUND_PROTOCOL"

run_one \
  "$T1_SCRIPT" \
  "$T1_OUTPUT" \
  "fixed_v2_clean_no_corruption" \
  "$CLEAN_PROTOCOL"

run_one \
  "$T1_SCRIPT" \
  "$T1_OUTPUT" \
  "fixed_v2_camera_crash_back_5f" \
  "$CRASH5_PROTOCOL"

run_one \
  "$T1_SCRIPT" \
  "$T1_OUTPUT" \
  "fixed_v2_camera_crash_back_10f" \
  "$CRASH10_PROTOCOL"

run_one \
  "$T1_SCRIPT" \
  "$T1_OUTPUT" \
  "fixed_v2_compound_fog_crash_10f" \
  "$COMPOUND_PROTOCOL"

echo
echo "[全部完成] 12组修复后评测已运行完毕"
