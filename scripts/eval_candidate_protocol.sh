#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "用法：$0 <实验名称> <协议文件>"
  exit 1
fi

NAME="$1"
PROTOCOL="$(realpath "$2")"

ROOT="$HOME/research/evidence3d"
CONFIG="$ROOT/configs/evidence_conserving/mini_debug_source_aware_ft400_fp32.py"
CHECKPOINT="$ROOT/outputs/exp_006_source_aware_ft400_fp32/iter_400.pth"

OUT_ROOT="$ROOT/outputs/protocol_evaluations/full_candidate"
EXP_DIR="$OUT_ROOT/$NAME"
TRACE_DIR="$EXP_DIR/traces"

for file in \
  "$PROTOCOL" \
  "$CONFIG" \
  "$CHECKPOINT"
do
  if [ ! -f "$file" ]; then
    echo "[错误] 文件不存在：$file"
    exit 1
  fi
done

rm -rf "$EXP_DIR"
mkdir -p "$TRACE_DIR"

cp \
  "$PROTOCOL" \
  "$EXP_DIR/protocol_used.json"

printf '%s\n' \
  "$PROTOCOL" \
  > "$EXP_DIR/protocol_source_path.txt"

sha256sum "$PROTOCOL" \
  | awk '{print $1}' \
  > "$EXP_DIR/protocol.sha256"

sha256sum "$CONFIG" \
  | awk '{print $1}' \
  > "$EXP_DIR/config.sha256"

sha256sum "$CHECKPOINT" \
  | awk '{print $1}' \
  > "$EXP_DIR/checkpoint.sha256"

export EVIDENCE3D_PROJECT_ROOT="$ROOT"
export EVIDENCE3D_ROOT="$ROOT"
export PYTHONPATH="$ROOT/repos/StreamPETR:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

export EVIDENCE3D_EVAL_TRACE="$TRACE_DIR"
export EVIDENCE3D_PROTOCOL_DEBUG=1

unset EVIDENCE3D_PROTOCOL

cd "$ROOT"

CUDA_VISIBLE_DEVICES=0 \
python tools/evaluate.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --protocol "$PROTOCOL" \
  --eval bbox \
  2>&1 | tee "$EXP_DIR/evaluation.log"

unset EVIDENCE3D_EVAL_TRACE
unset EVIDENCE3D_PROTOCOL_DEBUG

echo
echo "[完成] $NAME"
echo "日志：$EXP_DIR/evaluation.log"
echo "轨迹：$TRACE_DIR"
echo "协议副本：$EXP_DIR/protocol_used.json"
echo "协议哈希：$EXP_DIR/protocol.sha256"
