#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "用法："
  echo "$0 <实验名称> <协议文件> <full|pixel_only|quality_only>"
  exit 1
fi

NAME="$1"
PROTOCOL="$2"
MODE="$3"

case "$MODE" in
  full|pixel_only|quality_only)
    ;;
  *)
    echo "[错误] 不支持的消融模式：$MODE"
    exit 1
    ;;
esac

ROOT="$HOME/research/evidence3d"

if [ ! -f "$ROOT/$PROTOCOL" ] && [ ! -f "$PROTOCOL" ]; then
  echo "[错误] 协议文件不存在：$PROTOCOL"
  exit 1
fi

export EVIDENCE3D_VISUAL_ABLATION_MODE="$MODE"

echo "============================================================"
echo "实验名称：$NAME"
echo "消融模式：$MODE"
echo "协议文件：$PROTOCOL"
echo "============================================================"

"$ROOT/scripts/eval_candidate_protocol.sh" \
  "$NAME" \
  "$PROTOCOL"

EXP_DIR="$ROOT/outputs/protocol_evaluations/full_candidate/$NAME"

if [ ! -d "$EXP_DIR" ]; then
  echo "[错误] 评测目录未生成：$EXP_DIR"
  exit 1
fi

printf '%s\n' \
  "$MODE" \
  > "$EXP_DIR/visual_ablation_mode.txt"

cp \
  "$ROOT/datasets/corruption.py" \
  "$EXP_DIR/corruption_used.py"

sha256sum \
  "$ROOT/datasets/corruption.py" \
  > "$EXP_DIR/corruption.sha256"

cat > "$EXP_DIR/visual_ablation_metadata.json" <<EOF
{
  "mode": "$MODE",
  "experiment": "$NAME",
  "protocol": "$PROTOCOL",
  "environment_variable": "EVIDENCE3D_VISUAL_ABLATION_MODE"
}
EOF

{
  echo
  echo "[VisualAblation] mode=$MODE"
  echo "[VisualAblation] protocol=$PROTOCOL"
} >> "$EXP_DIR/evaluation.log"

echo
echo "[完成] $NAME"
echo "消融模式：$MODE"
echo "结果目录：$EXP_DIR"
