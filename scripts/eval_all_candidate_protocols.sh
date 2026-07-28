#!/usr/bin/env bash
set -uo pipefail

ROOT="$HOME/research/evidence3d"
RUN_ONE="$ROOT/scripts/eval_candidate_protocol.sh"
OUT_ROOT="$ROOT/outputs/protocol_evaluations/full_candidate"
FAILURES="$OUT_ROOT/failures.tsv"

cd "$ROOT"

mkdir -p "$OUT_ROOT"
: > "$FAILURES"

if [ ! -x "$RUN_ONE" ]; then
  echo "[错误] 单协议评测脚本不存在或不可执行：$RUN_ONE"
  exit 1
fi

mapfile -t PROTOCOLS < <(
  find "$ROOT/protocols" \
    -type f \
    \( -name '*.json' -o -name '*.yaml' -o -name '*.yml' \) \
    | sort
)

TOTAL="${#PROTOCOLS[@]}"

if [ "$TOTAL" -eq 0 ]; then
  echo "[错误] protocols目录中没有找到协议文件"
  exit 1
fi

echo "共发现 $TOTAL 个协议"
echo

INDEX=0

for PROTOCOL in "${PROTOCOLS[@]}"; do
  INDEX=$((INDEX + 1))

  RELATIVE="${PROTOCOL#"$ROOT/protocols/"}"
  NAME="${RELATIVE%.*}"
  NAME="${NAME//\//__}"
  NAME="${NAME// /_}"

  EXP_DIR="$OUT_ROOT/$NAME"
  LOG="$EXP_DIR/evaluation.log"

  echo "============================================================"
  echo "[$INDEX/$TOTAL] $NAME"
  echo "协议：$PROTOCOL"

  if [ -f "$LOG" ] && grep -q '^NDS:' "$LOG"; then
    echo "[跳过] 已存在完整评测结果"
    echo
    continue
  fi

  mkdir -p "$EXP_DIR"

  if "$RUN_ONE" "$NAME" "$PROTOCOL"; then
    echo "[成功] $NAME"
  else
    STATUS=$?
    printf '%s\t%s\t%s\n' \
      "$NAME" \
      "$STATUS" \
      "$PROTOCOL" \
      >> "$FAILURES"

    echo "[失败] $NAME，退出码：$STATUS"
  fi

  echo
done

echo "============================================================"
echo "批量评测结束"

if [ -s "$FAILURES" ]; then
  echo "以下协议运行失败："
  cat "$FAILURES"
else
  echo "所有协议均运行成功"
fi
