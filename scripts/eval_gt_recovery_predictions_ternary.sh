#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "用法：$0 <实验名称> <协议文件>"
  exit 1
fi

NAME="$1"
PROTOCOL_INPUT="$2"

ROOT="$HOME/research/evidence3d"
CONFIG="$ROOT/configs/evidence_conserving/mini_debug_ternary_ft400_fp32.py"
CHECKPOINT="$ROOT/outputs/mini_debug_ternary_ft400_fp32/iter_400.pth"

if [ ! -f "$PROTOCOL_INPUT" ]; then
  echo "[错误] 协议不存在：$PROTOCOL_INPUT"
  exit 1
fi

PROTOCOL="$(realpath "$PROTOCOL_INPUT")"

OUT_ROOT="$ROOT/outputs/gt_recovery_predictions/ternary_ft400_fp32"
EXP_DIR="$OUT_ROOT/$NAME"

PREDICTIONS="$EXP_DIR/predictions.pkl"
JSON_PREFIX="$EXP_DIR/nuscenes_results"
LOG="$EXP_DIR/evaluation.log"

for file in \
  "$PROTOCOL" \
  "$CONFIG" \
  "$CHECKPOINT" \
  "$ROOT/tools/evaluate.py" \
  "$ROOT/repos/StreamPETR/tools/test.py"
do
  if [ ! -f "$file" ]; then
    echo "[错误] 运行所需文件不存在：$file"
    exit 1
  fi
done

rm -rf "$EXP_DIR"

mkdir -p \
  "$EXP_DIR" \
  "$JSON_PREFIX"

cp "$PROTOCOL" \
  "$EXP_DIR/protocol_used.json"

cp "$CONFIG" \
  "$EXP_DIR/config_used.py"

cp "$ROOT/datasets/corruption.py" \
  "$EXP_DIR/corruption_used.py"

cp "$ROOT/repos/StreamPETR/tools/test.py" \
  "$EXP_DIR/test_used.py"

printf '%s\n' "$PROTOCOL" \
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

sha256sum "$ROOT/datasets/corruption.py" \
  | awk '{print $1}' \
  > "$EXP_DIR/corruption.sha256"

sha256sum "$ROOT/repos/StreamPETR/tools/test.py" \
  | awk '{print $1}' \
  > "$EXP_DIR/test_script.sha256"

export EVIDENCE3D_PROJECT_ROOT="$ROOT"
export EVIDENCE3D_ROOT="$ROOT"
export PYTHONPATH="$ROOT/repos/StreamPETR:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# 协议必须由tools/evaluate.py的--protocol参数统一注入。
# 清除外部残留，避免旧协议污染本次实验。
unset EVIDENCE3D_PROTOCOL
unset EVIDENCE3D_VISUAL_ABLATION_MODE
unset EVIDENCE3D_PROTOCOL_DEBUG
unset EVIDENCE3D_EVAL_TRACE

cd "$ROOT"

echo "============================================================"
echo "实验名称：$NAME"
echo "协议文件：$PROTOCOL"
echo "配置文件：$CONFIG"
echo "模型权重：$CHECKPOINT"
echo "预测PKL：$PREDICTIONS"
echo "JSON前缀：$JSON_PREFIX"
echo "============================================================"

set +e

CUDA_VISIBLE_DEVICES=0 \
python tools/evaluate.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --protocol "$PROTOCOL" \
  --eval bbox \
  -- \
  --out "$PREDICTIONS" \
  --eval bbox \
  --eval-options \
  "jsonfile_prefix=$JSON_PREFIX" \
  2>&1 | tee "$LOG"

STATUS="${PIPESTATUS[0]}"

set -e

if [ "$STATUS" -ne 0 ]; then
  echo
  echo "[错误] 评测失败，退出状态：$STATUS"
  echo "日志：$LOG"
  exit "$STATUS"
fi

if [ ! -s "$PREDICTIONS" ]; then
  echo "[错误] predictions.pkl未生成或为空：$PREDICTIONS"
  exit 1
fi

RESULT_JSON="$(
  find "$EXP_DIR" \
    -type f \
    -name 'results_nusc.json' \
    | sort \
    | head -n 1
)"

if [ -z "$RESULT_JSON" ]; then
  echo "[错误] 未生成results_nusc.json"
  echo "当前目录文件："
  find "$EXP_DIR" -maxdepth 6 -type f -printf '%P\n' | sort
  exit 1
fi

# 固定复制到实验根目录，方便后续GT恢复脚本统一读取。
if [ "$RESULT_JSON" != "$EXP_DIR/results_nusc.json" ]; then
  cp "$RESULT_JSON" \
    "$EXP_DIR/results_nusc.json"
fi

python - \
  "$PREDICTIONS" \
  "$EXP_DIR/results_nusc.json" <<'PY'
import json
import sys
from pathlib import Path

import mmcv


pkl_path = Path(sys.argv[1])
json_path = Path(sys.argv[2])

outputs = mmcv.load(str(pkl_path))

if isinstance(outputs, list):
    prediction_samples = len(outputs)
elif isinstance(outputs, dict):
    lengths = []

    for value in outputs.values():
        try:
            lengths.append(len(value))
        except TypeError:
            continue

    prediction_samples = max(lengths) if lengths else 0
else:
    prediction_samples = 0

payload = json.loads(
    json_path.read_text(encoding="utf-8")
)

results = payload.get("results", {})

print()
print("预测PKL顶层类型：", type(outputs).__name__)
print("预测PKL样本数：", prediction_samples)
print("结果JSON样本数：", len(results))
print(
    "结果JSON预测框总数：",
    sum(len(boxes) for boxes in results.values()),
)

if prediction_samples != 81:
    raise SystemExit(
        f"预测PKL样本数异常：{prediction_samples}，预期81"
    )

if len(results) != 81:
    raise SystemExit(
        f"结果JSON样本数异常：{len(results)}，预期81"
    )

print("预测文件完整性检查通过")
PY

cat > "$EXP_DIR/prediction_metadata.json" <<EOF
{
  "experiment": "$NAME",
  "protocol": "$PROTOCOL",
  "config": "$CONFIG",
  "checkpoint": "$CHECKPOINT",
  "predictions": "$PREDICTIONS",
  "results_nusc": "$EXP_DIR/results_nusc.json",
  "entrypoint": "tools/evaluate.py",
  "upstream_entrypoint": "repos/StreamPETR/tools/test.py"
}
EOF

echo
echo "[完成] $NAME"
echo "PKL：$PREDICTIONS"
echo "JSON：$EXP_DIR/results_nusc.json"
echo "日志：$LOG"
