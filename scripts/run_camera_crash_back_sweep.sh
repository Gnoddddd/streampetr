#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${EVIDENCE3D_PROJECT_ROOT:-$PWD}"
cd "$PROJECT_ROOT"

CONFIG="configs/evidence_conserving/mini_debug.py"
CHECKPOINT="outputs/exp_003_evidence_conserving/iter_4000.pth"
OUT_DIR="outputs/protocol_evaluations/camera_crash_back_sweep"

PROTOCOLS=(
  "protocols/presets/camera_crash_back_1f.json"
  "protocols/presets/camera_crash_back_3f.json"
  "protocols/presets/camera_crash_back_5f.json"
  "protocols/presets/camera_crash_back_10f.json"
  "protocols/presets/camera_crash_back_20f.json"
)

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[失败] 权重不存在：$CHECKPOINT"
  exit 1
fi

for protocol in "${PROTOCOLS[@]}"; do
  if [[ ! -f "$protocol" ]]; then
    echo "[失败] 协议不存在：$protocol"
    exit 1
  fi
done

# 重新开始，不保留上一次未完成的扫描结果。
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/results"

unset EVIDENCE3D_PROTOCOL
unset EVIDENCE3D_PROTOCOL_DEBUG

echo "配置文件：$CONFIG"
echo "模型权重：$CHECKPOINT"
echo "协议数量：${#PROTOCOLS[@]}"
echo

for protocol in "${PROTOCOLS[@]}"; do
  name="$(basename "$protocol" .json)"
  log="$OUT_DIR/${name}.log"
  saved_result="$OUT_DIR/results/${name}_results_nusc.json"

  echo "=================================================="
  echo "开始评测：$name"
  echo "协议文件：$protocol"
  echo "=================================================="

  unset EVIDENCE3D_PROTOCOL
  unset EVIDENCE3D_PROTOCOL_DEBUG

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python tools/evaluate.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --eval bbox \
    --protocol "$protocol" \
    2>&1 | tee "$log"

  # tools/test.py会在日志中明确输出结果文件相对路径。
  relative_result="$(
    grep -oE \
      'Results writes to .*/results_nusc\.json' \
      "$log" \
      | tail -n 1 \
      | sed 's/^Results writes to //'
  )"

  if [[ -z "$relative_result" ]]; then
    echo "[失败] 日志中没有找到结果文件路径：$log"
    exit 1
  fi

  if [[ "$relative_result" = /* ]]; then
    result_path="$relative_result"
  else
    result_path="$PROJECT_ROOT/repos/StreamPETR/$relative_result"
  fi

  if [[ ! -f "$result_path" ]]; then
    echo "[失败] 结果文件不存在：$result_path"
    exit 1
  fi

  cp "$result_path" "$saved_result"

  if grep -niE \
    "Traceback|AssertionError|NCCL|RuntimeError:|CUDA error|out of memory" \
    "$log"; then
    echo "[失败] 评测日志中存在异常：$log"
    exit 1
  fi

  echo "[完成] $name"
  echo "日志：$log"
  echo "预测：$saved_result"
  echo
done

echo "=================================================="
echo "五组CAM_BACK持续故障评测全部完成"
echo "=================================================="

python - <<'PY'
import csv
import json
import re
from pathlib import Path

root = Path(
    "outputs/protocol_evaluations/"
    "camera_crash_back_sweep"
)

clean_log = Path(
    "outputs/protocol_evaluations/"
    "fresh_compare/clean.log"
)

clean_result_path = Path(
    "outputs/protocol_evaluations/"
    "fresh_compare/clean_results_nusc.json"
)

output_csv = root / "camera_crash_back_sweep.csv"


def load_results(path):
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload.get("results", payload)


def parse_metrics(path):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    metrics = {}

    for name in (
        "mAP",
        "NDS",
        "mATE",
        "mASE",
        "mAOE",
        "mAVE",
        "mAAE",
    ):
        matches = re.findall(
            rf"(?m)^{name}:\s*([0-9.]+)",
            text,
        )
        metrics[name] = (
            float(matches[-1])
            if matches else None
        )

    car_matches = re.findall(
        r"(?m)^car\s+([0-9.]+)\s+",
        text,
    )
    metrics["car_AP"] = (
        float(car_matches[-1])
        if car_matches else None
    )

    return metrics


def count_boxes(results):
    return sum(
        len(detections)
        for detections in results.values()
        if isinstance(detections, list)
    )


if not clean_log.is_file():
    raise SystemExit(
        f"缺少干净场景日志：{clean_log}"
    )

if not clean_result_path.is_file():
    raise SystemExit(
        f"缺少干净预测结果：{clean_result_path}"
    )

clean_results = load_results(clean_result_path)
rows = []

clean_metrics = parse_metrics(clean_log)
clean_metrics.update({
    "scenario": "clean",
    "duration_frames": 0,
    "total_boxes": count_boxes(clean_results),
    "changed_samples": 0,
})
rows.append(clean_metrics)

for duration in (1, 3, 5, 10, 20):
    name = f"camera_crash_back_{duration}f"
    log_path = root / f"{name}.log"
    result_path = (
        root
        / "results"
        / f"{name}_results_nusc.json"
    )

    if not log_path.is_file():
        raise SystemExit(f"缺少日志：{log_path}")

    if not result_path.is_file():
        raise SystemExit(f"缺少预测结果：{result_path}")

    results = load_results(result_path)
    tokens = set(clean_results) | set(results)

    changed_samples = sum(
        clean_results.get(token) != results.get(token)
        for token in tokens
    )

    metrics = parse_metrics(log_path)
    metrics.update({
        "scenario": name,
        "duration_frames": duration,
        "total_boxes": count_boxes(results),
        "changed_samples": changed_samples,
    })
    rows.append(metrics)

columns = [
    "scenario",
    "duration_frames",
    "mAP",
    "NDS",
    "mATE",
    "mASE",
    "mAOE",
    "mAVE",
    "mAAE",
    "car_AP",
    "total_boxes",
    "changed_samples",
]

with output_csv.open(
    "w",
    encoding="utf-8",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=columns,
    )
    writer.writeheader()
    writer.writerows(rows)

print("\n持续时间扫描汇总：")

for row in rows:
    print(
        f"{row['duration_frames']:>2}帧",
        f"mAP={row['mAP']}",
        f"NDS={row['NDS']}",
        f"car_AP={row['car_AP']}",
        f"预测框={row['total_boxes']}",
        f"变化样本={row['changed_samples']}",
    )

print("\nCSV：", output_csv)
PY
