#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "用法："
  echo "bash scripts/run_protocol_group.sh GROUP_NAME REGEX"
  exit 1
fi

GROUP_NAME="$1"
REGEX="$2"

PROJECT_ROOT="${EVIDENCE3D_PROJECT_ROOT:-$PWD}"
cd "$PROJECT_ROOT"

CONFIG="configs/evidence_conserving/mini_debug.py"
CHECKPOINT="outputs/exp_003_evidence_conserving/iter_4000.pth"
OUT_DIR="outputs/protocol_evaluations/$GROUP_NAME"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[失败] 权重不存在：$CHECKPOINT"
  exit 1
fi

mapfile -t PROTOCOLS < <(
  find protocols/presets \
    -maxdepth 1 \
    -type f \
    -name "*.json" \
    | grep -Ei "$REGEX" \
    | sort -V
)

if [[ "${#PROTOCOLS[@]}" -eq 0 ]]; then
  echo "[失败] 没有找到匹配协议"
  echo "正则表达式：$REGEX"
  exit 1
fi

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/results"

unset EVIDENCE3D_PROTOCOL
unset EVIDENCE3D_PROTOCOL_DEBUG

echo "协议组：$GROUP_NAME"
echo "协议数量：${#PROTOCOLS[@]}"

printf '%s\n' "${PROTOCOLS[@]}"
echo

for protocol in "${PROTOCOLS[@]}"; do
  name="$(basename "$protocol" .json)"
  log="$OUT_DIR/${name}.log"
  saved_result="$OUT_DIR/results/${name}_results_nusc.json"

  echo "=================================================="
  echo "开始评测：$name"
  echo "协议：$protocol"
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

  relative_result="$(
    grep -oE \
      'Results writes to .*/results_nusc\.json' \
      "$log" \
      | tail -n 1 \
      | sed 's/^Results writes to //'
  )"

  if [[ -z "$relative_result" ]]; then
    echo "[失败] 日志中没有结果文件路径"
    exit 1
  fi

  if [[ "$relative_result" = /* ]]; then
    result_path="$relative_result"
  else
    result_path="$PROJECT_ROOT/repos/StreamPETR/$relative_result"
  fi

  if [[ ! -f "$result_path" ]]; then
    echo "[失败] 结果不存在：$result_path"
    exit 1
  fi

  cp "$result_path" "$saved_result"

  if grep -niE \
    "Traceback|AssertionError|NCCL|RuntimeError:|CUDA error|out of memory" \
    "$log"; then
    echo "[失败] 日志存在异常"
    exit 1
  fi

  echo "[完成] $name"
  echo
done

python - "$GROUP_NAME" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

group_name = sys.argv[1]

root = Path(
    "outputs/protocol_evaluations"
) / group_name

clean_log = Path(
    "outputs/protocol_evaluations/fresh_compare/clean.log"
)

clean_result_path = Path(
    "outputs/protocol_evaluations/fresh_compare/"
    "clean_results_nusc.json"
)

output_csv = root / f"{group_name}.csv"


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

    for metric in (
        "mAP",
        "NDS",
        "mATE",
        "mASE",
        "mAOE",
        "mAVE",
        "mAAE",
    ):
        matches = re.findall(
            rf"(?m)^{metric}:\s*([0-9.]+)",
            text,
        )
        metrics[metric] = (
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


clean_results = load_results(clean_result_path)

rows = []

clean_metrics = parse_metrics(clean_log)
clean_metrics.update({
    "scenario": "clean",
    "total_boxes": count_boxes(clean_results),
    "changed_samples": 0,
})
rows.append(clean_metrics)

for log_path in sorted(root.glob("*.log")):
    name = log_path.stem

    result_path = (
        root
        / "results"
        / f"{name}_results_nusc.json"
    )

    if not result_path.is_file():
        continue

    results = load_results(result_path)
    tokens = set(clean_results) | set(results)

    changed_samples = sum(
        clean_results.get(token) != results.get(token)
        for token in tokens
    )

    metrics = parse_metrics(log_path)
    metrics.update({
        "scenario": name,
        "total_boxes": count_boxes(results),
        "changed_samples": changed_samples,
    })
    rows.append(metrics)

columns = [
    "scenario",
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

print("\n评测结果：")

for row in rows:
    print(
        row["scenario"],
        f"mAP={row['mAP']}",
        f"NDS={row['NDS']}",
        f"car_AP={row['car_AP']}",
        f"框数={row['total_boxes']}",
        f"变化样本={row['changed_samples']}",
    )

print("\nCSV：", output_csv)
PY

echo "[完成] 协议组评测：$GROUP_NAME"
