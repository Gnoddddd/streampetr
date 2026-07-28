#!/usr/bin/env python3

import csv
import statistics
from collections import defaultdict
from pathlib import Path


SOURCE = Path(
    "outputs/fixed_v2_summary/"
    "recovery_sensitivity_all.csv"
)

OUTPUT_DIR = Path(
    "outputs/fixed_v2_summary/final_recovery"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def parse_delay(value):
    text = str(value or "").strip()

    if not text:
        return None

    return int(float(text))


with SOURCE.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as file:
    rows = list(csv.DictReader(file))

if not rows:
    raise RuntimeError("敏感性分析CSV为空")


overall = defaultdict(list)
by_protocol = defaultdict(list)

for row in rows:
    delay = parse_delay(
        row.get("robust_recovery_delay")
    )

    if delay is None:
        continue

    model = row["model"]
    setting = row["setting"]

    protocol = row["experiment"].replace(
        "fixed_v2_",
        "",
    )

    overall[(model, setting)].append(delay)
    by_protocol[
        (model, setting, protocol)
    ].append(delay)


overall_rows = []

for (model, setting), delays in sorted(
    overall.items()
):
    overall_rows.append({
        "model": model,
        "setting": setting,
        "cases": len(delays),
        "mean_delay": sum(delays) / len(delays),
        "median_delay": statistics.median(delays),
        "max_delay": max(delays),
        "min_delay": min(delays),
        "nonzero_cases": sum(
            delay > 0
            for delay in delays
        ),
        "zero_cases": sum(
            delay == 0
            for delay in delays
        ),
    })


overall_csv = (
    OUTPUT_DIR
    / "recovery_overall_summary.csv"
)

with overall_csv.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(
            overall_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(overall_rows)


protocol_rows = []

for key, delays in sorted(
    by_protocol.items()
):
    model, setting, protocol = key

    protocol_rows.append({
        "model": model,
        "setting": setting,
        "protocol": protocol,
        "scenes": len(delays),
        "mean_delay": sum(delays) / len(delays),
        "max_delay": max(delays),
        "min_delay": min(delays),
    })


protocol_csv = (
    OUTPUT_DIR
    / "recovery_by_protocol.csv"
)

with protocol_csv.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(
            protocol_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(protocol_rows)


markdown = (
    OUTPUT_DIR
    / "recovery_overall_summary.md"
)

with markdown.open(
    "w",
    encoding="utf-8",
) as file:
    file.write(
        "| 模型 | 设置 | 场景数 | "
        "平均延迟 | 中位数 | 最大延迟 | "
        "非零场景 |\n"
    )

    file.write(
        "|---|---|---:|---:|---:|---:|---:|\n"
    )

    for row in overall_rows:
        file.write(
            f"| {row['model']} "
            f"| {row['setting']} "
            f"| {row['cases']} "
            f"| {row['mean_delay']:.2f} "
            f"| {row['median_delay']:.2f} "
            f"| {row['max_delay']} "
            f"| {row['nonzero_cases']} |\n"
        )


print("[完成]", overall_csv)
print("[完成]", protocol_csv)
print("[完成]", markdown)
print()
print(markdown.read_text(encoding="utf-8"))
