#!/usr/bin/env python3

import csv
import json
from pathlib import Path


ROOT = Path.home() / "research/evidence3d"

EVAL_ROOT = (
    ROOT
    / "outputs/protocol_evaluations"
    / "full_candidate"
)

METRIC_PATH = (
    EVAL_ROOT
    / "candidate_protocol_summary.csv"
)

OUTPUT_PATH = (
    EVAL_ROOT
    / "fog_pixel_quality_ablation_summary.csv"
)

CASES = [
    {
        "mode": "full",
        "severity": 0.6,
        "experiment": "presets__fog_back_10f",
    },
    {
        "mode": "pixel_only",
        "severity": 0.6,
        "experiment": (
            "presets__fog_back_10f_s06_pixel_only"
        ),
    },
    {
        "mode": "quality_only",
        "severity": 0.6,
        "experiment": (
            "presets__fog_back_10f_s06_quality_only"
        ),
    },
    {
        "mode": "full",
        "severity": 0.9,
        "experiment": (
            "presets__fog_back_10f_s09"
        ),
    },
    {
        "mode": "pixel_only",
        "severity": 0.9,
        "experiment": (
            "presets__fog_back_10f_s09_pixel_only"
        ),
    },
    {
        "mode": "quality_only",
        "severity": 0.9,
        "experiment": (
            "presets__fog_back_10f_s09_quality_only"
        ),
    },
]


def flatten(value):
    if isinstance(value, list):
        result = []

        for item in value:
            result.extend(flatten(item))

        return result

    return [value]


def read_metric(row, requested_name):
    """兼容CarAP、car_ap、Car AP等字段写法。"""

    def normalize(name):
        return (
            str(name)
            .strip()
            .lower()
            .replace("_", "")
            .replace(" ", "")
            .replace("-", "")
        )

    target = normalize(requested_name)

    for key, value in row.items():
        if normalize(key) != target:
            continue

        if value is None:
            continue

        value = str(value).strip()

        if not value:
            continue

        return float(value)

    raise KeyError(
        f"找不到指标字段{requested_name!r}；"
        f"当前CSV字段为{list(row.keys())}"
    )


with METRIC_PATH.open(
    "r",
    encoding="utf-8-sig",
) as file:
    metric_rows = list(
        csv.DictReader(file)
    )

metrics = {
    row["experiment"]: row
    for row in metric_rows
}

rows = []

for case in CASES:
    experiment = case["experiment"]
    mode = case["mode"]
    severity = case["severity"]

    experiment_dir = (
        EVAL_ROOT / experiment
    )

    protocol_path = (
        experiment_dir
        / "protocol_used.json"
    )

    if not protocol_path.is_file():
        raise FileNotFoundError(
            protocol_path
        )

    protocol = json.loads(
        protocol_path.read_text(
            encoding="utf-8"
        )
    )

    events = protocol["scenes"]["*"]

    if len(events) != 1:
        raise RuntimeError(
            f"{experiment}应只有一个事件"
        )

    event = events[0]
    start = int(event["start_frame"])
    end = int(event["end_frame"])

    counts = {
        "active": {
            "keep": 0,
            "recover": 0,
            "defer": 0,
            "total": 0,
        },
        "post": {
            "keep": 0,
            "recover": 0,
            "defer": 0,
            "total": 0,
        },
    }

    trace_paths = sorted(
        (
            experiment_dir
            / "traces"
        ).glob("*.jsonl")
    )

    if not trace_paths:
        raise RuntimeError(
            f"{experiment}没有JSONL轨迹"
        )

    for trace_path in trace_paths:
        with trace_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line)
                frame = int(
                    record["frame_idx"]
                )

                if start <= frame <= end:
                    phase = "active"
                elif frame > end:
                    phase = "post"
                else:
                    continue

                diagnostics = record[
                    "diagnostics"
                ]

                prior = flatten(
                    diagnostics[
                        "prior_strength"
                    ]
                )
                actions = flatten(
                    diagnostics["action"]
                )

                for prior_value, action in zip(
                    prior,
                    actions,
                ):
                    if float(prior_value) <= 1e-6:
                        continue

                    action = int(action)

                    counts[phase]["total"] += 1

                    if action == 0:
                        counts[phase]["keep"] += 1
                    elif action == 1:
                        counts[phase]["recover"] += 1
                    elif action == 2:
                        counts[phase]["defer"] += 1
                    else:
                        raise RuntimeError(
                            f"未知动作：{action}"
                        )

    metric = metrics.get(experiment)

    if metric is None:
        raise RuntimeError(
            f"汇总表中没有实验：{experiment}"
        )

    active_total = counts["active"]["total"]
    post_total = counts["post"]["total"]

    pixel_modified = (
        mode in {"full", "pixel_only"}
    )

    quality_reduced = (
        mode in {"full", "quality_only"}
    )

    camera_quality = (
        max(0.05, 1.0 - severity)
        if quality_reduced
        else 1.0
    )

    rows.append({
        "experiment": experiment,
        "mode": mode,
        "severity": severity,
        "pixel_modified": pixel_modified,
        "quality_reduced": quality_reduced,
        "camera_quality": camera_quality,
        "mAP": read_metric(metric, "mAP"),
        "NDS": read_metric(metric, "NDS"),
        "CarAP": read_metric(metric, "CarAP"),
        "active_queries": active_total,
        "active_keep": (
            counts["active"]["keep"]
            / active_total
        ),
        "active_recover": (
            counts["active"]["recover"]
            / active_total
        ),
        "active_defer": (
            counts["active"]["defer"]
            / active_total
        ),
        "post_queries": post_total,
        "post_keep": (
            counts["post"]["keep"]
            / post_total
        ),
        "post_recover": (
            counts["post"]["recover"]
            / post_total
        ),
        "post_defer": (
            counts["post"]["defer"]
            / post_total
        ),
    })


fields = list(rows[0].keys())

with OUTPUT_PATH.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=fields,
    )
    writer.writeheader()
    writer.writerows(rows)


print(
    f"{'强度':>6s}"
    f"{'模式':15s}"
    f"{'像素':>6s}"
    f"{'质量':>8s}"
    f"{'mAP':>9s}"
    f"{'NDS':>9s}"
    f"{'A-K':>9s}"
    f"{'A-R':>9s}"
    f"{'A-D':>9s}"
    f"{'P-K':>9s}"
    f"{'P-R':>9s}"
)

print("-" * 98)

for row in rows:
    print(
        f"{row['severity']:6.1f}"
        f"{row['mode']:15s}"
        f"{str(row['pixel_modified']):>6s}"
        f"{row['camera_quality']:8.2f}"
        f"{row['mAP']:9.4f}"
        f"{row['NDS']:9.4f}"
        f"{row['active_keep']:9.4f}"
        f"{row['active_recover']:9.4f}"
        f"{row['active_defer']:9.4f}"
        f"{row['post_keep']:9.4f}"
        f"{row['post_recover']:9.4f}"
    )

print()
print("A-K/A-R/A-D：退化阶段Keep/Recover/Defer")
print("P-K/P-R：恢复阶段Keep/Recover")
print("已生成：", OUTPUT_PATH)
