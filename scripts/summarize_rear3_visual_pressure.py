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
    / "rear3_visual_pressure_summary.csv"
)

CASES = []

for corruption in (
    "dark",
    "fog",
    "motion_blur",
):
    CASES.append({
        "corruption": corruption,
        "scope": "single_back",
        "experiment": (
            f"presets__{corruption}"
            f"_back_10f_s09"
        ),
    })

    CASES.append({
        "corruption": corruption,
        "scope": "rear3",
        "experiment": (
            f"presets__{corruption}"
            f"_rear3_10f_s09"
        ),
    })


def normalize(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", "")
        .replace(" ", "")
        .replace("-", "")
    )


def read_metric(row, requested):
    target = normalize(requested)

    for key, value in row.items():
        if normalize(key) != target:
            continue

        if value is None:
            continue

        value = str(value).strip()

        if value:
            return float(value)

    raise KeyError(
        f"找不到指标{requested!r}，"
        f"字段为{list(row.keys())}"
    )


def flatten(value):
    if isinstance(value, list):
        result = []

        for item in value:
            result.extend(flatten(item))

        return result

    return [value]


with METRIC_PATH.open(
    "r",
    encoding="utf-8-sig",
    newline="",
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
    experiment_dir = EVAL_ROOT / experiment

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
            f"{experiment}应有且只有一个事件"
        )

    event = events[0]

    start = int(event["start_frame"])
    end = int(event["end_frame"])

    corruption = case["corruption"]

    affected_cameras = sorted(
        event[corruption]
    )

    counts = {
        "pre": {
            "keep": 0,
            "recover": 0,
            "defer": 0,
            "total": 0,
        },
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
            experiment_dir / "traces"
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

                if frame < start:
                    phase = "pre"
                elif frame <= end:
                    phase = "active"
                else:
                    phase = "post"

                diagnostics = record[
                    "diagnostics"
                ]

                prior = flatten(
                    diagnostics["prior_strength"]
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
                            f"未知动作值：{action}"
                        )

    metric = metrics.get(experiment)

    if metric is None:
        raise RuntimeError(
            f"汇总CSV中没有{experiment}"
        )

    def ratio(phase, action):
        total = counts[phase]["total"]

        if not total:
            return 0.0

        return (
            counts[phase][action]
            / total
        )

    rows.append({
        "experiment": experiment,
        "corruption": corruption,
        "scope": case["scope"],
        "camera_count": len(
            affected_cameras
        ),
        "affected_cameras": " ".join(
            affected_cameras
        ),
        "severity": 0.9,
        "mAP": read_metric(metric, "mAP"),
        "NDS": read_metric(metric, "NDS"),
        "CarAP": read_metric(
            metric,
            "CarAP",
        ),
        "pre_keep": ratio(
            "pre",
            "keep",
        ),
        "pre_recover": ratio(
            "pre",
            "recover",
        ),
        "active_keep": ratio(
            "active",
            "keep",
        ),
        "active_recover": ratio(
            "active",
            "recover",
        ),
        "active_defer": ratio(
            "active",
            "defer",
        ),
        "post_keep": ratio(
            "post",
            "keep",
        ),
        "post_recover": ratio(
            "post",
            "recover",
        ),
        "post_defer": ratio(
            "post",
            "defer",
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
    f"{'退化':14s}"
    f"{'范围':14s}"
    f"{'相机':>6s}"
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
        f"{row['corruption']:14s}"
        f"{row['scope']:14s}"
        f"{row['camera_count']:6d}"
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
