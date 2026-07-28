#!/usr/bin/env python3

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path.home() / "research/evidence3d"

EVAL_ROOT = (
    ROOT
    / "outputs/protocol_evaluations"
    / "full_candidate"
)

OUTPUT = (
    EVAL_ROOT
    / "visual_corruption_phase_summary.csv"
)

CORRUPTIONS = (
    "dark",
    "fog",
    "motion_blur",
)

DURATIONS = (
    5,
    10,
    20,
)


def flatten(value):
    if isinstance(value, list):
        result = []

        for item in value:
            result.extend(flatten(item))

        return result

    return [value]


rows = []

for corruption in CORRUPTIONS:
    for duration in DURATIONS:
        experiment = (
            f"presets__{corruption}_back_{duration}f"
        )

        experiment_dir = EVAL_ROOT / experiment
        protocol_path = (
            experiment_dir
            / "protocol_used.json"
        )
        trace_dir = (
            experiment_dir
            / "traces"
        )

        if not protocol_path.is_file():
            print(
                "[跳过] 缺少协议副本：",
                protocol_path,
            )
            continue

        if not trace_dir.is_dir():
            print(
                "[跳过] 缺少轨迹目录：",
                trace_dir,
            )
            continue

        protocol = json.loads(
            protocol_path.read_text(
                encoding="utf-8"
            )
        )

        events = protocol["scenes"]["*"]

        if len(events) != 1:
            raise RuntimeError(
                f"{experiment}预期只有一个事件，"
                f"实际为{len(events)}个"
            )

        event = events[0]

        start_frame = int(
            event["start_frame"]
        )
        end_frame = int(
            event["end_frame"]
        )

        severity = float(
            event[corruption]["CAM_BACK"]
        )

        counts = defaultdict(
            lambda: {
                "trace_records": 0,
                "propagated": 0,
                "keep": 0,
                "recover": 0,
                "defer": 0,
            }
        )

        trace_files = sorted(
            trace_dir.glob("*.jsonl")
        )

        if not trace_files:
            print(
                "[跳过] 没有JSONL轨迹：",
                trace_dir,
            )
            continue

        for trace_path in trace_files:
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

                    if frame < start_frame:
                        phase = "pre_fault"
                    elif frame <= end_frame:
                        phase = "active_corruption"
                    else:
                        phase = "post_recovery"

                    diagnostics = record.get(
                        "diagnostics",
                        {},
                    )

                    if (
                        "prior_strength"
                        not in diagnostics
                        or "action"
                        not in diagnostics
                    ):
                        continue

                    prior = flatten(
                        diagnostics[
                            "prior_strength"
                        ]
                    )
                    action = flatten(
                        diagnostics["action"]
                    )

                    size = min(
                        len(prior),
                        len(action),
                    )

                    counts[phase][
                        "trace_records"
                    ] += 1

                    for index in range(size):
                        if (
                            float(prior[index])
                            <= 1e-6
                        ):
                            continue

                        counts[phase][
                            "propagated"
                        ] += 1

                        value = int(
                            action[index]
                        )

                        if value == 0:
                            counts[phase][
                                "keep"
                            ] += 1
                        elif value == 1:
                            counts[phase][
                                "recover"
                            ] += 1
                        elif value == 2:
                            counts[phase][
                                "defer"
                            ] += 1
                        else:
                            raise RuntimeError(
                                f"未知动作值：{value}"
                            )

        for phase in (
            "pre_fault",
            "active_corruption",
            "post_recovery",
        ):
            values = counts[phase]
            total = values["propagated"]

            action_total = (
                values["keep"]
                + values["recover"]
                + values["defer"]
            )

            if action_total != total:
                raise RuntimeError(
                    f"{experiment} {phase}："
                    f"动作数{action_total} != "
                    f"传播查询数{total}"
                )

            rows.append({
                "experiment": experiment,
                "corruption": corruption,
                "duration_frames": duration,
                "severity": severity,
                "phase": phase,
                "trace_records": (
                    values["trace_records"]
                ),
                "propagated_queries": total,
                "KEEP": values["keep"],
                "RECOVER": values["recover"],
                "DEFER": values["defer"],
                "keep_ratio": (
                    values["keep"] / total
                    if total else None
                ),
                "recover_ratio": (
                    values["recover"] / total
                    if total else None
                ),
                "defer_ratio": (
                    values["defer"] / total
                    if total else None
                ),
            })


fields = [
    "experiment",
    "corruption",
    "duration_frames",
    "severity",
    "phase",
    "trace_records",
    "propagated_queries",
    "KEEP",
    "RECOVER",
    "DEFER",
    "keep_ratio",
    "recover_ratio",
    "defer_ratio",
]


with OUTPUT.open(
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
    f"{'时长':>6s}"
    f"{'阶段':20s}"
    f"{'数量':>8s}"
    f"{'KEEP':>10s}"
    f"{'RECOVER':>10s}"
    f"{'DEFER':>10s}"
)

print("-" * 78)

for row in rows:
    total = row["propagated_queries"]

    if not total:
        continue

    print(
        f"{row['corruption']:14s}"
        f"{row['duration_frames']:6d}"
        f"{row['phase']:20s}"
        f"{total:8d}"
        f"{row['keep_ratio']:10.4f}"
        f"{row['recover_ratio']:10.4f}"
        f"{row['defer_ratio']:10.4f}"
    )

print()
print("实验数：", len(rows))
print("已生成：", OUTPUT)
