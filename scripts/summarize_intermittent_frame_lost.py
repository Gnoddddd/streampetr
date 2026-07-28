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
    / "intermittent_frame_lost_summary.csv"
)


def flatten(value):
    if isinstance(value, list):
        result = []

        for item in value:
            result.extend(flatten(item))

        return result

    return [value]


rows = []

for count in (1, 3, 5, 10, 20):
    experiment = (
        f"presets__frame_lost_back_{count}f"
    )

    experiment_dir = (
        EVAL_ROOT / experiment
    )

    protocol_path = (
        experiment_dir
        / "protocol_used.json"
    )

    if not protocol_path.is_file():
        print(
            "[跳过] 缺少协议副本：",
            protocol_path,
        )
        continue

    payload = json.loads(
        protocol_path.read_text(
            encoding="utf-8"
        )
    )

    events = payload["scenes"]["*"]

    event_frames = {
        int(event["start_frame"])
        for event in events
        if "CAM_BACK" in event.get(
            "lost_cameras",
            [],
        )
    }

    first_event = min(event_frames)
    last_event = max(event_frames)

    counts = defaultdict(
        lambda: {
            "records": 0,
            "propagated": 0,
            "keep": 0,
            "recover": 0,
            "defer": 0,
        }
    )

    trace_dir = (
        experiment_dir / "traces"
    )

    for trace_path in sorted(
        trace_dir.glob("*.jsonl")
    ):
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

                if frame < first_event:
                    phase = "pre_fault"
                elif frame in event_frames:
                    phase = "active_loss"
                elif frame > last_event:
                    phase = "final_recovery"
                else:
                    phase = "normal_gap"

                diagnostics = record.get(
                    "diagnostics",
                    {},
                )

                prior = flatten(
                    diagnostics["prior_strength"]
                )

                action = flatten(
                    diagnostics["action"]
                )

                size = min(
                    len(prior),
                    len(action),
                )

                counts[phase]["records"] += 1

                for index in range(size):
                    if float(prior[index]) <= 1e-6:
                        continue

                    counts[phase][
                        "propagated"
                    ] += 1

                    value = int(
                        action[index]
                    )

                    if value == 0:
                        counts[phase]["keep"] += 1
                    elif value == 1:
                        counts[phase]["recover"] += 1
                    elif value == 2:
                        counts[phase]["defer"] += 1

    for phase in (
        "pre_fault",
        "active_loss",
        "normal_gap",
        "final_recovery",
    ):
        values = counts[phase]
        total = values["propagated"]

        rows.append({
            "experiment": experiment,
            "event_count": count,
            "event_frames": " ".join(
                str(frame)
                for frame in sorted(
                    event_frames
                )
            ),
            "phase": phase,
            "trace_records": values["records"],
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
    "event_count",
    "event_frames",
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
    f"{'实验':37s}"
    f"{'阶段':17s}"
    f"{'数量':>8s}"
    f"{'KEEP':>10s}"
    f"{'RECOVER':>10s}"
    f"{'DEFER':>10s}"
)

print("-" * 92)

for row in rows:
    total = row["propagated_queries"]

    if not total:
        continue

    print(
        f"{row['experiment'][:37]:37s}"
        f"{row['phase']:17s}"
        f"{total:8d}"
        f"{row['keep_ratio']:10.4f}"
        f"{row['recover_ratio']:10.4f}"
        f"{row['defer_ratio']:10.4f}"
    )

print()
print("已生成：", OUTPUT)
