#!/usr/bin/env python3

import csv
import json
import re
from pathlib import Path


ROOT = Path.home() / "research/evidence3d"
EVAL_ROOT = (
    ROOT
    / "outputs/protocol_evaluations"
    / "full_candidate"
)
OUTPUT = EVAL_ROOT / "phase_action_summary.csv"

FAULT_START = 3


def flatten(value):
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(flatten(item))
        return result
    return [value]


def infer_duration(name):
    patterns = [
        r"recovery_back_after_(\d+)f",
        r"_back_(\d+)f",
        r"_(\d+)f",
    ]

    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))

    return None


def infer_phase(name, frame, duration):
    if frame < FAULT_START:
        return "pre_fault"

    if "persistent" in name:
        return "fault"

    if duration is None:
        return "unspecified"

    if frame < FAULT_START + duration:
        return "fault"

    return "post_recovery"


rows = []

for experiment_dir in sorted(EVAL_ROOT.iterdir()):
    if not experiment_dir.is_dir():
        continue

    experiment = experiment_dir.name
    duration = infer_duration(experiment)
    counts = {}

    trace_dir = experiment_dir / "traces"

    for trace_path in sorted(trace_dir.glob("*.jsonl")):
        with trace_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line)
                diagnostics = record.get(
                    "diagnostics",
                    {},
                )

                if (
                    "prior_strength" not in diagnostics
                    or "action" not in diagnostics
                ):
                    continue

                frame = int(record["frame_idx"])
                phase = infer_phase(
                    experiment,
                    frame,
                    duration,
                )

                counts.setdefault(
                    phase,
                    {
                        "records": 0,
                        "propagated": 0,
                        "keep": 0,
                        "recover": 0,
                        "defer": 0,
                    },
                )

                counts[phase]["records"] += 1

                prior = flatten(
                    diagnostics["prior_strength"]
                )
                action = flatten(
                    diagnostics["action"]
                )

                size = min(len(prior), len(action))

                for index in range(size):
                    if float(prior[index]) <= 1e-6:
                        continue

                    counts[phase]["propagated"] += 1

                    action_value = int(action[index])

                    if action_value == 0:
                        counts[phase]["keep"] += 1
                    elif action_value == 1:
                        counts[phase]["recover"] += 1
                    elif action_value == 2:
                        counts[phase]["defer"] += 1

    for phase, values in counts.items():
        total = values["propagated"]
        action_total = (
            values["keep"]
            + values["recover"]
            + values["defer"]
        )

        if action_total != total:
            raise RuntimeError(
                f"{experiment} {phase}: "
                f"动作数{action_total} != 传播查询数{total}"
            )

        rows.append({
            "experiment": experiment,
            "duration_frames": duration,
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
    "duration_frames",
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
    f"{'实验':39s}"
    f"{'阶段':15s}"
    f"{'数量':>8s}"
    f"{'KEEP':>10s}"
    f"{'RECOVER':>10s}"
    f"{'DEFER':>10s}"
)

print("-" * 94)

for row in rows:
    total = row["propagated_queries"]

    if total == 0:
        continue

    print(
        f"{row['experiment'][:39]:39s}"
        f"{row['phase']:15s}"
        f"{total:8d}"
        f"{row['keep_ratio']:10.4f}"
        f"{row['recover_ratio']:10.4f}"
        f"{row['defer_ratio']:10.4f}"
    )

print()
print("已生成：", OUTPUT)
