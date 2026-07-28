#!/usr/bin/env python3

import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path.home() / "research/evidence3d"

EVAL_ROOT = (
    ROOT
    / "outputs"
    / "stage1_r50_evidence_trace_v1"
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
    if "clean_no_corruption" in name:
        return None

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
    if "clean_no_corruption" in name:
        return "clean"

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

if not EVAL_ROOT.is_dir():
    raise FileNotFoundError(EVAL_ROOT)

for experiment_dir in sorted(EVAL_ROOT.iterdir()):
    if not experiment_dir.is_dir():
        continue

    experiment = experiment_dir.name

    if not experiment.startswith("trace_v1_"):
        continue

    duration = infer_duration(experiment)
    counts = {}
    missing_diagnostics = 0

    trace_dir = experiment_dir / "evidence_trace"
    trace_files = sorted(trace_dir.glob("*.jsonl"))

    if not trace_files:
        print("[警告] 没有轨迹：", experiment)
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
                diagnostics = record.get(
                    "diagnostics",
                    {},
                )

                if (
                    "prior_strength" not in diagnostics
                    or "action" not in diagnostics
                ):
                    missing_diagnostics += 1
                    continue

                frame = int(record["frame_idx"])
                phase = infer_phase(
                    experiment,
                    frame,
                    duration,
                )

                counts.setdefault(
                    phase,
                    Counter(),
                )

                counts[phase]["records"] += 1

                prior = flatten(
                    diagnostics["prior_strength"]
                )

                actions = flatten(
                    diagnostics["action"]
                )

                if len(prior) != len(actions):
                    raise RuntimeError(
                        f"{experiment} frame={frame}: "
                        f"prior={len(prior)}, "
                        f"action={len(actions)}"
                    )

                for prior_value, action_value in zip(
                    prior,
                    actions,
                ):
                    if float(prior_value) <= 1e-6:
                        continue

                    counts[phase]["propagated"] += 1

                    action = int(action_value)

                    if action == 0:
                        counts[phase]["KEEP"] += 1
                    elif action == 1:
                        counts[phase]["RECOVER"] += 1
                    elif action == 2:
                        counts[phase]["DEFER"] += 1
                    else:
                        counts[phase]["UNKNOWN"] += 1

    if missing_diagnostics:
        print(
            f"[警告] {experiment}有"
            f"{missing_diagnostics}条记录缺少诊断字段"
        )

    for phase, values in counts.items():
        total = values["propagated"]

        action_total = (
            values["KEEP"]
            + values["RECOVER"]
            + values["DEFER"]
        )

        if values["UNKNOWN"]:
            raise RuntimeError(
                f"{experiment} {phase}: "
                f"未知动作数={values['UNKNOWN']}"
            )

        if action_total != total:
            raise RuntimeError(
                f"{experiment} {phase}: "
                f"动作数{action_total} != "
                f"传播查询数{total}"
            )

        rows.append({
            "experiment": experiment,
            "duration_frames": (
                duration
                if duration is not None
                else ""
            ),
            "phase": phase,
            "trace_records": values["records"],
            "propagated_queries": total,
            "KEEP": values["KEEP"],
            "RECOVER": values["RECOVER"],
            "DEFER": values["DEFER"],
            "keep_ratio": (
                values["KEEP"] / total
                if total
                else ""
            ),
            "recover_ratio": (
                values["RECOVER"] / total
                if total
                else ""
            ),
            "defer_ratio": (
                values["DEFER"] / total
                if total
                else ""
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
    f"{'实验':40s}"
    f"{'阶段':16s}"
    f"{'数量':>9s}"
    f"{'KEEP':>10s}"
    f"{'RECOVER':>10s}"
    f"{'DEFER':>10s}"
)

print("-" * 95)

for row in rows:
    total = row["propagated_queries"]

    if not total:
        continue

    print(
        f"{row['experiment'][:40]:40s}"
        f"{row['phase']:16s}"
        f"{total:9d}"
        f"{row['keep_ratio']:10.4f}"
        f"{row['recover_ratio']:10.4f}"
        f"{row['defer_ratio']:10.4f}"
    )

print()
print("[完成]", OUTPUT)
