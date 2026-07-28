#!/usr/bin/env python3

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path.home() / "research/evidence3d"

TRACE_ROOT = (
    ROOT
    / "outputs"
    / "stage1_r50_evidence_trace_v1"
)

OUTPUT = TRACE_ROOT / "frame_action_summary.csv"


def flatten(value):
    if isinstance(value, list):
        result = []

        for item in value:
            result.extend(flatten(item))

        return result

    return [value]


rows = []

for experiment_dir in sorted(TRACE_ROOT.iterdir()):
    if not experiment_dir.is_dir():
        continue

    if not experiment_dir.name.startswith("trace_v1_"):
        continue

    frame_counts = {}

    trace_dir = experiment_dir / "evidence_trace"

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

                frame_idx = int(record["frame_idx"])

                key = (
                    frame_idx,
                    str(record.get("scene_token", "")),
                )

                values = frame_counts.setdefault(
                    key,
                    Counter(),
                )

                values["records"] += 1

                prior = flatten(
                    diagnostics["prior_strength"]
                )

                actions = flatten(
                    diagnostics["action"]
                )

                if len(prior) != len(actions):
                    raise RuntimeError(
                        f"{experiment_dir.name} "
                        f"frame={frame_idx}: "
                        f"prior={len(prior)}, "
                        f"action={len(actions)}"
                    )

                for prior_value, action_value in zip(
                    prior,
                    actions,
                ):
                    if float(prior_value) <= 1e-6:
                        continue

                    values["propagated"] += 1
                    action = int(action_value)

                    if action == 0:
                        values["KEEP"] += 1
                    elif action == 1:
                        values["RECOVER"] += 1
                    elif action == 2:
                        values["DEFER"] += 1
                    else:
                        values["UNKNOWN"] += 1

    for (frame_idx, scene_token), values in sorted(
        frame_counts.items()
    ):
        total = values["propagated"]

        if values["UNKNOWN"]:
            raise RuntimeError(
                f"{experiment_dir.name} "
                f"frame={frame_idx}: "
                f"未知动作={values['UNKNOWN']}"
            )

        rows.append({
            "experiment": experiment_dir.name,
            "scene_token": scene_token,
            "frame_idx": frame_idx,
            "propagated_queries": total,
            "KEEP": values["KEEP"],
            "RECOVER": values["RECOVER"],
            "DEFER": values["DEFER"],
            "keep_ratio": (
                values["KEEP"] / total
                if total else ""
            ),
            "recover_ratio": (
                values["RECOVER"] / total
                if total else ""
            ),
            "defer_ratio": (
                values["DEFER"] / total
                if total else ""
            ),
        })


fields = [
    "experiment",
    "scene_token",
    "frame_idx",
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

print("[完成]", OUTPUT)
print("记录数：", len(rows))
