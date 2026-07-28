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

OUTPUT_CSV = (
    EVAL_ROOT
    / "candidate_protocol_summary.csv"
)


def flatten(value):
    if isinstance(value, list):
        result = []

        for item in value:
            result.extend(flatten(item))

        return result

    return [value]


def extract_float(pattern, text):
    match = re.search(
        pattern,
        text,
        flags=re.MULTILINE,
    )

    if match is None:
        return None

    return float(match.group(1))


rows = []

for log_path in sorted(
    EVAL_ROOT.glob("*/evaluation.log")
):
    experiment_dir = log_path.parent
    experiment = experiment_dir.name

    text = log_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    map_value = extract_float(
        r"^mAP:\s*([0-9.]+)",
        text,
    )
    nds_value = extract_float(
        r"^NDS:\s*([0-9.]+)",
        text,
    )
    car_ap = extract_float(
        r"^car\s+([0-9.]+)",
        text,
    )

    propagated = 0
    keep = 0
    recover = 0
    defer = 0
    trace_records = 0

    trace_files = sorted(
        (experiment_dir / "traces").glob("*.jsonl")
    )

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
                    continue

                prior = flatten(
                    diagnostics["prior_strength"]
                )
                action = flatten(
                    diagnostics["action"]
                )

                size = min(len(prior), len(action))
                trace_records += 1

                for index in range(size):
                    if float(prior[index]) <= 1e-6:
                        continue

                    propagated += 1
                    action_value = int(action[index])

                    if action_value == 0:
                        keep += 1
                    elif action_value == 1:
                        recover += 1
                    elif action_value == 2:
                        defer += 1

    action_total = keep + recover + defer

    if action_total != propagated:
        status = "action_count_mismatch"
    elif map_value is None or nds_value is None:
        status = "metrics_missing"
    else:
        status = "ok"

    rows.append({
        "experiment": experiment,
        "mAP": map_value,
        "NDS": nds_value,
        "car_AP": car_ap,
        "trace_files": len(trace_files),
        "trace_records": trace_records,
        "propagated_queries": propagated,
        "KEEP": keep,
        "RECOVER": recover,
        "DEFER": defer,
        "keep_ratio": (
            keep / propagated
            if propagated else None
        ),
        "recover_ratio": (
            recover / propagated
            if propagated else None
        ),
        "defer_ratio": (
            defer / propagated
            if propagated else None
        ),
        "status": status,
    })


fields = [
    "experiment",
    "mAP",
    "NDS",
    "car_AP",
    "trace_files",
    "trace_records",
    "propagated_queries",
    "KEEP",
    "RECOVER",
    "DEFER",
    "keep_ratio",
    "recover_ratio",
    "defer_ratio",
    "status",
]

with OUTPUT_CSV.open(
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


def show_number(value, digits=4):
    if value is None:
        return "-"

    return f"{value:.{digits}f}"


print(
    f"{'experiment':42s} "
    f"{'mAP':>7s} "
    f"{'NDS':>7s} "
    f"{'CarAP':>7s} "
    f"{'KEEP':>8s} "
    f"{'RECOVER':>8s} "
    f"{'DEFER':>8s} "
    f"{'status':>12s}"
)

print("-" * 112)

for row in rows:
    print(
        f"{row['experiment'][:42]:42s} "
        f"{show_number(row['mAP']):>7s} "
        f"{show_number(row['NDS']):>7s} "
        f"{show_number(row['car_AP']):>7s} "
        f"{show_number(row['keep_ratio']):>8s} "
        f"{show_number(row['recover_ratio']):>8s} "
        f"{show_number(row['defer_ratio']):>8s} "
        f"{row['status']:>12s}"
    )

print()
print("实验数：", len(rows))
print("汇总文件：", OUTPUT_CSV)
