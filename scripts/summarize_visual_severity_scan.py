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

METRIC_CSV = (
    EVAL_ROOT
    / "candidate_protocol_summary.csv"
)

OUTPUT_CSV = (
    EVAL_ROOT
    / "visual_severity_scan_summary.csv"
)

CORRUPTIONS = (
    "dark",
    "fog",
    "motion_blur",
)

EXPERIMENTS = {
    0.3: "s03",
    0.6: None,
    0.9: "s09",
}


def flatten(value):
    if isinstance(value, list):
        result = []

        for item in value:
            result.extend(flatten(item))

        return result

    return [value]


with METRIC_CSV.open(
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

for corruption in CORRUPTIONS:
    for severity, tag in EXPERIMENTS.items():
        if tag is None:
            experiment = (
                f"presets__{corruption}_back_10f"
            )
        else:
            experiment = (
                f"presets__{corruption}"
                f"_back_10f_{tag}"
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
                "[跳过] 缺少协议：",
                protocol_path,
            )
            continue

        protocol = json.loads(
            protocol_path.read_text(
                encoding="utf-8"
            )
        )

        event = protocol["scenes"]["*"][0]

        start = int(event["start_frame"])
        end = int(event["end_frame"])

        action_counts = {
            "keep": 0,
            "recover": 0,
            "defer": 0,
        }

        propagated = 0

        for trace_path in sorted(
            (
                experiment_dir / "traces"
            ).glob("*.jsonl")
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

                    if not (
                        start <= frame <= end
                    ):
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
                        if (
                            float(prior_value)
                            <= 1e-6
                        ):
                            continue

                        propagated += 1
                        action = int(action)

                        if action == 0:
                            action_counts["keep"] += 1
                        elif action == 1:
                            action_counts[
                                "recover"
                            ] += 1
                        elif action == 2:
                            action_counts["defer"] += 1

        metric = metrics.get(
            experiment,
            {},
        )

        rows.append({
            "experiment": experiment,
            "corruption": corruption,
            "severity": severity,
            "camera_quality": max(
                0.05,
                1.0 - severity,
            ),
            "mAP": metric.get("mAP"),
            "NDS": metric.get("NDS"),
            "CarAP": metric.get("CarAP"),
            "active_queries": propagated,
            "keep_ratio": (
                action_counts["keep"]
                / propagated
                if propagated else None
            ),
            "recover_ratio": (
                action_counts["recover"]
                / propagated
                if propagated else None
            ),
            "defer_ratio": (
                action_counts["defer"]
                / propagated
                if propagated else None
            ),
        })


fields = [
    "experiment",
    "corruption",
    "severity",
    "camera_quality",
    "mAP",
    "NDS",
    "CarAP",
    "active_queries",
    "keep_ratio",
    "recover_ratio",
    "defer_ratio",
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


print(
    f"{'退化':14s}"
    f"{'强度':>8s}"
    f"{'质量':>8s}"
    f"{'mAP':>9s}"
    f"{'NDS':>9s}"
    f"{'KEEP':>10s}"
    f"{'RECOVER':>10s}"
    f"{'DEFER':>10s}"
)

print("-" * 78)

for row in rows:
    print(
        f"{row['corruption']:14s}"
        f"{row['severity']:8.1f}"
        f"{row['camera_quality']:8.2f}"
        f"{float(row['mAP']):9.4f}"
        f"{float(row['NDS']):9.4f}"
        f"{row['keep_ratio']:10.4f}"
        f"{row['recover_ratio']:10.4f}"
        f"{row['defer_ratio']:10.4f}"
    )

print()
print("已生成：", OUTPUT_CSV)
