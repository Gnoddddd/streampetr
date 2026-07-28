#!/usr/bin/env python3

import csv
from pathlib import Path


SOURCE = Path(
    "outputs/stage1_r50_evidence_trace_v1/"
    "frame_action_summary.csv"
)

OUTPUT = Path(
    "outputs/stage1_r50_evidence_trace_v1/"
    "compound_keyframe_summary.csv"
)

TARGET_EXPERIMENT = (
    "trace_v1_compound_fog_crash_10f"
)

KEY_FRAMES = {
    1: "normal",
    2: "pre_fault",
    3: "fault_onset",
    4: "fault_early",
    6: "fault_middle",
    9: "fault_middle",
    11: "fault_late",
    12: "fault_end",
    13: "first_recovery",
    14: "second_recovery",
}

with SOURCE.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as file:
    rows = list(csv.DictReader(file))

output_rows = []

for row in rows:
    if row["experiment"] != TARGET_EXPERIMENT:
        continue

    frame = int(row["frame_idx"])

    if frame not in KEY_FRAMES:
        continue

    total = int(
        row.get("propagated_queries") or 0
    )

    if total == 0:
        continue

    output_rows.append({
        "experiment": row["experiment"],
        "scene_token": row["scene_token"],
        "frame_idx": frame,
        "phase_label": KEY_FRAMES[frame],
        "propagated_queries": total,
        "KEEP": int(row["KEEP"]),
        "RECOVER": int(row["RECOVER"]),
        "DEFER": int(row["DEFER"]),
        "keep_ratio": float(row["keep_ratio"]),
        "recover_ratio": float(row["recover_ratio"]),
        "defer_ratio": float(row["defer_ratio"]),
        "non_keep_ratio": (
            float(row["recover_ratio"])
            + float(row["defer_ratio"])
        ),
    })

output_rows.sort(
    key=lambda row: (
        row["scene_token"],
        row["frame_idx"],
    )
)

fields = [
    "experiment",
    "scene_token",
    "frame_idx",
    "phase_label",
    "propagated_queries",
    "KEEP",
    "RECOVER",
    "DEFER",
    "keep_ratio",
    "recover_ratio",
    "defer_ratio",
    "non_keep_ratio",
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
    writer.writerows(output_rows)

print("[完成]", OUTPUT)
print("记录数：", len(output_rows))
