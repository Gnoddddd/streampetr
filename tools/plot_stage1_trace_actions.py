#!/usr/bin/env python3

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


SOURCE = Path(
    "outputs/stage1_r50_evidence_trace_v1/"
    "frame_action_summary.csv"
)

OUTPUT_DIR = Path(
    "outputs/stage1_r50_evidence_trace_v1/"
    "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EXPERIMENT = (
    "trace_v1_compound_fog_crash_10f"
)

with SOURCE.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as file:
    rows = list(csv.DictReader(file))

by_scene = defaultdict(list)

for row in rows:
    if row["experiment"] != EXPERIMENT:
        continue

    if int(row.get("propagated_queries") or 0) <= 0:
        continue

    by_scene[row["scene_token"]].append({
        "frame": int(row["frame_idx"]),
        "keep": float(row["keep_ratio"]),
        "recover": float(row["recover_ratio"]),
        "defer": float(row["defer_ratio"]),
    })

for scene_token, values in sorted(
    by_scene.items()
):
    values.sort(key=lambda item: item["frame"])

    frames = [
        item["frame"]
        for item in values
    ]

    plt.figure(figsize=(10, 5))

    plt.plot(
        frames,
        [item["keep"] for item in values],
        marker="o",
        label="KEEP",
    )

    plt.plot(
        frames,
        [item["recover"] for item in values],
        marker="o",
        label="RECOVER",
    )

    plt.plot(
        frames,
        [item["defer"] for item in values],
        marker="o",
        label="DEFER",
    )

    plt.axvline(
        3,
        linestyle="--",
        label="Fault start",
    )

    plt.axvline(
        13,
        linestyle="--",
        label="Recovery start",
    )

    plt.xlabel("Frame index")
    plt.ylabel("Action ratio")
    plt.ylim(0.0, 1.05)
    plt.title(
        "Compound Fog + Camera Crash\n"
        + scene_token
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / f"compound_actions_{scene_token}.png"
    )

    plt.savefig(
        output,
        dpi=200,
    )

    plt.close()

    print("[完成]", output)
