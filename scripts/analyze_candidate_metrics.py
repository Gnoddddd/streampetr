#!/usr/bin/env python3

import csv
import re
from pathlib import Path


ROOT = Path.home() / "research/evidence3d"

SUMMARY = (
    ROOT
    / "outputs/protocol_evaluations/full_candidate"
    / "candidate_protocol_summary.csv"
)

CLEAN_LOG = (
    ROOT
    / "outputs/protocol_evaluations"
    / "source_aware_ft400_clean.log"
)

OUTPUT = (
    ROOT
    / "outputs/protocol_evaluations/full_candidate"
    / "candidate_protocol_deltas.csv"
)


def extract(pattern, text):
    match = re.search(
        pattern,
        text,
        flags=re.MULTILINE,
    )

    if match is None:
        raise RuntimeError(
            f"没有找到指标：{pattern}"
        )

    return float(match.group(1))


if not SUMMARY.is_file():
    raise FileNotFoundError(SUMMARY)

if not CLEAN_LOG.is_file():
    raise FileNotFoundError(CLEAN_LOG)


clean_text = CLEAN_LOG.read_text(
    encoding="utf-8",
    errors="ignore",
)

clean_map = extract(
    r"^mAP:\s*([0-9.]+)",
    clean_text,
)

clean_nds = extract(
    r"^NDS:\s*([0-9.]+)",
    clean_text,
)

clean_car_ap = extract(
    r"^car\s+([0-9.]+)",
    clean_text,
)


with SUMMARY.open(
    "r",
    encoding="utf-8-sig",
) as file:
    source_rows = list(csv.DictReader(file))


rows = []

for row in source_rows:
    if not row.get("mAP") or not row.get("NDS"):
        continue

    experiment = row["experiment"]

    map_value = float(row["mAP"])
    nds_value = float(row["NDS"])

    car_ap = (
        float(row["car_AP"])
        if row.get("car_AP")
        else None
    )

    duration_match = re.search(
        r"(?:back_|after_)(\d+)f",
        experiment,
    )

    duration = (
        int(duration_match.group(1))
        if duration_match
        else None
    )

    if "camera_crash" in experiment:
        family = "camera_crash"
    elif "frame_lost" in experiment:
        family = "frame_lost"
    elif "compound" in experiment:
        family = "compound"
    elif "multi_camera" in experiment:
        family = "multi_camera"
    elif "recovery" in experiment:
        family = "recovery"
    else:
        family = "other"

    delta_map = map_value - clean_map
    delta_nds = nds_value - clean_nds

    delta_car_ap = (
        car_ap - clean_car_ap
        if car_ap is not None
        else None
    )

    relative_nds_drop = (
        (clean_nds - nds_value)
        / clean_nds
        * 100.0
        if clean_nds != 0
        else None
    )

    rows.append({
        "experiment": experiment,
        "family": family,
        "duration_frames": duration,
        "mAP": map_value,
        "NDS": nds_value,
        "car_AP": car_ap,
        "delta_mAP": delta_map,
        "delta_NDS": delta_nds,
        "delta_car_AP": delta_car_ap,
        "relative_NDS_drop_percent": relative_nds_drop,
        "keep_ratio": row.get("keep_ratio"),
        "recover_ratio": row.get("recover_ratio"),
        "defer_ratio": row.get("defer_ratio"),
    })


rows.sort(
    key=lambda item: item["delta_NDS"]
)


fields = [
    "experiment",
    "family",
    "duration_frames",
    "mAP",
    "NDS",
    "car_AP",
    "delta_mAP",
    "delta_NDS",
    "delta_car_AP",
    "relative_NDS_drop_percent",
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


print("Clean基准")
print("mAP =", clean_map)
print("NDS =", clean_nds)
print("Car AP =", clean_car_ap)

print()
print(
    f"{'实验':45s}"
    f"{'mAP':>8s}"
    f"{'NDS':>8s}"
    f"{'ΔNDS':>10s}"
    f"{'下降%':>10s}"
)

print("-" * 81)

for row in rows:
    drop = row["relative_NDS_drop_percent"]

    print(
        f"{row['experiment'][:45]:45s}"
        f"{row['mAP']:8.4f}"
        f"{row['NDS']:8.4f}"
        f"{row['delta_NDS']:10.4f}"
        f"{drop:10.2f}"
    )

print()
print("已生成：", OUTPUT)
