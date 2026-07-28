#!/usr/bin/env python3

import csv
import re
from pathlib import Path


MODELS = {
    "B0_official": Path(
        "outputs/gt_recovery_predictions/"
        "official_r50_900q_baseline"
    ),
    "B1_classification": Path(
        "outputs/gt_recovery_predictions/"
        "classification_active_r50_900q"
    ),
    "T1_ternary": Path(
        "outputs/gt_recovery_predictions/"
        "stage1_active_r50_900q"
    ),
}

EXPERIMENTS = {
    "clean": "fixed_v2_clean_no_corruption",
    "camera_crash_5f": "fixed_v2_camera_crash_back_5f",
    "camera_crash_10f": "fixed_v2_camera_crash_back_10f",
    "compound_fog_crash_10f": (
        "fixed_v2_compound_fog_crash_10f"
    ),
}

METRICS = [
    "mAP",
    "mATE",
    "mASE",
    "mAOE",
    "mAVE",
    "mAAE",
    "NDS",
]


def parse_log(path):
    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    values = {}

    for metric in METRICS:
        matches = re.findall(
            rf"(?m)^{re.escape(metric)}:\s*"
            r"([-+]?(?:\d+\.\d+|\d+))",
            text,
        )

        if not matches:
            raise RuntimeError(
                f"{path}: 未找到{metric}"
            )

        values[metric] = float(matches[-1])

    return values


rows = []

for model, root in MODELS.items():
    metrics_by_protocol = {}

    for protocol, experiment in EXPERIMENTS.items():
        log_path = (
            root
            / experiment
            / "evaluation.log"
        )

        if not log_path.is_file():
            raise FileNotFoundError(log_path)

        metrics_by_protocol[protocol] = parse_log(
            log_path
        )

    clean = metrics_by_protocol["clean"]

    for protocol, metrics in metrics_by_protocol.items():
        map_drop = clean["mAP"] - metrics["mAP"]
        nds_drop = clean["NDS"] - metrics["NDS"]

        map_retention = (
            metrics["mAP"] / clean["mAP"] * 100.0
            if clean["mAP"]
            else 0.0
        )

        nds_retention = (
            metrics["NDS"] / clean["NDS"] * 100.0
            if clean["NDS"]
            else 0.0
        )

        row = {
            "model": model,
            "protocol": protocol,
            **metrics,
            "mAP_drop_from_clean": map_drop,
            "NDS_drop_from_clean": nds_drop,
            "mAP_retention_percent": map_retention,
            "NDS_retention_percent": nds_retention,
        }

        rows.append(row)

output_dir = Path("outputs/fixed_v2_summary")
output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

csv_path = output_dir / "protocol_metrics.csv"

fieldnames = [
    "model",
    "protocol",
    *METRICS,
    "mAP_drop_from_clean",
    "NDS_drop_from_clean",
    "mAP_retention_percent",
    "NDS_retention_percent",
]

with csv_path.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for row in rows:
        writer.writerow(row)

md_path = output_dir / "protocol_metrics.md"

with md_path.open(
    "w",
    encoding="utf-8",
) as file:
    file.write(
        "| 模型 | 协议 | mAP | NDS | "
        "mAP下降 | NDS下降 | "
        "mAP保持率 | NDS保持率 |\n"
    )

    file.write(
        "|---|---|---:|---:|---:|---:|---:|---:|\n"
    )

    for row in rows:
        file.write(
            f"| {row['model']} "
            f"| {row['protocol']} "
            f"| {row['mAP']:.4f} "
            f"| {row['NDS']:.4f} "
            f"| {row['mAP_drop_from_clean']:.4f} "
            f"| {row['NDS_drop_from_clean']:.4f} "
            f"| {row['mAP_retention_percent']:.2f}% "
            f"| {row['NDS_retention_percent']:.2f}% |\n"
        )

print("[完成]", csv_path)
print("[完成]", md_path)

print()
print(md_path.read_text(encoding="utf-8"))
