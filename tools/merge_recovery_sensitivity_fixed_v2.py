#!/usr/bin/env python3

import csv
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

SETTINGS = [
    "w2_t095",
    "w2_t100",
    "w3_t095",
]

rows = []
fieldnames = [
    "model",
    "setting",
]

for model, root in MODELS.items():
    for setting in SETTINGS:
        path = (
            root
            / (
                "gt_recovery_delay_robust_fixed_v2_"
                f"{setting}_summary.csv"
            )
        )

        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                raise RuntimeError(
                    f"{path}缺少CSV表头"
                )

            for name in reader.fieldnames:
                if name not in fieldnames:
                    fieldnames.append(name)

            for row in reader:
                rows.append({
                    "model": model,
                    "setting": setting,
                    **row,
                })

output = Path(
    "outputs/fixed_v2_summary/"
    "recovery_sensitivity_all.csv"
)

with output.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
        extrasaction="ignore",
    )

    writer.writeheader()
    writer.writerows(rows)

print("[完成]", output)
print("记录数：", len(rows))
