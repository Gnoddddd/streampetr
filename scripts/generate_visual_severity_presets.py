#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path.home() / "research/evidence3d"
OUTPUT_DIR = ROOT / "protocols/presets"

CORRUPTIONS = (
    "dark",
    "fog",
    "motion_blur",
)

SEVERITIES = (
    0.3,
    0.9,
)

CAMERA = "CAM_BACK"
START_FRAME = 3
END_FRAME = 12


def severity_tag(value):
    return f"s{int(round(value * 10)):02d}"


for corruption in CORRUPTIONS:
    for severity in SEVERITIES:
        event = {
            "start_frame": START_FRAME,
            "end_frame": END_FRAME,
            "failed_cameras": [],
            "lost_cameras": [],
            "dark": {},
            "fog": {},
            "motion_blur": {},
        }

        event[corruption] = {
            CAMERA: severity,
        }

        payload = {
            "version": 1,
            "scenes": {
                "*": [event],
            },
        }

        tag = severity_tag(severity)

        output = (
            OUTPUT_DIR
            / f"{corruption}_back_10f_{tag}.json"
        )

        output.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            f"[已生成] {output} "
            f"severity={severity}"
        )
