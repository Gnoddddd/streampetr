#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path.home() / "research/evidence3d"
OUTPUT = ROOT / "protocols/presets"

CAMERA = "CAM_BACK"
START_FRAME = 3
SEVERITY = 0.6
DURATIONS = (5, 10, 20)
CORRUPTIONS = (
    "dark",
    "fog",
    "motion_blur",
)


def make_event(
    corruption,
    duration,
):
    event = {
        "start_frame": START_FRAME,
        "end_frame": (
            START_FRAME + duration - 1
        ),
        "failed_cameras": [],
        "lost_cameras": [],
        "dark": {},
        "fog": {},
        "motion_blur": {},
    }

    event[corruption] = {
        CAMERA: SEVERITY,
    }

    return event


OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)

for corruption in CORRUPTIONS:
    for duration in DURATIONS:
        payload = {
            "version": 1,
            "scenes": {
                "*": [
                    make_event(
                        corruption,
                        duration,
                    )
                ],
            },
        }

        output = (
            OUTPUT
            / f"{corruption}_back_{duration}f.json"
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
            "[已生成]",
            output,
        )
