#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path.home() / "research/evidence3d"
OUTPUT_DIR = ROOT / "protocols/presets"

CAMERAS = (
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)

CORRUPTIONS = (
    "dark",
    "fog",
    "motion_blur",
)

START_FRAME = 3
END_FRAME = 12
SEVERITY = 0.9


def make_event(corruption):
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
        camera: SEVERITY
        for camera in CAMERAS
    }

    return event


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

for corruption in CORRUPTIONS:
    payload = {
        "version": 1,
        "scenes": {
            "*": [
                make_event(corruption)
            ],
        },
    }

    output = (
        OUTPUT_DIR
        / f"{corruption}_rear3_10f_s09.json"
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

    print("[已生成]", output)
    print("  类型：", corruption)
    print("  相机：", list(CAMERAS))
    print("  帧：", START_FRAME, "到", END_FRAME)
    print("  严重程度：", SEVERITY)
