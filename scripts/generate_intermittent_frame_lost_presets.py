#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path.home() / "research/evidence3d"
OUTPUT_DIR = ROOT / "protocols/presets"

# 20个确定性间歇丢帧位置。
# 最大事件帧为35，从而保留第36～40帧作为最终恢复窗口。
MASTER_FRAMES = [
    3,
    5,
    6,
    8,
    10,
    11,
    13,
    15,
    16,
    18,
    20,
    21,
    23,
    25,
    26,
    28,
    30,
    31,
    33,
    35,
]

# 嵌套事件集合：
# 1f ⊂ 3f ⊂ 5f ⊂ 10f ⊂ 20f
EVENT_FRAMES = {
    1: [
        3,
    ],
    3: [
        3,
        20,
        35,
    ],
    5: [
        3,
        11,
        20,
        28,
        35,
    ],
    10: [
        3,
        6,
        11,
        15,
        20,
        23,
        28,
        31,
        33,
        35,
    ],
    20: MASTER_FRAMES,
}


def make_event(frame):
    return {
        "start_frame": frame,
        "end_frame": frame,
        "failed_cameras": [],
        "lost_cameras": [
            "CAM_BACK",
        ],
        "dark": {},
        "fog": {},
        "motion_blur": {},
    }


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

for count, frames in EVENT_FRAMES.items():
    if len(frames) != count:
        raise RuntimeError(
            f"{count}f事件数错误："
            f"实际为{len(frames)}"
        )

    if frames != sorted(set(frames)):
        raise RuntimeError(
            f"{count}f包含重复或无序帧："
            f"{frames}"
        )

    payload = {
        "version": 1,
        "scenes": {
            "*": [
                make_event(frame)
                for frame in frames
            ],
        },
    }

    output = (
        OUTPUT_DIR
        / f"frame_lost_back_{count}f.json"
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
        f"[已生成] {output}"
    )
    print(
        f"         事件帧：{frames}"
    )
