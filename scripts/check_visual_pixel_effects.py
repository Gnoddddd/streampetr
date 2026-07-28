#!/usr/bin/env python3

import copy
from pathlib import Path

import numpy as np

from datasets.corruption import (
    ApplyPartialObservation,
    CAMERA_NAMES,
)


ROOT = Path.home() / "research/evidence3d"
CAMERA = "CAM_BACK"
CAMERA_INDEX = list(CAMERA_NAMES).index(CAMERA)

rng = np.random.default_rng(20260726)

base_image = rng.integers(
    0,
    256,
    size=(128, 192, 3),
    dtype=np.uint8,
)

base_images = [
    base_image.copy()
    for _ in CAMERA_NAMES
]

outputs = {}

for corruption in (
    "dark",
    "fog",
    "motion_blur",
):
    protocol = (
        ROOT
        / "protocols/presets"
        / f"{corruption}_back_5f.json"
    )

    transform = ApplyPartialObservation(
        training=False,
        schedule_file=str(protocol),
        camera_crash_prob=0.0,
        frame_lost_prob=0.0,
        dark_prob=0.0,
        fog_prob=0.0,
        motion_blur_prob=0.0,
    )

    results = {
        "img": [
            image.copy()
            for image in base_images
        ],
        "sample_idx": (
            f"pixel_check_{corruption}"
        ),
        "scene_token": "*",
        "frame_idx": 3,
    }

    result = transform(
        copy.deepcopy(results)
    )

    output = np.asarray(
        result["img"][CAMERA_INDEX]
    )

    outputs[corruption] = output

    original = base_image.astype(
        np.float32
    )
    modified = output.astype(
        np.float32
    )

    difference = np.abs(
        modified - original
    )

    print()
    print("=" * 72)
    print("退化：", corruption)
    print("输出dtype：", output.dtype)
    print("原图均值：", float(original.mean()))
    print("退化均值：", float(modified.mean()))
    print(
        "平均绝对像素差：",
        float(difference.mean()),
    )
    print(
        "最大像素差：",
        float(difference.max()),
    )
    print(
        "变化像素比例：",
        float(
            np.mean(difference > 0)
        ),
    )

    for key in (
        "camera_quality",
        "camera_fresh_mask",
        "camera_online_mask",
        "corruption_severity",
    ):
        if key in result:
            value = np.asarray(
                result[key]
            ).reshape(-1)

            print(
                f"{key}[CAM_BACK]：",
                float(value[CAMERA_INDEX]),
            )


print()
print("=" * 72)
print("不同退化输出的两两差异")

names = list(outputs)

for index, first in enumerate(names):
    for second in names[index + 1:]:
        first_image = outputs[
            first
        ].astype(np.float32)

        second_image = outputs[
            second
        ].astype(np.float32)

        difference = np.abs(
            first_image - second_image
        )

        print(
            f"{first} vs {second}:",
            "平均绝对差=",
            float(difference.mean()),
            "完全相同=",
            bool(np.array_equal(
                outputs[first],
                outputs[second],
            )),
        )
