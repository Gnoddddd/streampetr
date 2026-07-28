#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np

from datasets.corruption import (
    ApplyPartialObservation,
    CAMERA_NAMES,
)


ROOT = Path.home() / "research/evidence3d"

PROTOCOL = (
    ROOT
    / "protocols/presets"
    / "fog_back_10f.json"
)

CAMERA_INDEX = list(
    CAMERA_NAMES
).index("CAM_BACK")

rng = np.random.default_rng(20260726)

original = rng.integers(
    0,
    256,
    size=(128, 192, 3),
    dtype=np.uint8,
)

expected = {
    "full": {
        "pixel_changed": True,
        "quality": 0.4,
    },
    "pixel_only": {
        "pixel_changed": True,
        "quality": 1.0,
    },
    "quality_only": {
        "pixel_changed": False,
        "quality": 0.4,
    },
}


try:
    for mode, expectation in expected.items():
        os.environ[
            "EVIDENCE3D_VISUAL_ABLATION_MODE"
        ] = mode

        transform = ApplyPartialObservation(
            training=False,
            schedule_file=str(PROTOCOL),
        )

        results = {
            "img": [
                original.copy()
                for _ in CAMERA_NAMES
            ],
            "sample_idx": (
                f"visual_ablation_{mode}"
            ),
            "scene_token": "*",
            "frame_idx": 3,
        }

        result = transform(results)

        output = np.asarray(
            result["img"][CAMERA_INDEX]
        )

        quality = float(
            np.asarray(
                result["camera_quality"]
            )[CAMERA_INDEX]
        )

        fresh = float(
            np.asarray(
                result["camera_fresh_mask"]
            )[CAMERA_INDEX]
        )

        online = float(
            np.asarray(
                result["camera_online_mask"]
            )[CAMERA_INDEX]
        )

        severity = float(
            np.asarray(
                result["corruption_severity"]
            )[CAMERA_INDEX]
        )

        pixel_changed = not np.array_equal(
            output,
            original,
        )

        mean_difference = float(
            np.abs(
                output.astype(np.float32)
                - original.astype(np.float32)
            ).mean()
        )

        assert (
            pixel_changed
            == expectation["pixel_changed"]
        ), (
            mode,
            pixel_changed,
            expectation,
        )

        assert np.isclose(
            quality,
            expectation["quality"],
            atol=1e-5,
        ), (
            mode,
            quality,
            expectation,
        )

        assert np.isclose(fresh, 1.0)
        assert np.isclose(online, 1.0)
        assert np.isclose(
            severity,
            0.6,
            atol=1e-5,
        )

        print()
        print("=" * 72)
        print("模式：", mode)
        print("repr：", transform)
        print("像素是否变化：", pixel_changed)
        print("平均绝对像素差：", mean_difference)
        print("camera_quality：", quality)
        print("camera_fresh_mask：", fresh)
        print("camera_online_mask：", online)
        print("corruption_severity：", severity)

finally:
    os.environ.pop(
        "EVIDENCE3D_VISUAL_ABLATION_MODE",
        None,
    )


print()
print("三种视觉消融模式全部验证通过")
