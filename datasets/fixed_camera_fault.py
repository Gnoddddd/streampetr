"""Deterministic single-camera faults for paired train-only audits."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .corruption import (
    CAMERA_NAMES,
    _copy_images,
    _register_pipeline,
    apply_dark,
    apply_motion_blur,
)


@_register_pipeline
class ApplyFixedCameraFault:
    """Apply one fixed fault without consuming global augmentation RNG state."""

    def __init__(
        self,
        mode: str,
        camera: str = "CAM_BACK",
        severity: float = 0.9,
        camera_names: Sequence[str] = CAMERA_NAMES,
    ) -> None:
        aliases = {"blur": "motion_blur", "motionblur": "motion_blur"}
        self.mode = aliases.get(str(mode).lower(), str(mode).lower())
        if self.mode not in {"dark", "motion_blur", "crash"}:
            raise ValueError(f"unsupported fixed camera fault: {mode}")
        self.camera_names = tuple(camera_names)
        if camera not in self.camera_names:
            raise ValueError(f"unknown camera {camera}")
        self.camera = camera
        self.camera_index = self.camera_names.index(camera)
        self.severity = float(np.clip(severity, 0.0, 1.0))

    def __call__(self, results: Dict) -> Dict:
        images = results.get("img")
        if images is None or len(images) != len(self.camera_names):
            raise ValueError("fixed camera fault requires all camera images")
        output = _copy_images(images)
        index = self.camera_index
        online = np.ones(len(self.camera_names), dtype=np.float32)
        quality = np.ones(len(self.camera_names), dtype=np.float32)
        fresh = np.ones(len(self.camera_names), dtype=np.float32)
        severity = np.zeros(len(self.camera_names), dtype=np.float32)
        if self.mode == "dark":
            output[index] = apply_dark(output[index], self.severity)
            quality[index] = max(0.05, 1.0 - self.severity)
            severity[index] = self.severity
        elif self.mode == "motion_blur":
            output[index] = apply_motion_blur(output[index], self.severity)
            quality[index] = max(0.05, 1.0 - self.severity)
            severity[index] = self.severity
        else:
            output[index][...] = 0
            online[index] = quality[index] = fresh[index] = 0.0
            severity[index] = 1.0
        results.update({
            "img": output,
            "camera_online_mask": online,
            "camera_quality": quality,
            "camera_fresh_mask": fresh,
            "corruption_severity": severity,
            "paired_fault_mode": self.mode,
            "paired_fault_camera": self.camera,
        })
        return results
