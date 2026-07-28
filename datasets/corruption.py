"""Reproducible partial-observation corruptions for StreamPETR pipelines."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - optional in core unit tests
    cv2 = None

try:
    from mmdet.datasets.builder import PIPELINES
except Exception:  # pragma: no cover - legacy OpenMMLab not installed
    PIPELINES = None

from protocols.partial_observation import ProtocolSchedule

CAMERA_NAMES: Tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def _stable_seed(base_seed: int, sample_token: str, frame_idx: int) -> int:
    text = f"{base_seed}:{sample_token}:{frame_idx}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


def _copy_images(images: Sequence[np.ndarray]) -> List[np.ndarray]:
    return [np.array(image, copy=True) for image in images]


def apply_camera_crash(
    images: Sequence[np.ndarray], failed_indices: Iterable[int], fill_value: float = 0.0
) -> List[np.ndarray]:
    output = _copy_images(images)
    for index in failed_indices:
        output[int(index)][...] = fill_value
    return output


def apply_dark(image: np.ndarray, severity: float) -> np.ndarray:
    severity = float(np.clip(severity, 0.0, 1.0))
    factor = 1.0 - 0.85 * severity
    dark = image.astype(np.float32) * factor
    return np.clip(dark, 0, 255).astype(image.dtype)


def apply_fog(image: np.ndarray, severity: float) -> np.ndarray:
    severity = float(np.clip(severity, 0.0, 1.0))
    alpha = 0.65 * severity
    fog = image.astype(np.float32) * (1.0 - alpha) + 255.0 * alpha
    return np.clip(fog, 0, 255).astype(image.dtype)


def apply_motion_blur(image: np.ndarray, severity: float) -> np.ndarray:
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity <= 0.0:
        return np.array(image, copy=True)
    kernel_size = int(3 + 12 * severity)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if cv2 is None:
        # Dependency-light horizontal moving average fallback.
        pad = kernel_size // 2
        work = np.pad(image.astype(np.float32), ((0, 0), (pad, pad), (0, 0)), mode="edge")
        cumulative = np.cumsum(work, axis=1, dtype=np.float32)
        cumulative = np.pad(
            cumulative, ((0, 0), (1, 0), (0, 0)), mode="constant"
        )
        blurred = (
            cumulative[:, kernel_size:] - cumulative[:, :-kernel_size]
        ) / kernel_size
        return np.clip(blurred, 0, 255).astype(image.dtype)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(image, -1, kernel)


def _register_pipeline(cls):
    if PIPELINES is not None:
        return PIPELINES.register_module()(cls)
    return cls


@_register_pipeline
class ApplyPartialObservation:
    """Apply random training corruptions or a deterministic protocol schedule.

    The transform runs after ``LoadMultiViewImageFromFiles`` and emits three
    tensors consumed by the observability/evidence modules:

    - ``camera_online_mask``: physical camera availability;
    - ``camera_quality``: usable current-image quality in [0,1];
    - ``camera_fresh_mask``: whether the image is a genuinely new observation.

    Frame loss uses a black placeholder for worker-safe training. The explicit
    ``fresh`` state prevents that placeholder or a held frame from becoming new
    temporal evidence.
    """

    def __init__(
        self,
        camera_names: Sequence[str] = CAMERA_NAMES,
        training: bool = True,
        schedule_file: Optional[str] = None,
        seed: int = 0,
        camera_crash_prob: float = 0.25,
        max_failed_cameras: int = 2,
        frame_lost_prob: float = 0.10,
        dark_prob: float = 0.10,
        fog_prob: float = 0.08,
        motion_blur_prob: float = 0.08,
        max_severity: float = 0.8,
        visual_ablation_mode: str = "full",
        curriculum_cfg: Optional[Dict] = None,
    ) -> None:
        self.camera_names = tuple(camera_names)
        self.training = bool(training)
        self.seed = int(seed)
        self.camera_crash_prob = float(camera_crash_prob)
        self.max_failed_cameras = max(int(max_failed_cameras), 1)
        self.frame_lost_prob = float(frame_lost_prob)
        self.dark_prob = float(dark_prob)
        self.fog_prob = float(fog_prob)
        self.motion_blur_prob = float(motion_blur_prob)
        self.max_severity = float(max_severity)
        self.curriculum_cfg: Optional[Dict] = None
        if curriculum_cfg is not None:
            self.configure_curriculum(curriculum_cfg)

        requested_mode = os.environ.get(
            "EVIDENCE3D_VISUAL_ABLATION_MODE",
            visual_ablation_mode,
        )
        requested_mode = (
            str(requested_mode)
            .strip()
            .lower()
            .replace("-", "_")
        )

        allowed_modes = {
            "full",
            "pixel_only",
            "quality_only",
        }

        if requested_mode not in allowed_modes:
            raise ValueError(
                "visual_ablation_mode must be one of "
                f"{sorted(allowed_modes)}, got "
                f"{requested_mode!r}"
            )

        self.visual_ablation_mode = requested_mode
        self.schedule_file = schedule_file
        if schedule_file and not Path(schedule_file).is_file():
            raise FileNotFoundError(f"Protocol schedule not found: {schedule_file}")
        self.schedule = (
            ProtocolSchedule.from_json(schedule_file) if schedule_file else None
        )

    def _random_state(self, rng: np.random.Generator) -> Dict:
        state = {
            "failed_cameras": [],
            "lost_cameras": [],
            "dark": {},
            "fog": {},
            "motion_blur": {},
        }
        camera_count = len(self.camera_names)
        if rng.random() < self.camera_crash_prob:
            count = int(rng.integers(1, min(self.max_failed_cameras, camera_count) + 1))
            indices = rng.choice(camera_count, size=count, replace=False)
            state["failed_cameras"] = [self.camera_names[int(i)] for i in indices]
        available = [name for name in self.camera_names if name not in state["failed_cameras"]]
        if available and rng.random() < self.frame_lost_prob:
            state["lost_cameras"] = [str(rng.choice(available))]
        for key, probability in (
            ("dark", self.dark_prob),
            ("fog", self.fog_prob),
            ("motion_blur", self.motion_blur_prob),
        ):
            if available and rng.random() < probability:
                name = str(rng.choice(available))
                state[key][name] = float(rng.uniform(0.2, self.max_severity))
        return state

    def configure_curriculum(self, curriculum_cfg: Dict) -> None:
        """Enable a deterministic, worker-safe temporal corruption curriculum."""
        cfg = dict(curriculum_cfg)
        ratios = dict(
            cfg.get(
                "ratios",
                {
                    "clean": 0.45,
                    "crash_or_lost": 0.20,
                    "visual": 0.15,
                    "long_fault": 0.10,
                    "compound": 0.10,
                },
            )
        )
        required = {
            "clean",
            "crash_or_lost",
            "visual",
            "long_fault",
            "compound",
        }
        if set(ratios) != required:
            raise ValueError(
                f"curriculum ratios must contain exactly {sorted(required)}"
            )
        total = sum(float(value) for value in ratios.values())
        if any(float(value) < 0.0 for value in ratios.values()) or abs(total - 1.0) > 1e-6:
            raise ValueError("curriculum ratios must be non-negative and sum to one")
        durations = tuple(
            int(value)
            for value in cfg.get("durations", (1, 3, 5, 10, 20))
        )
        if not durations or any(value <= 0 for value in durations):
            raise ValueError("curriculum durations must be positive")
        self.curriculum_cfg = {
            "ratios": ratios,
            "durations": durations,
            "cycle_frames": max(int(cfg.get("cycle_frames", 40)), 20),
        }

    def _curriculum_state(self, scene_token: str, frame_idx: int) -> Dict:
        empty = {
            "failed_cameras": [],
            "lost_cameras": [],
            "dark": {},
            "fog": {},
            "motion_blur": {},
        }
        cfg = self.curriculum_cfg
        if cfg is None:
            return empty
        cycle_frames = cfg["cycle_frames"]
        categories = tuple(cfg["ratios"])
        probabilities = tuple(cfg["ratios"][name] for name in categories)
        current_cycle = frame_idx // cycle_frames
        for cycle in (current_cycle - 1, current_cycle):
            if cycle < 0:
                continue
            rng = np.random.default_rng(
                _stable_seed(self.seed, scene_token, cycle)
            )
            category = str(rng.choice(categories, p=probabilities))
            if category == "clean":
                continue
            duration_choices = cfg["durations"]
            if category == "long_fault":
                duration_choices = tuple(
                    value for value in duration_choices if value >= 10
                )
            duration = int(rng.choice(duration_choices))
            start = cycle * cycle_frames + int(
                rng.integers(0, max(cycle_frames - duration + 1, 1))
            )
            if not start <= frame_idx < start + duration:
                continue
            state = {key: ({} if isinstance(value, dict) else [])
                     for key, value in empty.items()}
            camera = str(rng.choice(self.camera_names))
            if category in {"crash_or_lost", "long_fault"}:
                key = (
                    "failed_cameras"
                    if rng.random() < 0.5
                    else "lost_cameras"
                )
                state[key] = [camera]
            elif category == "visual":
                key = str(rng.choice(("fog", "dark", "motion_blur")))
                state[key][camera] = float(
                    rng.uniform(0.2, self.max_severity)
                )
            elif category == "compound":
                state["failed_cameras"] = [camera]
                available = [
                    name for name in self.camera_names if name != camera
                ]
                visual_camera = str(rng.choice(available))
                key = str(rng.choice(("fog", "dark", "motion_blur")))
                state[key][visual_camera] = float(
                    rng.uniform(0.2, self.max_severity)
                )
            return state
        return empty

    def __call__(self, results: Dict) -> Dict:
        images = results.get("img")
        if images is None:
            raise KeyError("ApplyPartialObservation requires results['img']")
        if len(images) != len(self.camera_names):
            raise ValueError(
                f"Expected {len(self.camera_names)} images, got {len(images)}"
            )
        sample_token = str(results.get("sample_idx", "unknown"))
        scene_token = str(results.get("scene_token", "*"))
        frame_idx = int(results.get("frame_idx", 0))
        rng = np.random.default_rng(_stable_seed(self.seed, sample_token, frame_idx))

        state: Dict = {
            "failed_cameras": [],
            "lost_cameras": [],
            "dark": {},
            "fog": {},
            "motion_blur": {},
        }
        if self.schedule is not None:
            if os.environ.get("EVIDENCE3D_PROTOCOL_DEBUG") == "1":
                print(
                    "[ProtocolDebug]",
                    "sample_idx=", sample_token,
                    "scene_token=", scene_token,
                    "frame_idx=", frame_idx,
                    "schedule_file=", self.schedule_file,
                    flush=True,
                )

            state = self.schedule.state_for(
                scene_token,
                frame_idx,
                self.camera_names,
            )

            if os.environ.get("EVIDENCE3D_PROTOCOL_DEBUG") == "1":
                print(
                    "[ProtocolState]",
                    state,
                    flush=True,
                )
        elif self.training and os.environ.get(
            "EVIDENCE3D_DISABLE_RANDOM_CORRUPTION", "0"
        ).lower() not in {"1", "true", "yes", "on"}:
            state = (
                self._curriculum_state(scene_token, frame_idx)
                if self.curriculum_cfg is not None
                else self._random_state(rng)
            )

        output = _copy_images(images)
        online = np.ones(len(self.camera_names), dtype=np.float32)
        quality = np.ones(len(self.camera_names), dtype=np.float32)
        fresh = np.ones(len(self.camera_names), dtype=np.float32)
        severity = np.zeros(len(self.camera_names), dtype=np.float32)
        name_to_index = {name: index for index, name in enumerate(self.camera_names)}

        apply_visual_pixels = (
            self.visual_ablation_mode
            in {"full", "pixel_only"}
        )
        apply_quality_prior = (
            self.visual_ablation_mode
            in {"full", "quality_only"}
        )

        for name in state.get("failed_cameras", []):
            if name not in name_to_index:
                continue
            index = name_to_index[name]
            output[index][...] = 0
            online[index] = 0.0
            quality[index] = 0.0
            fresh[index] = 0.0
            severity[index] = 1.0
        for name in state.get("lost_cameras", []):
            if name not in name_to_index:
                continue
            index = name_to_index[name]
            output[index][...] = 0
            # Camera may be online, but no current observation arrived.
            quality[index] = 0.0
            fresh[index] = 0.0
            severity[index] = 1.0

        for key, transform in (
            ("dark", apply_dark),
            ("fog", apply_fog),
            ("motion_blur", apply_motion_blur),
        ):
            for name, value in state.get(key, {}).items():
                if name not in name_to_index:
                    continue
                index = name_to_index[name]
                if quality[index] <= 0.0:
                    continue
                value = float(np.clip(value, 0.0, 1.0))

                if apply_visual_pixels:
                    output[index] = transform(
                        output[index],
                        value,
                    )

                if apply_quality_prior:
                    quality[index] *= max(
                        0.05,
                        1.0 - value,
                    )

                # severity仅作为退化事实和诊断记录。
                # 三种消融模式都保留实际协议严重程度。
                severity[index] = max(
                    severity[index],
                    value,
                )

        results["img"] = output
        results["camera_online_mask"] = online
        results["camera_quality"] = quality
        results["camera_fresh_mask"] = fresh
        results["corruption_severity"] = severity
        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(training={self.training}, "
            f"schedule_file={self.schedule_file!r}, "
            f"seed={self.seed}, "
            f"visual_ablation_mode="
            f"{self.visual_ablation_mode!r})"
        )
