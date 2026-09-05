#!/usr/bin/env python3
"""Generate the preregistered view-deficit schedules with seed 314159."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import mmcv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "protocols/counterfactual_view_deficit"
REPORT = ROOT / "reports/stage3/counterfactual_view_deficit_audit"
SEED = 314159
CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)


def scene_frames(split: str) -> dict[str, list[int]]:
    path = (
        ROOT / "data/nuscenes-mini"
        / f"nuscenes2d_temporal_infos_{split}.pkl"
    )
    grouped: dict[str, list[int]] = {}
    for info in mmcv.load(path)["infos"]:
        grouped.setdefault(str(info["scene_token"]), []).append(int(info["frame_idx"]))
    return {scene: sorted(frames) for scene, frames in grouped.items()}


def event(start: int, duration: int, cameras: list[str]) -> dict:
    return {
        "start_frame": start,
        "end_frame": start + duration - 1,
        "failed_cameras": cameras,
        "lost_cameras": [],
        "dark": {},
        "fog": {},
        "motion_blur": {},
    }


def train_or_seen(split: str, rng: np.random.Generator) -> dict:
    scenes = {}
    for scene, frames in sorted(scene_frames(split).items()):
        maximum = max(frames)
        start = 2
        events = []
        while start <= maximum:
            duration = int(rng.choice((1, 3, 5)))
            camera_index = int(rng.integers(0, len(CAMERAS)))
            if bool(rng.integers(0, 2)):
                cameras = [CAMERAS[camera_index]]
            else:
                cameras = [
                    CAMERAS[camera_index],
                    CAMERAS[(camera_index + 1) % len(CAMERAS)],
                ]
            if start + duration - 1 <= maximum:
                events.append(event(start, duration, cameras))
            start += duration + 2
        scenes[scene] = events
    return {"version": 1, "scenes": scenes}


def fixed_val(cameras: list[str], duration: int) -> dict:
    return {
        "version": 1,
        "scenes": {
            scene: [event(3, duration, cameras)]
            for scene in sorted(scene_frames("val"))
        },
    }


def write_schedule(name: str, payload: dict, family: str, seen: bool,
                   rows: list[dict]) -> None:
    path = OUTPUT / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    frames = 0
    events = 0
    camera_counts = []
    durations = []
    for values in payload["scenes"].values():
        for value in values:
            duration = int(value["end_frame"]) - int(value["start_frame"]) + 1
            frames += duration
            events += 1
            durations.append(duration)
            camera_counts.append(len(value["failed_cameras"]))
    rows.append({
        "name": name,
        "split": "train" if name == "train_seen" else "val",
        "family": family,
        "seen_during_residual_fit": seen,
        "seed": SEED,
        "scenes": len(payload["scenes"]),
        "events": events,
        "corrupted_scene_frames": frames,
        "durations": ";".join(map(str, sorted(set(durations)))),
        "camera_counts": ";".join(map(str, sorted(set(camera_counts)))),
        "schedule_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "schedule": str(path.relative_to(ROOT)),
    })


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    rng = np.random.default_rng(SEED)
    write_schedule(
        "train_seen", train_or_seen("train", rng),
        "single_or_adjacent_double_duration_1_3_5", True, rows,
    )
    write_schedule(
        "val_seen", train_or_seen("val", rng),
        "single_or_adjacent_double_duration_1_3_5", True, rows,
    )
    write_schedule(
        "val_nonadjacent_double",
        fixed_val(["CAM_FRONT", "CAM_BACK"], 5),
        "nonadjacent_double", False, rows,
    )
    write_schedule(
        "val_three_camera",
        fixed_val(["CAM_FRONT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"], 5),
        "three_camera", False, rows,
    )
    write_schedule(
        "val_duration_10",
        fixed_val(["CAM_BACK"], 10),
        "long_duration_10", False, rows,
    )
    write_schedule(
        "val_duration_20",
        fixed_val(["CAM_BACK"], 20),
        "long_duration_20", False, rows,
    )
    write_schedule(
        "val_natural_recovery",
        fixed_val(["CAM_BACK"], 5),
        "natural_recovery_after_5", False, rows,
    )
    with (REPORT / "corruption_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
