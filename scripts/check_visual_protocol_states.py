#!/usr/bin/env python3

import ast
import re
from pathlib import Path


ROOT = Path.home() / "research/evidence3d"

EVAL_ROOT = (
    ROOT
    / "outputs/protocol_evaluations"
    / "full_candidate"
)

TYPES = (
    "dark",
    "fog",
    "motion_blur",
)

DURATIONS = (
    5,
    10,
    20,
)


for corruption in TYPES:
    for duration in DURATIONS:
        experiment = (
            f"presets__{corruption}"
            f"_back_{duration}f"
        )

        log_path = (
            EVAL_ROOT
            / experiment
            / "evaluation.log"
        )

        if not log_path.is_file():
            print("[缺失]", log_path)
            continue

        current_scene = None
        current_frame = None
        active = {}

        for line in log_path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines():
            if "[ProtocolDebug]" in line:
                scene_match = re.search(
                    r"scene_token=\s*(\S+)",
                    line,
                )

                frame_match = re.search(
                    r"frame_idx=\s*(\d+)",
                    line,
                )

                if scene_match:
                    current_scene = (
                        scene_match.group(1)
                    )

                if frame_match:
                    current_frame = int(
                        frame_match.group(1)
                    )

            elif "[ProtocolState]" in line:
                raw = line.split(
                    "[ProtocolState]",
                    1,
                )[1].strip()

                try:
                    state = ast.literal_eval(raw)
                except Exception:
                    continue

                values = state.get(
                    corruption,
                    {},
                )

                if "CAM_BACK" in values:
                    active.setdefault(
                        current_scene,
                        [],
                    ).append(
                        (
                            current_frame,
                            values["CAM_BACK"],
                        )
                    )

        print()
        print("=" * 80)
        print(experiment)

        if not active:
            print("未从日志中找到退化状态")
            continue

        for scene, events in sorted(
            active.items()
        ):
            unique = sorted(set(events))

            print("scene：", scene)
            print("生效记录：", unique)
            print("生效帧数：", len(unique))
