#!/usr/bin/env python3
"""Generate deterministic PartialObs-3D protocol presets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from protocols.partial_observation import ProtocolEvent, ProtocolSchedule


def write(path: Path, events):
    ProtocolSchedule({"*": events}).to_json(str(path))
    print(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "protocols" / "presets"),
    )
    parser.add_argument("--start-frame", type=int, default=3)
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = int(args.start_frame)

    for duration in (1, 3, 5, 10, 20):
        write(
            output / f"camera_crash_back_{duration}f.json",
            [ProtocolEvent(start, start + duration - 1, failed_cameras=["CAM_BACK"])],
        )
        write(
            output / f"frame_lost_back_{duration}f.json",
            [ProtocolEvent(start, start + duration - 1, lost_cameras=["CAM_BACK"])],
        )

    write(
        output / "multi_camera_left_10f.json",
        [
            ProtocolEvent(
                start,
                start + 9,
                failed_cameras=["CAM_FRONT_LEFT", "CAM_BACK_LEFT"],
            )
        ],
    )
    write(
        output / "compound_fog_crash_10f.json",
        [
            ProtocolEvent(
                start,
                start + 9,
                failed_cameras=["CAM_BACK"],
                fog={"CAM_FRONT": 0.55, "CAM_FRONT_RIGHT": 0.55},
            )
        ],
    )
    write(
        output / "recovery_back_after_5f.json",
        [ProtocolEvent(start, start + 4, failed_cameras=["CAM_BACK"])],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
