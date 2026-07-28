"""Camera Crash protocol helpers."""

from __future__ import annotations

from typing import Iterable, List

from .partial_observation import ProtocolEvent


def persistent_camera_crash(
    cameras: Iterable[str], start_frame: int, duration: int
) -> ProtocolEvent:
    if duration <= 0:
        raise ValueError("duration must be positive")
    return ProtocolEvent(
        start_frame=int(start_frame),
        end_frame=int(start_frame + duration - 1),
        failed_cameras=list(cameras),
    )
