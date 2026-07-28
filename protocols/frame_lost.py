"""Frame Lost protocol helpers and offline sequence simulation."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from .partial_observation import ProtocolEvent


def frame_lost_event(
    cameras: Iterable[str], start_frame: int, duration: int
) -> ProtocolEvent:
    if duration <= 0:
        raise ValueError("duration must be positive")
    return ProtocolEvent(
        start_frame=int(start_frame),
        end_frame=int(start_frame + duration - 1),
        lost_cameras=list(cameras),
    )


def hold_last_frame(sequence: Sequence[np.ndarray], lost_indices: Iterable[int]) -> List[np.ndarray]:
    output = [np.array(frame, copy=True) for frame in sequence]
    lost = set(int(index) for index in lost_indices)
    for index in sorted(lost):
        if index <= 0 or index >= len(output):
            continue
        output[index] = np.array(output[index - 1], copy=True)
    return output
