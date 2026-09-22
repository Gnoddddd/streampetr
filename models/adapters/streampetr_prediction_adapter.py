"""Convert decoded StreamPETR output without touching its query memory."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class PreviousPrediction:
    """Detector-independent decoded 3D predictions from one previous frame.

    Coordinates use the frame named by ``coordinate_frame``.  For the
    StreamPETR adapter this is the previous key lidar frame: x forward, y left,
    z up.  ``center_3d`` is the box gravity center and velocity is expressed in
    the same frame in metres/second.  Every tensor is detached on construction.
    """

    center_3d: Tensor  # [N, 3]
    size_3d: Tensor  # [N, 3], (width, length, height)
    yaw: Tensor  # [N]
    velocity: Optional[Tensor]  # [N, 2], or None
    score: Tensor  # [N]
    label: Tensor  # [N]
    timestamp: float
    coordinate_frame: str = "previous_lidar"

    def __post_init__(self) -> None:
        count = self.center_3d.shape[0]
        expected = {
            "center_3d": (count, 3),
            "size_3d": (count, 3),
            "yaw": (count,),
            "score": (count,),
            "label": (count,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError("%s must have shape %s" % (name, shape))
            object.__setattr__(self, name, value.detach())
        if self.velocity is not None:
            if tuple(self.velocity.shape) != (count, 2):
                raise ValueError("velocity must have shape [N,2]")
            object.__setattr__(self, "velocity", self.velocity.detach())


def from_streampetr_result(
    result: Mapping[str, Any], timestamp: float, coordinate_frame: str = "previous_lidar"
) -> PreviousPrediction:
    """Convert one normal ``simple_test`` result into ``PreviousPrediction``.

    Accepted input is either the inner ``pts_bbox`` mapping or the outer result
    containing it.  This function reads predictions only; GT fields are neither
    accepted nor inspected.
    """
    prediction = result.get("pts_bbox", result)
    boxes = prediction["boxes_3d"]
    scores = torch.as_tensor(prediction["scores_3d"])
    labels = torch.as_tensor(prediction["labels_3d"])
    tensor = boxes.tensor if hasattr(boxes, "tensor") else torch.as_tensor(boxes)
    if tensor.ndim != 2 or tensor.shape[1] < 7:
        raise ValueError("StreamPETR boxes must have shape [N,>=7]")
    if hasattr(boxes, "gravity_center"):
        centers = boxes.gravity_center
    else:
        # StreamPETR's decoded result uses bottom-centered z before wrapping in
        # LiDARInstance3DBoxes; reproduce gravity_center for plain test tensors.
        centers = tensor[:, :3].clone()
        centers[:, 2] += tensor[:, 5] * 0.5
    velocity = tensor[:, 7:9] if tensor.shape[1] >= 9 else None
    return PreviousPrediction(
        center_3d=centers,
        size_3d=tensor[:, 3:6],
        yaw=tensor[:, 6],
        velocity=velocity,
        score=scores,
        label=labels,
        timestamp=float(timestamp),
        coordinate_frame=coordinate_frame,
    )


__all__ = ["PreviousPrediction", "from_streampetr_result"]
