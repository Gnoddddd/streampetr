"""Serializable PartialObs-3D corruption schedules."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class ProtocolEvent:
    start_frame: int
    end_frame: int
    failed_cameras: List[str] = field(default_factory=list)
    lost_cameras: List[str] = field(default_factory=list)
    dark: Dict[str, float] = field(default_factory=dict)
    fog: Dict[str, float] = field(default_factory=dict)
    motion_blur: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame must be greater than or equal to start_frame")
        for name, mapping in (
            ("dark", self.dark),
            ("fog", self.fog),
            ("motion_blur", self.motion_blur),
        ):
            for camera, severity in mapping.items():
                if not 0.0 <= float(severity) <= 1.0:
                    raise ValueError(
                        f"{name} severity for {camera} must be in [0,1]"
                    )

    def active(self, frame_idx: int) -> bool:
        return self.start_frame <= frame_idx <= self.end_frame

    @classmethod
    def from_dict(cls, value: Mapping) -> "ProtocolEvent":
        return cls(
            start_frame=int(value["start_frame"]),
            end_frame=int(value["end_frame"]),
            failed_cameras=list(value.get("failed_cameras", [])),
            lost_cameras=list(value.get("lost_cameras", [])),
            dark={str(k): float(v) for k, v in value.get("dark", {}).items()},
            fog={str(k): float(v) for k, v in value.get("fog", {}).items()},
            motion_blur={str(k): float(v) for k, v in value.get("motion_blur", {}).items()},
        )


class ProtocolSchedule:
    def __init__(self, scenes: Optional[Mapping[str, Iterable[ProtocolEvent]]] = None) -> None:
        self.scenes: Dict[str, List[ProtocolEvent]] = {
            str(scene): list(events) for scene, events in (scenes or {}).items()
        }

    @classmethod
    def from_json(cls, path: str) -> "ProtocolSchedule":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = int(payload.get("version", 1))
        if version != 1:
            raise ValueError(f"Unsupported protocol version: {version}")
        scenes = {
            scene: [ProtocolEvent.from_dict(event) for event in events]
            for scene, events in payload.get("scenes", {}).items()
        }
        return cls(scenes)

    def to_json(self, path: str) -> None:
        payload = {
            "version": 1,
            "scenes": {
                scene: [asdict(event) for event in events]
                for scene, events in self.scenes.items()
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def state_for(
        self, scene_token: str, frame_idx: int, camera_names: Sequence[str]
    ) -> Dict:
        events = list(self.scenes.get("*", [])) + list(self.scenes.get(scene_token, []))
        state = {
            "failed_cameras": [],
            "lost_cameras": [],
            "dark": {},
            "fog": {},
            "motion_blur": {},
        }
        valid = set(camera_names)
        for event in events:
            if not event.active(frame_idx):
                continue
            state["failed_cameras"].extend(
                name for name in event.failed_cameras if name in valid
            )
            state["lost_cameras"].extend(
                name for name in event.lost_cameras if name in valid
            )
            for key in ("dark", "fog", "motion_blur"):
                state[key].update(
                    {
                        name: float(value)
                        for name, value in getattr(event, key).items()
                        if name in valid
                    }
                )
        state["failed_cameras"] = sorted(set(state["failed_cameras"]))
        state["lost_cameras"] = sorted(set(state["lost_cameras"]))
        return state
