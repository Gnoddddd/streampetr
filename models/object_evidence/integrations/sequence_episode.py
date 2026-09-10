"""Deterministic persistent-fault state for native temporal training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import random
from typing import Dict, List, Optional, Sequence

from ..fault_sampler import FAULT_TYPES, SEVERITIES, FaultSpec


def _identity_seed(base_seed: int, identity: str) -> int:
    payload = f"{int(base_seed)}:{identity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def deterministic_fault_spec(base_seed: int, identity: str, num_cameras: int = 6) -> FaultSpec:
    """Derive a stable fault spec without process-local RNG state."""
    rng = random.Random(_identity_seed(base_seed, identity))
    fault_type = rng.choice(FAULT_TYPES)
    severity = 1.0 if fault_type == "crash" else rng.choice(SEVERITIES)
    return FaultSpec(rng.randrange(int(num_cameras)), fault_type, severity)


def deterministic_fault_selected(base_seed: int, identity: str, probability: float) -> bool:
    if not 0 <= probability <= 1:
        raise ValueError("fault probability must be in [0,1]")
    rng = random.Random(_identity_seed(base_seed, f"selection:{identity}"))
    return rng.random() < float(probability)


@dataclass(frozen=True)
class SequenceFrameDecision:
    scene_token: str
    sample_token: str
    sequence_id: str
    frame_index: int
    fault_sequence: bool
    fault_spec: FaultSpec
    fault_active: bool
    sequence_start: bool


@dataclass
class _SlotState:
    scene_token: str
    sequence_id: str
    frame_index: int
    fault_sequence: bool
    fault_spec: FaultSpec


class PersistentSequenceFaultState:
    """Track one deterministic episode per StreamPETR sequence/batch lane."""

    def __init__(
        self, base_seed: int = 2026, pair_probability: float = 0.5,
        num_cameras: int = 6, onset_frames: int = 2,
    ) -> None:
        self.base_seed = int(base_seed)
        self.pair_probability = float(pair_probability)
        self.num_cameras = int(num_cameras)
        self.onset_frames = int(onset_frames)
        self._slots: Dict[int, _SlotState] = {}

    def advance_batch(
        self,
        scene_tokens: Sequence[str],
        sample_tokens: Sequence[str],
        prev_exists: Sequence[bool],
        force_fault: bool = False,
        force_active: bool = False,
    ) -> List[SequenceFrameDecision]:
        if not (len(scene_tokens) == len(sample_tokens) == len(prev_exists)):
            raise ValueError("scene/sample/prev batch dimensions must match")
        decisions = []
        for slot, (scene, sample, has_previous) in enumerate(
            zip(scene_tokens, sample_tokens, prev_exists)
        ):
            scene, sample = str(scene), str(sample)
            prior = self._slots.get(slot)
            sequence_start = (
                not bool(has_previous) or prior is None or prior.scene_token != scene
            )
            if sequence_start:
                sequence_id = f"{scene}:{sample}"
                selected = force_fault or deterministic_fault_selected(
                    self.base_seed, sequence_id, self.pair_probability
                )
                prior = _SlotState(
                    scene_token=scene,
                    sequence_id=sequence_id,
                    frame_index=0,
                    fault_sequence=selected,
                    fault_spec=deterministic_fault_spec(
                        self.base_seed, sequence_id, self.num_cameras
                    ),
                )
            else:
                prior.frame_index += 1
                if force_fault:
                    prior.fault_sequence = True
            self._slots[slot] = prior
            active = prior.fault_sequence and prior.frame_index >= self.onset_frames
            decisions.append(SequenceFrameDecision(
                scene_token=scene,
                sample_token=sample,
                sequence_id=prior.sequence_id,
                frame_index=prior.frame_index,
                fault_sequence=prior.fault_sequence,
                fault_spec=prior.fault_spec,
                fault_active=bool(active or (force_fault and force_active)),
                sequence_start=sequence_start,
            ))
        return decisions

    def state_dict(self) -> Dict:
        return {
            "base_seed": self.base_seed,
            "pair_probability": self.pair_probability,
            "num_cameras": self.num_cameras,
            "onset_frames": self.onset_frames,
            "slots": {
                int(slot): {
                    **asdict(state),
                    "fault_spec": asdict(state.fault_spec),
                }
                for slot, state in self._slots.items()
            },
        }

    def load_state_dict(self, state: Dict) -> None:
        self.base_seed = int(state["base_seed"])
        self.pair_probability = float(state["pair_probability"])
        self.num_cameras = int(state["num_cameras"])
        self.onset_frames = int(state["onset_frames"])
        self._slots = {}
        for slot, values in state.get("slots", {}).items():
            values = dict(values)
            values["fault_spec"] = FaultSpec(**values["fault_spec"])
            self._slots[int(slot)] = _SlotState(**values)


def deterministic_sample_faults(
    base_seed: int,
    sample_tokens: Sequence[str],
    probability: float,
    num_cameras: int = 6,
    force_fault: bool = False,
) -> List[Optional[FaultSpec]]:
    """Return restart-stable BEVDepth sample-level fault assignments."""
    result = []
    for token in sample_tokens:
        identity = str(token)
        selected = force_fault or deterministic_fault_selected(
            base_seed, identity, probability
        )
        result.append(
            deterministic_fault_spec(base_seed, identity, num_cameras)
            if selected else None
        )
    return result
