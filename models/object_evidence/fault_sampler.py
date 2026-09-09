"""Reproducible paired camera-fault episode sampler."""

from dataclasses import dataclass
import random
from typing import Tuple


FAULT_TYPES = ("blur", "dark", "crash")
SEVERITIES = (0.3, 0.6, 0.9)


@dataclass(frozen=True)
class FaultSpec:
    camera: int
    fault_type: str
    severity: float

    @property
    def strength(self) -> float:
        return 1.0 if self.fault_type == "crash" else self.severity


@dataclass(frozen=True)
class FaultEpisode:
    spec: FaultSpec
    active: Tuple[bool, ...]


class PairedFaultSampler:
    def __init__(self, seed: int = 0, num_cameras: int = 6, pair_probability: float = 0.5):
        if num_cameras <= 0 or not 0 <= pair_probability <= 1:
            raise ValueError("invalid sampler configuration")
        self._rng = random.Random(seed)
        self.num_cameras = int(num_cameras)
        self.pair_probability = float(pair_probability)

    def paired_iteration(self) -> bool:
        return self._rng.random() < self.pair_probability

    def sample(self, clip_length: int) -> FaultEpisode:
        if clip_length < 1:
            raise ValueError("clip_length must be positive")
        fault_type = self._rng.choice(FAULT_TYPES)
        severity = 1.0 if fault_type == "crash" else self._rng.choice(SEVERITIES)
        spec = FaultSpec(self._rng.randrange(self.num_cameras), fault_type, severity)
        onset = 0 if clip_length == 1 else min(2, clip_length - 1)
        return FaultEpisode(spec, tuple(index >= onset for index in range(clip_length)))
