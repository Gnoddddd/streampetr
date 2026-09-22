"""Pure planning, aggregation, and persistence for full-clean GeoCorr evaluation."""

import gzip
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


GROUP_NAMES = ("all", "top25", "top50", "top100", "q1", "q2", "q3", "q4")
VALID_VIEW_NAMES = ("valid_views_1", "valid_views_2", "valid_views_ge3")
GROUP_METRIC_NAMES = (
    "num_objects", "num_valid_queries", "num_same_view_queries",
    "normalized_full_entropy", "normalized_position_entropy",
    "normalized_view_entropy", "position_top1_probability",
    "center_candidate_mass", "position_top1_minus_top2_margin",
    "center_as_top1_rate", "same_view_normalized_position_entropy",
    "same_view_position_top1_probability", "same_view_center_mass",
    "same_view_top1_margin", "same_view_center_as_top1_rate",
    "score.min", "score.median", "score.max",
    "shuffled_history_control.correct_position_entropy",
    "shuffled_history_control.shuffled_position_entropy",
    "shuffled_history_control.delta_position_entropy",
    "shuffled_history_control.correct_position_top1",
    "shuffled_history_control.shuffled_position_top1",
    "shuffled_history_control.delta_position_top1",
    "shuffled_history_control.correct_center_mass",
    "shuffled_history_control.shuffled_center_mass",
    "shuffled_history_control.delta_center_mass",
    "shuffled_history_control.correct_top1_margin",
    "shuffled_history_control.shuffled_top1_margin",
    "shuffled_history_control.delta_top1_margin",
    "shuffled_history_control.correct_center_as_top1",
    "shuffled_history_control.shuffled_center_as_top1",
    "shuffled_history_control.delta_center_as_top1",
)
VALID_VIEW_METRIC_NAMES = (
    "query_count", "normalized_position_entropy", "position_top1_probability",
    "center_candidate_mass", "center_as_top1_rate",
    "position_top1_minus_top2_margin",
)


@dataclass(frozen=True)
class FrameRecord:
    dataset_index: int
    scene_token: str
    sample_token: str
    previous_sample_token: str
    timestamp: int


@dataclass(frozen=True)
class PairRecord:
    previous: FrameRecord
    current: FrameRecord

    @property
    def key(self) -> str:
        return "%s->%s" % (self.previous.sample_token, self.current.sample_token)


def build_frame_and_pair_plan(
    data_infos: Sequence[Mapping[str, object]],
) -> Tuple[List[FrameRecord], List[PairRecord]]:
    """Sort by scene/timestamp and derive only metadata-linked adjacent pairs."""
    frames = [
        FrameRecord(
            dataset_index=index,
            scene_token=str(info["scene_token"]),
            sample_token=str(info["token"]),
            previous_sample_token=str(info.get("prev", "")),
            timestamp=int(info["timestamp"]),
        )
        for index, info in enumerate(data_infos)
    ]
    frames.sort(key=lambda frame: (frame.scene_token, frame.timestamp, frame.sample_token))
    if len({frame.sample_token for frame in frames}) != len(frames):
        raise ValueError("dataset contains duplicate sample tokens")
    pairs = []
    for previous, current in zip(frames, frames[1:]):
        if (
            previous.scene_token == current.scene_token
            and current.previous_sample_token == previous.sample_token
        ):
            pairs.append(PairRecord(previous, current))
    return frames, pairs


def resume_frame_start(
    frames: Sequence[FrameRecord],
    pairs: Sequence[PairRecord],
    processed_keys: Set[str],
) -> int:
    """Return the first frame of the earliest scene containing unfinished work."""
    unfinished = next((pair for pair in pairs if pair.key not in processed_keys), None)
    if unfinished is None:
        return len(frames)
    for index, frame in enumerate(frames):
        if frame.scene_token == unfinished.current.scene_token:
            return index
    raise ValueError("unfinished pair scene is absent from frame plan")


def cache_rollover(previous: Optional[object], current: object) -> object:
    """Explicit one-item cache transition, kept pure for regression testing."""
    del previous
    return current


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def descriptive_statistics(values: Sequence[float]) -> Dict[str, object]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) != len(values):
        raise ValueError("non-finite metric cannot be accumulated")
    if not finite:
        return {key: 0.0 if key != "count" else 0 for key in (
            "count", "mean", "median", "std", "p25", "p75"
        )}
    ordered = sorted(finite)
    mean = sum(ordered) / len(ordered)
    variance = sum((value - mean) ** 2 for value in ordered) / len(ordered)
    return {
        "count": len(ordered),
        "mean": mean,
        "median": _quantile(ordered, 0.5),
        "std": math.sqrt(variance),
        "p25": _quantile(ordered, 0.25),
        "p75": _quantile(ordered, 0.75),
    }


def compact_group_summary(group: Mapping[str, object]) -> Dict[str, object]:
    """Create the stable per-pair schema consumed by the aggregate evaluator."""
    scalar_metrics = (
        "normalized_full_entropy",
        "normalized_position_entropy",
        "normalized_view_entropy",
        "position_top1_probability",
        "center_candidate_mass",
        "position_top1_minus_top2_margin",
        "same_view_normalized_position_entropy",
        "same_view_position_top1_probability",
        "same_view_center_mass",
        "same_view_top1_margin",
    )
    result = {
        "num_objects": int(group["num_objects"]),
        "num_valid_queries": int(group["num_valid_queries"]),
        "num_same_view_queries": int(group["num_same_view_queries"]),
        "center_as_top1_rate": float(group["center_as_top1_rate"]),
        "same_view_center_as_top1_rate": float(
            group["same_view_center_as_top1_rate"]
        ),
        "score": dict(group["score"]),
        "position_top1_offset_histogram": dict(
            group["position_top1_offset_histogram"]
        ),
    }
    for name in scalar_metrics:
        result[name] = float(group[name]["mean"])
    control = group["shuffled_history_control"]
    control_names = {
        "normalized_position_entropy": "position_entropy",
        "position_top1_probability": "position_top1",
        "center_candidate_mass": "center_mass",
        "position_top1_minus_top2_margin": "top1_margin",
        "center_as_top1_rate": "center_as_top1",
    }
    comparison = {}
    for source, target in control_names.items():
        entry = control[source]
        if source == "center_as_top1_rate":
            comparison["correct_" + target] = float(entry["correct"])
            comparison["shuffled_" + target] = float(entry["shuffled"])
            comparison["delta_" + target] = float(entry["delta"])
        else:
            comparison["correct_" + target] = float(entry["correct"]["mean"])
            comparison["shuffled_" + target] = float(entry["shuffled"]["mean"])
            comparison["delta_" + target] = float(entry["delta"]["mean"])
    result["shuffled_history_control"] = comparison
    return result


def compact_valid_view_summary(group: Mapping[str, object]) -> Dict[str, object]:
    return {
        "query_count": int(group["query_count"]),
        "normalized_position_entropy": float(
            group["normalized_position_entropy"]["mean"]
        ),
        "position_top1_probability": float(
            group["position_top1_probability"]["mean"]
        ),
        "center_candidate_mass": float(group["center_candidate_mass"]["mean"]),
        "center_as_top1_rate": float(group["center_as_top1_rate"]),
        "position_top1_minus_top2_margin": float(
            group["position_top1_minus_top2_margin"]["mean"]
        ),
    }


def _flatten_scalars(value: Mapping[str, object], prefix: str = "") -> Dict[str, float]:
    flattened = {}
    for key, child in value.items():
        name = "%s.%s" % (prefix, key) if prefix else key
        if isinstance(child, Mapping):
            if key not in ("position_top1_offset_histogram",):
                flattened.update(_flatten_scalars(child, name))
        elif isinstance(child, (int, float)):
            flattened[name] = float(child)
    return flattened


class ResultAccumulator:
    """Bounded-by-pair CPU aggregation; no feature/query tensors are retained."""

    def __init__(self) -> None:
        self.records: List[Mapping[str, object]] = []

    def add(self, record: Mapping[str, object]) -> None:
        self.records.append(record)

    def summary(self) -> Dict[str, object]:
        scene_tokens = set()
        frame_tokens = set()
        prediction_objects = 0
        valid_queries = 0
        same_queries = 0
        for record in self.records:
            scene_tokens.add(record["scene_token"])
            frame_tokens.update(
                (record["previous_sample_token"], record["current_sample_token"])
            )
            prediction_objects += int(record["num_predictions"])
            valid_queries += int(record["num_valid_queries"])
            same_queries += int(record["num_same_view_queries"])
        output: Dict[str, object] = {
            "num_scenes": len(scene_tokens),
            "num_frames": len(frame_tokens),
            "num_pairs": len(self.records),
            "num_prediction_objects": prediction_objects,
            "num_valid_queries": valid_queries,
            "num_same_view_queries": same_queries,
            "aggregation_unit": "pair-level metric means",
        }
        for group_name in GROUP_NAMES:
            values_by_metric: Dict[str, List[float]] = {
                name: [] for name in GROUP_METRIC_NAMES
            }
            histogram: Dict[str, int] = {}
            for record in self.records:
                group = record["groups"][group_name]
                for metric, value in _flatten_scalars(group).items():
                    values_by_metric.setdefault(metric, []).append(value)
                for offset, count in group["position_top1_offset_histogram"].items():
                    histogram[offset] = histogram.get(offset, 0) + int(count)
            output[group_name] = {
                metric: descriptive_statistics(values)
                for metric, values in sorted(values_by_metric.items())
            }
            output[group_name]["position_top1_offset_histogram"] = histogram
        for group_name in VALID_VIEW_NAMES:
            values_by_metric = {name: [] for name in VALID_VIEW_METRIC_NAMES}
            for record in self.records:
                group = record["valid_view_groups"][group_name]
                for metric, value in _flatten_scalars(group).items():
                    values_by_metric.setdefault(metric, []).append(value)
            output[group_name] = {
                metric: descriptive_statistics(values)
                for metric, values in sorted(values_by_metric.items())
            }
        return output


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(encoded)
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def iter_jsonl_gz(path: Path) -> Iterable[Mapping[str, object]]:
    if not path.exists():
        return
    with gzip.open(str(path), "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class EvaluationStore:
    """Crash-tolerant append store with duplicate prevention and rebuildable state."""

    def __init__(self, output_dir: Path, resume: bool) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pair_path = output_dir / "per_pair.jsonl.gz"
        self.error_path = output_dir / "errors.jsonl"
        self.progress_path = output_dir / "progress.json"
        if not resume and any(
            path.exists() for path in (self.pair_path, self.error_path, self.progress_path)
        ):
            raise FileExistsError(
                "output contains prior progress; choose a new directory or use --resume"
            )
        self.accumulator = ResultAccumulator()
        self.processed_keys: Set[str] = set()
        for record in iter_jsonl_gz(self.pair_path):
            key = str(record["pair_key"])
            if key in self.processed_keys:
                raise ValueError("duplicate pair in per_pair file: %s" % key)
            self.processed_keys.add(key)
            self.accumulator.add(record)
        if self.error_path.exists():
            with self.error_path.open() as handle:
                for line in handle:
                    if line.strip():
                        self.processed_keys.add(str(json.loads(line)["pair_key"]))

    def append_pair(self, record: Mapping[str, object]) -> bool:
        key = str(record["pair_key"])
        if key in self.processed_keys:
            return False
        encoded = json.dumps(record, sort_keys=True, allow_nan=False)
        with gzip.open(str(self.pair_path), "at") as handle:
            handle.write(encoded + "\n")
        self.processed_keys.add(key)
        self.accumulator.add(record)
        return True

    def append_error(self, record: Mapping[str, object]) -> bool:
        key = str(record["pair_key"])
        if key in self.processed_keys:
            return False
        encoded = json.dumps(record, sort_keys=True, allow_nan=False)
        with self.error_path.open("a") as handle:
            handle.write(encoded + "\n")
        self.processed_keys.add(key)
        return True

    def write_progress(self, payload: Mapping[str, object]) -> None:
        atomic_write_json(self.progress_path, payload)


__all__ = [
    "EvaluationStore",
    "FrameRecord",
    "GROUP_NAMES",
    "PairRecord",
    "ResultAccumulator",
    "VALID_VIEW_NAMES",
    "atomic_write_json",
    "build_frame_and_pair_plan",
    "cache_rollover",
    "compact_group_summary",
    "compact_valid_view_summary",
    "descriptive_statistics",
    "iter_jsonl_gz",
    "resume_frame_start",
]
