"""OccNuScenes condition discovery, temporal-pair manifests, and auditing.

Manifest paths are deliberately dataset-relative: clean paths are relative to
the nuScenes root and dirty paths are relative to the OccNuScenes root.  This
keeps manifests portable while :class:`PairedOccNuScenesDataset` exposes
resolved paths for the three Stage 3 branches.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from torch.utils.data import Dataset

from datasets.temporal_occ_nuscenes import NUSCENES_CAMERA_ORDER


IMAGE_SUFFIXES = frozenset((".jpg", ".jpeg", ".png"))
MANIFEST_PATH_FIELDS = (
    "previous_clean_paths",
    "current_clean_paths",
    "current_dirty_paths",
)
ERROR_CATEGORIES = (
    "missing_clean",
    "missing_dirty",
    "missing_metadata",
    "temporal_errors",
    "camera_errors",
    "duplicates",
    "split_leakage",
)
_SEVERITY_PREFIX = re.compile(r"^(?P<severity>(?:\d+(?:\.\d*)?|\.\d+))(?:_|-|$)")


@dataclass(frozen=True)
class ConditionImage:
    path: Path
    relative_path: str
    raw_condition: str
    corruption_type: str
    severity: Optional[float]
    camera: str
    filename: str


@dataclass
class IntegrityReport:
    errors: Dict[str, List[str]] = field(
        default_factory=lambda: {name: [] for name in ERROR_CATEGORIES}
    )
    conditions: List[str] = field(default_factory=list)
    corruption_types: List[str] = field(default_factory=list)
    per_condition: Dict[str, Dict[str, int]] = field(default_factory=dict)
    per_corruption: Dict[str, int] = field(default_factory=dict)
    per_severity: Dict[str, int] = field(default_factory=dict)
    split: Dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not any(self.errors.values())

    def add(self, category: str, message: str) -> None:
        if category not in self.errors:
            raise KeyError("unknown integrity category: %s" % category)
        self.errors[category].append(message)


class NuScenesMetadata:
    """Small indexed view over the three nuScenes tables needed by Stage 3-A."""

    def __init__(
        self,
        sample_data: Sequence[Mapping[str, Any]],
        samples: Sequence[Mapping[str, Any]],
        scenes: Sequence[Mapping[str, Any]],
        calibrated_sensors: Sequence[Mapping[str, Any]] = (),
        sensors: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.sample_data_by_token = _unique_index(sample_data, "token", "sample_data")
        self.sample_data_by_filename = _unique_index(
            [
                row for row in sample_data
                if _is_sample_camera_filename(str(row.get("filename", "")))
            ],
            "filename",
            "sample camera filename",
        )
        self.samples_by_token = _unique_index(samples, "token", "sample")
        self.scenes_by_token = _unique_index(scenes, "token", "scene")
        self.calibrated_sensors_by_token = _unique_index(
            calibrated_sensors, "token", "calibrated_sensor"
        )
        self.sensors_by_token = _unique_index(sensors, "token", "sensor")
        self.sample_camera_data: Dict[
            Tuple[str, str], Mapping[str, Any]
        ] = {}
        for row in sample_data:
            if row.get("is_key_frame") is not True:
                continue
            identity = self.camera_identity(row)
            if identity is None:
                continue
            modality, channel = identity
            if modality != "camera" or channel not in NUSCENES_CAMERA_ORDER:
                continue
            key = (str(row.get("sample_token", "")), channel)
            existing = self.sample_camera_data.get(key)
            if existing is not None:
                raise ValueError(
                    "duplicate keyframe sample camera channel: %s %s (%s, %s)"
                    % (key[0], channel, existing.get("token"), row.get("token"))
                )
            self.sample_camera_data[key] = row

    @classmethod
    def from_directory(cls, directory: Union[Path, str]) -> "NuScenesMetadata":
        root = Path(directory)
        return cls(
            _read_json_table(root / "sample_data.json"),
            _read_json_table(root / "sample.json"),
            _read_json_table(root / "scene.json"),
            _read_json_table(root / "calibrated_sensor.json"),
            _read_json_table(root / "sensor.json"),
        )

    def camera_identity(
        self, sample_data: Mapping[str, Any]
    ) -> Optional[Tuple[str, str]]:
        calibrated = self.calibrated_sensors_by_token.get(
            str(sample_data.get("calibrated_sensor_token", ""))
        )
        if calibrated is None:
            return None
        sensor = self.sensors_by_token.get(str(calibrated.get("sensor_token", "")))
        if sensor is None:
            return None
        return str(sensor.get("modality", "")), str(sensor.get("channel", ""))


def _read_json_table(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError("nuScenes table is not a list: %s" % path)
    return value


def _unique_index(
    rows: Sequence[Mapping[str, Any]], key: str, label: str
) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise ValueError("duplicate %s: %s" % (label, value))
        result[value] = row
    return result


def _is_sample_camera_filename(filename: str) -> bool:
    """Return whether ``filename`` is a canonical nuScenes sample camera image."""
    normalized = filename.replace("\\", "/")
    path = PurePosixPath(normalized)
    return bool(
        not path.is_absolute()
        and len(path.parts) == 3
        and path.parts[0] == "samples"
        and path.parts[1] in NUSCENES_CAMERA_ORDER
        and path.name not in ("", ".", "..")
        and path.suffix.lower() in IMAGE_SUFFIXES
    )


def parse_condition(raw_condition: str) -> Tuple[str, Optional[float]]:
    """Return top-level corruption type and a numeric child prefix, if any."""
    parts = PurePosixPath(raw_condition.replace("\\", "/")).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError("invalid raw condition: %r" % raw_condition)
    severity = None
    for component in parts[1:]:
        match = _SEVERITY_PREFIX.match(component)
        if match:
            severity = float(match.group("severity"))
            break
    return parts[0], severity


def discover_condition_images(
    dirty_root: Union[Path, str],
    corruption_types: Optional[Sequence[str]] = None,
    conditions: Optional[Sequence[str]] = None,
) -> List[ConditionImage]:
    """Recursively discover images and derive condition data before ``CAM_*``."""
    root = Path(dirty_root)
    allowed_types = set(corruption_types) if corruption_types else None
    allowed_conditions = set(conditions) if conditions else None
    discovered = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(root)
        camera_indices = [
            index for index, component in enumerate(relative.parts)
            if component.startswith("CAM_")
        ]
        if not camera_indices or camera_indices[0] == 0:
            continue
        camera_index = camera_indices[0]
        raw_condition = PurePosixPath(*relative.parts[:camera_index]).as_posix()
        corruption_type, severity = parse_condition(raw_condition)
        if allowed_types is not None and corruption_type not in allowed_types:
            continue
        if allowed_conditions is not None and raw_condition not in allowed_conditions:
            continue
        discovered.append(ConditionImage(
            path=path,
            relative_path=relative.as_posix(),
            raw_condition=raw_condition,
            corruption_type=corruption_type,
            severity=severity,
            camera=relative.parts[camera_index],
            filename=relative.name,
        ))
    return discovered


def clean_filename_for(image: ConditionImage) -> str:
    return PurePosixPath("samples", image.camera, image.filename).as_posix()


def _clean_filename_candidates(image: ConditionImage) -> List[str]:
    candidates = [clean_filename_for(image)]
    dirty = PurePosixPath(image.filename)
    if dirty.stem.endswith("_obstructed"):
        normalized_name = dirty.stem[:-len("_obstructed")] + dirty.suffix
        normalized = PurePosixPath(
            "samples", image.camera, normalized_name
        ).as_posix()
        if normalized not in candidates:
            candidates.append(normalized)
    return candidates


def resolve_clean_filename(
    image: ConditionImage, metadata: NuScenesMetadata
) -> str:
    """Resolve one dirty basename to exactly one Camera sample_data filename."""
    matches = [
        candidate for candidate in _clean_filename_candidates(image)
        if candidate in metadata.sample_data_by_filename
    ]
    if not matches:
        raise LookupError(
            "no Camera sample_data for dirty image %s (candidates: %s)"
            % (image.relative_path, _clean_filename_candidates(image))
        )
    if len(matches) != 1:
        raise ValueError(
            "ambiguous Camera sample_data for dirty image %s: %s"
            % (image.relative_path, matches)
        )
    return matches[0]


SUPPORTED_SPLITS = ("mini_train", "mini_val", "train", "val", "all")
_DISJOINT_SPLIT_PAIRS = (("mini_train", "mini_val"), ("train", "val"))


def official_splits() -> Dict[str, Sequence[str]]:
    """Load official nuScenes scene splits lazily for filtering.

    The returned mapping intentionally includes both mini and full trainval
    splits.  A mini split is not a subset-label for a full split: for example,
    a mini-train scene may belong to full validation.
    """
    try:
        from nuscenes.utils.splits import create_splits_scenes
    except ImportError as exc:  # pragma: no cover - environment-specific message
        raise RuntimeError("nuscenes-devkit is required for official split filtering") from exc
    splits = create_splits_scenes()
    return {
        name: tuple(splits[name])
        for name in ("mini_train", "mini_val", "train", "val")
    }


def official_mini_splits() -> Dict[str, Sequence[str]]:
    """Backward-compatible mini-only view of :func:`official_splits`."""
    splits = official_splits()
    return {name: splits[name] for name in ("mini_train", "mini_val")}


def _split_membership(
    split_scene_names: Mapping[str, Sequence[str]],
) -> Tuple[Dict[str, str], List[str]]:
    membership: Dict[str, str] = {}
    overlap = []
    for left, right in _DISJOINT_SPLIT_PAIRS:
        overlap.extend(
            "%s/%s: %s" % (left, right, name)
            for name in sorted(
                set(split_scene_names.get(left, ()))
                & set(split_scene_names.get(right, ()))
            )
        )
    # Preserve historical mini labels for legacy/all manifests. Full train/val
    # filtering below uses the requested set directly, never this precedence.
    for split_name in ("mini_train", "mini_val", "train", "val"):
        for scene_name in split_scene_names.get(split_name, ()):
            membership.setdefault(scene_name, split_name)
    return membership, overlap


def _camera_rows(
    sample_token: str, metadata: NuScenesMetadata
) -> Optional[List[Mapping[str, Any]]]:
    rows = []
    for camera in NUSCENES_CAMERA_ORDER:
        row = metadata.sample_camera_data.get((sample_token, camera))
        if row is None:
            return None
        expected = PurePosixPath("samples", camera, PurePosixPath(str(row["filename"])).name)
        if PurePosixPath(str(row["filename"])) != expected:
            return None
        rows.append(row)
    return rows


def build_pair_records(
    nuscenes_root: Union[Path, str],
    dirty_root: Union[Path, str],
    metadata: NuScenesMetadata,
    split: str = "all",
    corruption_types: Optional[Sequence[str]] = ("Dirt", "Water-blur"),
    conditions: Optional[Sequence[str]] = None,
    split_scene_names: Optional[Mapping[str, Sequence[str]]] = None,
) -> Tuple[List[Dict[str, Any]], IntegrityReport]:
    """Build deterministic ``(clean t-1, clean t, dirty t)`` records."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError("unsupported split: %s" % split)
    split_scene_names = split_scene_names or official_splits()
    membership, split_overlap = _split_membership(split_scene_names)
    requested_scene_names = (
        None if split == "all" else set(split_scene_names.get(split, ()))
    )
    report = IntegrityReport()
    for name in split_overlap:
        report.add("split_leakage", "official split overlap: %s" % name)

    images = discover_condition_images(dirty_root, corruption_types, conditions)
    grouped: Dict[str, Dict[str, Dict[str, ConditionImage]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    matched_per_condition: Counter[str] = Counter()
    scene_sets: Dict[str, set[str]] = defaultdict(set)
    for image in images:
        try:
            clean_filename = resolve_clean_filename(image, metadata)
        except LookupError as exc:
            report.add("missing_metadata", str(exc))
            continue
        except ValueError as exc:
            report.add("duplicates", str(exc))
            continue
        if not (Path(nuscenes_root) / clean_filename).is_file():
            report.add("missing_clean", "missing clean image: %s" % clean_filename)
            continue
        sample_data = metadata.sample_data_by_filename[clean_filename]
        sample_token = str(sample_data.get("sample_token", ""))
        sample = metadata.samples_by_token.get(sample_token)
        if sample is None:
            report.add("missing_metadata", "no sample for %s" % sample_token)
            continue
        identity = metadata.camera_identity(sample_data)
        indexed = metadata.sample_camera_data.get((sample_token, image.camera))
        if (
            sample_data.get("is_key_frame") is not True
            or identity != ("camera", image.camera)
            or indexed is None
            or str(indexed.get("token")) != str(sample_data["token"])
        ):
            report.add(
                "camera_errors",
                "sample camera mismatch: %s %s" % (sample_token, image.camera),
            )
            continue
        by_camera = grouped[image.raw_condition][sample_token]
        if image.camera in by_camera:
            report.add(
                "duplicates",
                "duplicate dirty camera: %s %s %s"
                % (image.raw_condition, sample_token, image.camera),
            )
            continue
        by_camera[image.camera] = image
        matched_per_condition[image.raw_condition] += 1
        scene_sets[image.raw_condition].add(str(sample.get("scene_token", "")))

    records: List[Dict[str, Any]] = []
    complete_per_condition: Counter[str] = Counter()
    for raw_condition in sorted(grouped):
        sample_images = grouped[raw_condition]
        complete_tokens = {
            token for token, cameras in sample_images.items()
            if set(cameras) == set(NUSCENES_CAMERA_ORDER)
        }
        for token, cameras in sorted(sample_images.items()):
            for camera in sorted(set(NUSCENES_CAMERA_ORDER) - set(cameras)):
                report.add(
                    "missing_dirty",
                    "missing dirty camera: %s %s %s"
                    % (raw_condition, token, camera),
                )
            extra = sorted(set(cameras) - set(NUSCENES_CAMERA_ORDER))
            if extra:
                report.add(
                    "camera_errors",
                    "unexpected dirty cameras: %s %s %s"
                    % (raw_condition, token, extra),
                )
        complete_per_condition[raw_condition] = len(complete_tokens)
        corruption_type, severity = parse_condition(raw_condition)
        for current_token in sorted(complete_tokens):
            current = metadata.samples_by_token[current_token]
            scene = metadata.scenes_by_token.get(str(current.get("scene_token", "")))
            if scene is None:
                report.add("missing_metadata", "missing scene for %s" % current_token)
                continue
            scene_name = str(scene.get("name", ""))
            record_split = membership.get(scene_name)
            if requested_scene_names is not None and scene_name not in requested_scene_names:
                continue
            if split != "all":
                record_split = split
            previous_token = str(current.get("prev", ""))
            if not previous_token or previous_token not in complete_tokens:
                continue
            previous = metadata.samples_by_token.get(previous_token)
            if previous is None:
                report.add("missing_metadata", "missing previous sample %s" % previous_token)
                continue
            if str(previous.get("scene_token")) != str(current.get("scene_token")):
                report.add("temporal_errors", "scene boundary %s -> %s" % (
                    previous_token, current_token))
                continue
            previous_rows = _camera_rows(previous_token, metadata)
            current_rows = _camera_rows(current_token, metadata)
            if previous_rows is None or current_rows is None:
                report.add("camera_errors", "incomplete camera metadata: %s -> %s" % (
                    previous_token, current_token))
                continue
            missing_pair_clean = [
                str(row["filename"])
                for row in previous_rows + current_rows
                if not (Path(nuscenes_root) / str(row["filename"])).is_file()
            ]
            if missing_pair_clean:
                for filename in missing_pair_clean:
                    report.add("missing_clean", "missing clean image: %s" % filename)
                continue
            previous_timestamp = int(previous["timestamp"])
            current_timestamp = int(current["timestamp"])
            dt = (current_timestamp - previous_timestamp) / 1_000_000.0
            if current_timestamp <= previous_timestamp or dt <= 0:
                report.add("temporal_errors", "non-positive dt: %s -> %s" % (
                    previous_token, current_token))
                continue
            dirty_by_camera = sample_images[current_token]
            record = {
                "scene_token": str(current["scene_token"]),
                "scene_name": scene_name,
                "split": record_split or "unassigned",
                "previous_sample_token": previous_token,
                "current_sample_token": current_token,
                "previous_timestamp": previous_timestamp,
                "current_timestamp": current_timestamp,
                "dt": dt,
                "raw_condition": raw_condition,
                "corruption_type": corruption_type,
                "severity": severity,
                "camera_names": list(NUSCENES_CAMERA_ORDER),
                "previous_clean_paths": [str(row["filename"]) for row in previous_rows],
                "current_clean_paths": [str(row["filename"]) for row in current_rows],
                "current_dirty_paths": [
                    dirty_by_camera[camera].relative_path
                    for camera in NUSCENES_CAMERA_ORDER
                ],
                "previous_sample_data_tokens": [str(row["token"]) for row in previous_rows],
                "current_sample_data_tokens": [str(row["token"]) for row in current_rows],
                "previous_calibrated_sensor_tokens": [
                    str(row["calibrated_sensor_token"]) for row in previous_rows
                ],
                "current_calibrated_sensor_tokens": [
                    str(row["calibrated_sensor_token"]) for row in current_rows
                ],
                "previous_ego_pose_tokens": [str(row["ego_pose_token"]) for row in previous_rows],
                "current_ego_pose_tokens": [str(row["ego_pose_token"]) for row in current_rows],
            }
            records.append(record)

    records.sort(key=manifest_sort_key)
    report.conditions = sorted(grouped)
    report.corruption_types = sorted({parse_condition(value)[0] for value in grouped})
    pair_counts = Counter(row["raw_condition"] for row in records)
    dirty_counts = Counter(image.raw_condition for image in images)
    for condition in report.conditions:
        report.per_condition[condition] = {
            "dirty_images": dirty_counts[condition],
            "clean_matched": matched_per_condition[condition],
            "samples": len(grouped[condition]),
            "scenes": len(scene_sets[condition]),
            "legal_temporal_pairs": pair_counts[condition],
            "six_camera_complete_samples": complete_per_condition[condition],
        }
    report.per_corruption = dict(Counter(row["corruption_type"] for row in records))
    report.per_severity = dict(Counter(_severity_label(row["severity"]) for row in records))
    _populate_split_summary(report, records, split_scene_names)
    return records, report


def _severity_label(value: Optional[float]) -> str:
    return "None" if value is None else str(value)


def manifest_sort_key(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    severity = record.get("severity")
    return (
        str(record.get("scene_name", record.get("scene_token", ""))),
        int(record["current_timestamp"]),
        str(record["corruption_type"]),
        severity is None,
        0.0 if severity is None else float(severity),
        str(record["raw_condition"]),
    )


def write_manifest(
    path: Union[Path, str], records: Sequence[Mapping[str, Any]]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in sorted(records, key=manifest_sort_key):
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def read_manifest(path: Union[Path, str]) -> List[Dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("manifest line %d is not an object" % line_number)
                records.append(value)
    return records


def _populate_split_summary(
    report: IntegrityReport,
    records: Sequence[Mapping[str, Any]],
    split_scene_names: Mapping[str, Sequence[str]],
) -> None:
    for split_name in ("mini_train", "mini_val", "train", "val"):
        names = set(split_scene_names.get(split_name, ()))
        matching = [row for row in records if row.get("scene_name") in names]
        report.split[split_name + "_scenes"] = len({row["scene_name"] for row in matching})
        report.split[split_name + "_pairs"] = len(matching)


def audit_pair_records(
    records: Sequence[Mapping[str, Any]],
    nuscenes_root: Union[Path, str],
    dirty_root: Union[Path, str],
    metadata: NuScenesMetadata,
    split_scene_names: Optional[Mapping[str, Sequence[str]]] = None,
) -> IntegrityReport:
    """Validate manifest image, metadata, temporal, camera, and split integrity."""
    split_scene_names = split_scene_names or official_splits()
    _, split_overlap = _split_membership(split_scene_names)
    report = IntegrityReport()
    for name in split_overlap:
        report.add("split_leakage", "official split overlap: %s" % name)
    seen = set()
    condition_scenes: Dict[str, set[str]] = defaultdict(set)
    condition_samples: Dict[str, set[str]] = defaultdict(set)
    clean_matches: Counter[str] = Counter()
    complete: Counter[str] = Counter()
    for index, record in enumerate(records):
        label = "record[%d]" % index
        raw_condition = str(record.get("raw_condition", ""))
        condition_scenes[raw_condition].add(str(record.get("scene_token", "")))
        condition_samples[raw_condition].add(str(record.get("current_sample_token", "")))
        key = (raw_condition, str(record.get("current_sample_token", "")))
        if key in seen:
            report.add("duplicates", "%s duplicate key %r" % (label, key))
        seen.add(key)
        cameras = record.get("camera_names")
        if list(cameras or ()) != list(NUSCENES_CAMERA_ORDER):
            report.add("camera_errors", "%s camera order/completeness" % label)
        lengths_ok = True
        for field_name in MANIFEST_PATH_FIELDS + (
            "previous_sample_data_tokens", "current_sample_data_tokens",
            "previous_calibrated_sensor_tokens", "current_calibrated_sensor_tokens",
            "previous_ego_pose_tokens", "current_ego_pose_tokens",
        ):
            if len(record.get(field_name, ())) != len(NUSCENES_CAMERA_ORDER):
                report.add("camera_errors", "%s %s length" % (label, field_name))
                lengths_ok = False
        for field_name, root, category in (
            ("previous_clean_paths", Path(nuscenes_root), "missing_clean"),
            ("current_clean_paths", Path(nuscenes_root), "missing_clean"),
            ("current_dirty_paths", Path(dirty_root), "missing_dirty"),
        ):
            for relative in record.get(field_name, ()):
                if not (root / str(relative)).is_file():
                    report.add(category, "%s missing %s" % (label, relative))

        clean_paths = list(record.get("current_clean_paths", ()))
        dirty_paths = list(record.get("current_dirty_paths", ()))
        for camera_index, camera in enumerate(NUSCENES_CAMERA_ORDER):
            if camera_index >= len(clean_paths) or camera_index >= len(dirty_paths):
                continue
            dirty = PurePosixPath(str(dirty_paths[camera_index]))
            clean = PurePosixPath(str(clean_paths[camera_index]))
            expected_parent = PurePosixPath(raw_condition, camera)
            resolved_clean = None
            if dirty.parent == expected_parent:
                dirty_image = ConditionImage(
                    path=Path(dirty.as_posix()),
                    relative_path=dirty.as_posix(),
                    raw_condition=raw_condition,
                    corruption_type=str(record.get("corruption_type", "")),
                    severity=record.get("severity"),
                    camera=camera,
                    filename=dirty.name,
                )
                try:
                    resolved_clean = resolve_clean_filename(dirty_image, metadata)
                except LookupError:
                    resolved_clean = None
                except ValueError as exc:
                    report.add("duplicates", "%s %s" % (label, exc))
            if (
                dirty.parent != expected_parent
                or resolved_clean != clean.as_posix()
                or clean.parent != PurePosixPath("samples", camera)
            ):
                report.add(
                    "camera_errors",
                    "%s current dirty/clean path mismatch for %s" % (label, camera),
                )

        previous_token = str(record.get("previous_sample_token", ""))
        current_token = str(record.get("current_sample_token", ""))
        previous = metadata.samples_by_token.get(previous_token)
        current = metadata.samples_by_token.get(current_token)
        scene = metadata.scenes_by_token.get(str(record.get("scene_token", "")))
        if previous is None or current is None or scene is None:
            report.add("missing_metadata", "%s sample/scene lookup" % label)
            continue
        if str(current.get("prev", "")) != previous_token:
            report.add("temporal_errors", "%s current.prev mismatch" % label)
        if str(previous.get("scene_token")) != str(current.get("scene_token")):
            report.add("temporal_errors", "%s cross-scene pair" % label)
        previous_timestamp = int(previous["timestamp"])
        current_timestamp = int(current["timestamp"])
        expected_dt = (current_timestamp - previous_timestamp) / 1_000_000.0
        if previous_timestamp >= current_timestamp or expected_dt <= 0:
            report.add("temporal_errors", "%s non-positive timestamp delta" % label)
        if (
            int(record.get("previous_timestamp", -1)) != previous_timestamp
            or int(record.get("current_timestamp", -1)) != current_timestamp
            or abs(float(record.get("dt", -1)) - expected_dt) > 1e-12
        ):
            report.add("temporal_errors", "%s timestamp/dt mismatch" % label)
        if str(scene.get("name")) != str(record.get("scene_name")):
            report.add("missing_metadata", "%s scene name mismatch" % label)
        declared_split = str(record.get("split", "unassigned"))
        scene_name = str(scene.get("name"))
        if declared_split not in split_scene_names:
            expected_split = "unassigned"
            split_matches = declared_split == expected_split
        else:
            expected_split = declared_split
            split_matches = scene_name in set(split_scene_names[declared_split])
        if not split_matches:
            report.add("split_leakage", "%s split mismatch" % label)

        for prefix, sample, token_field, path_field in (
            ("previous", previous, "previous_sample_data_tokens", "previous_clean_paths"),
            ("current", current, "current_sample_data_tokens", "current_clean_paths"),
        ):
            tokens = list(record.get(token_field, ()))
            paths = list(record.get(path_field, ()))
            calibrated_tokens = list(record.get(
                prefix + "_calibrated_sensor_tokens", ()
            ))
            ego_pose_tokens = list(record.get(prefix + "_ego_pose_tokens", ()))
            for camera_index, camera in enumerate(NUSCENES_CAMERA_ORDER):
                if camera_index >= len(tokens) or camera_index >= len(paths):
                    continue
                row = metadata.sample_data_by_token.get(str(tokens[camera_index]))
                if row is None:
                    report.add("missing_metadata", "%s %s sample_data" % (label, prefix))
                    continue
                identity = metadata.camera_identity(row)
                indexed = metadata.sample_camera_data.get((str(sample["token"]), camera))
                if (
                    str(row.get("sample_token")) != str(sample["token"])
                    or str(row.get("filename")) != str(paths[camera_index])
                    or row.get("is_key_frame") is not True
                    or identity != ("camera", camera)
                    or indexed is None
                    or str(indexed.get("token")) != str(row["token"])
                ):
                    report.add("camera_errors", "%s %s %s metadata mismatch" % (
                        label, prefix, camera))
                elif (
                    camera_index >= len(calibrated_tokens)
                    or camera_index >= len(ego_pose_tokens)
                    or str(row.get("calibrated_sensor_token"))
                    != str(calibrated_tokens[camera_index])
                    or str(row.get("ego_pose_token"))
                    != str(ego_pose_tokens[camera_index])
                ):
                    report.add(
                        "missing_metadata",
                        "%s %s %s calibration/pose mismatch"
                        % (label, prefix, camera),
                    )
                elif prefix == "current":
                    clean_matches[raw_condition] += 1
        if lengths_ok and list(cameras or ()) == list(NUSCENES_CAMERA_ORDER):
            complete[raw_condition] += 1

    report.conditions = sorted({str(row.get("raw_condition", "")) for row in records})
    report.corruption_types = sorted({str(row.get("corruption_type", "")) for row in records})
    pair_counts = Counter(str(row.get("raw_condition", "")) for row in records)
    for condition in report.conditions:
        report.per_condition[condition] = {
            "dirty_images": pair_counts[condition] * len(NUSCENES_CAMERA_ORDER),
            "clean_matched": clean_matches[condition],
            "samples": len(condition_samples[condition]),
            "scenes": len(condition_scenes[condition]),
            "legal_temporal_pairs": pair_counts[condition],
            "six_camera_complete_samples": complete[condition],
        }
    report.per_corruption = dict(Counter(
        str(row.get("corruption_type", "")) for row in records
    ))
    report.per_severity = dict(Counter(
        _severity_label(row.get("severity")) for row in records
    ))
    _populate_split_summary(report, records, split_scene_names)
    return report


def format_integrity_report(report: IntegrityReport) -> str:
    lines = ["CONDITIONS", *["  %s" % value for value in report.conditions]]
    lines += ["CORRUPTION TYPES", *[
        "  %s" % value for value in report.corruption_types
    ], "", "PER CONDITION:"]
    for condition in report.conditions:
        lines.append("  %s: %s" % (condition, json.dumps(
            report.per_condition.get(condition, {}), sort_keys=True)))
    lines += ["PER CORRUPTION:"]
    lines += ["  %s: %d" % item for item in sorted(report.per_corruption.items())]
    lines += ["PER SEVERITY:"]
    lines += ["  %s: %d" % item for item in sorted(report.per_severity.items())]
    lines += ["SPLIT:"]
    lines += ["  %s: %d" % item for item in sorted(report.split.items())]
    for category in ERROR_CATEGORIES:
        lines.append("%s: %d" % (category.replace("_", " ").upper(), len(report.errors[category])))
        lines.extend("  %s" % value for value in report.errors[category])
    lines.append("GATE = %s" % ("PASS" if report.passed else "FAIL"))
    return "\n".join(lines)


class PairedOccNuScenesDataset(Dataset):
    """Path-only Stage 3-A dataset contract for history/teacher/student input."""

    def __init__(
        self,
        manifest: Union[Path, str, Sequence[Mapping[str, Any]]],
        nuscenes_root: Union[Path, str],
        dirty_root: Union[Path, str],
        require_files: bool = True,
    ) -> None:
        self.records = read_manifest(manifest) if isinstance(manifest, (str, Path)) else [
            dict(record) for record in manifest
        ]
        self.nuscenes_root = Path(nuscenes_root)
        self.dirty_root = Path(dirty_root)
        self.require_files = require_files

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        result = dict(self.records[index])
        for field_name in ("previous_clean_paths", "current_clean_paths"):
            result[field_name] = [
                str(self.nuscenes_root / value) for value in result[field_name]
            ]
        result["current_dirty_paths"] = [
            str(self.dirty_root / value) for value in result["current_dirty_paths"]
        ]
        if self.require_files:
            missing = [
                value for field_name in MANIFEST_PATH_FIELDS
                for value in result[field_name] if not Path(value).is_file()
            ]
            if missing:
                raise FileNotFoundError("missing paired images: %s" % missing)
        return result


__all__ = [
    "ConditionImage", "IntegrityReport", "NuScenesMetadata",
    "PairedOccNuScenesDataset", "audit_pair_records", "build_pair_records",
    "clean_filename_for", "discover_condition_images", "format_integrity_report",
    "manifest_sort_key", "official_mini_splits", "official_splits", "parse_condition",
    "read_manifest", "resolve_clean_filename", "write_manifest",
]
