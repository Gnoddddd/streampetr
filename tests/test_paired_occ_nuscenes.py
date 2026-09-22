import copy
import json
import sys
import types
from pathlib import Path

import pytest

from datasets.paired_occ_nuscenes import (
    ConditionImage,
    NuScenesMetadata,
    PairedOccNuScenesDataset,
    audit_pair_records,
    build_pair_records,
    clean_filename_for,
    discover_condition_images,
    official_splits,
    parse_condition,
    read_manifest,
    resolve_clean_filename,
    write_manifest,
)
from datasets.temporal_occ_nuscenes import NUSCENES_CAMERA_ORDER


SPLITS = {
    "mini_train": ("scene-train",),
    "mini_val": ("scene-val",),
    "train": ("scene-train",),
    "val": ("scene-val",),
}

FULL_SPLITS = {
    "mini_train": ("scene-full-train", "scene-mini-train-full-val"),
    "mini_val": ("scene-full-val",),
    "train": ("scene-full-train",),
    "val": ("scene-mini-train-full-val", "scene-full-val"),
}


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _synthetic_dataset(
    tmp_path: Path,
    conditions=("Dirt/0.1_dirt",),
    scene_names=("scene-train",),
):
    clean_root = tmp_path / "nuscenes"
    dirty_root = tmp_path / "dirty"
    sample_data = []
    samples = []
    scenes = []
    calibrated_sensors = []
    sensors = []
    for camera in NUSCENES_CAMERA_ORDER:
        sensor_token = "sensor-%s" % camera
        calibrated_token = "calibrated-%s" % camera
        sensors.append({
            "token": sensor_token,
            "channel": camera,
            "modality": "camera",
        })
        calibrated_sensors.append({
            "token": calibrated_token,
            "sensor_token": sensor_token,
        })
    for scene_index, scene_name in enumerate(scene_names):
        scene_token = "scene-token-%d" % scene_index
        scenes.append({"token": scene_token, "name": scene_name})
        previous_token = ""
        for frame_index in range(2):
            sample_token = "sample-%d-%d" % (scene_index, frame_index)
            timestamp = (scene_index * 10 + frame_index + 1) * 1_000_000
            for camera_index, camera in enumerate(NUSCENES_CAMERA_ORDER):
                filename = "%s-%d-%d.jpg" % (camera.lower(), scene_index, frame_index)
                relative = "samples/%s/%s" % (camera, filename)
                token = "sd-%d-%d-%d" % (scene_index, frame_index, camera_index)
                sample_data.append({
                    "token": token,
                    "sample_token": sample_token,
                    "timestamp": timestamp + camera_index,
                    "filename": relative,
                    "calibrated_sensor_token": "calibrated-%s" % camera,
                    "ego_pose_token": "pose-%s" % token,
                    "is_key_frame": True,
                })
                _write(clean_root / relative)
                for condition in conditions:
                    _write(dirty_root / condition / camera / filename)
            samples.append({
                "token": sample_token,
                "scene_token": scene_token,
                "timestamp": timestamp,
                "prev": previous_token,
                "next": "sample-%d-%d" % (scene_index, frame_index + 1)
                if frame_index == 0 else "",
            })
            previous_token = sample_token
    metadata_dir = clean_root / "v1.0-trainval"
    metadata_dir.mkdir(parents=True)
    for name, rows in (
        ("sample_data", sample_data),
        ("sample", samples),
        ("scene", scenes),
        ("calibrated_sensor", calibrated_sensors),
        ("sensor", sensors),
    ):
        (metadata_dir / (name + ".json")).write_text(
            json.dumps(rows), encoding="utf-8"
        )
    return clean_root, dirty_root, NuScenesMetadata.from_directory(metadata_dir)


def _build(clean_root, dirty_root, metadata, **kwargs):
    split_scene_names = kwargs.pop("split_scene_names", SPLITS)
    return build_pair_records(
        clean_root,
        dirty_root,
        metadata,
        split_scene_names=split_scene_names,
        **kwargs,
    )


def _sample_data_row(token, filename):
    return {
        "token": token,
        "sample_token": "sample",
        "timestamp": 1,
        "filename": filename,
        "calibrated_sensor_token": "cal-%s" % token,
        "ego_pose_token": "pose-%s" % token,
        "is_key_frame": True,
    }


def _condition_image(filename, camera="CAM_FRONT"):
    relative = "Dirt/0.1_dirt/%s/%s" % (camera, filename)
    return ConditionImage(
        path=Path(relative),
        relative_path=relative,
        raw_condition="Dirt/0.1_dirt",
        corruption_type="Dirt",
        severity=0.1,
        camera=camera,
        filename=filename,
    )


def test_clean_filename_resolver_prefers_the_only_exact_match():
    row = _sample_data_row("camera-token", "samples/CAM_FRONT/foo.jpg")
    metadata = NuScenesMetadata([row], [], [])

    assert resolve_clean_filename(_condition_image("foo.jpg"), metadata) == row["filename"]


def test_clean_filename_resolver_strips_one_terminal_obstructed_suffix():
    row = _sample_data_row("camera-token", "samples/CAM_FRONT/foo.jpg")
    metadata = NuScenesMetadata([row], [], [])

    assert (
        resolve_clean_filename(_condition_image("foo_obstructed.jpg"), metadata)
        == row["filename"]
    )


def test_clean_filename_resolver_does_not_replace_nonterminal_obstructed():
    row = _sample_data_row("camera-token", "samples/CAM_FRONT/foo_extra.jpg")
    metadata = NuScenesMetadata([row], [], [])

    with pytest.raises(LookupError, match="no Camera sample_data"):
        resolve_clean_filename(_condition_image("foo_obstructed_extra.jpg"), metadata)


def test_clean_filename_resolver_rejects_exact_normalized_ambiguity():
    exact = _sample_data_row(
        "exact-camera-token", "samples/CAM_FRONT/foo_obstructed.jpg"
    )
    normalized = _sample_data_row(
        "normalized-camera-token", "samples/CAM_FRONT/foo.jpg"
    )
    metadata = NuScenesMetadata([exact, normalized], [], [])

    with pytest.raises(ValueError, match="ambiguous Camera sample_data"):
        resolve_clean_filename(_condition_image("foo_obstructed.jpg"), metadata)


def test_radar_filename_duplicate_does_not_block_camera_filename_index():
    camera = _sample_data_row("camera-token", "samples/CAM_FRONT/cam.jpg")
    uppercase_camera = _sample_data_row(
        "uppercase-camera-token", "samples/CAM_BACK_RIGHT/cam.PNG"
    )
    radar_filename = "samples/RADAR_FRONT_RIGHT/radar.pcd"
    radar_a = _sample_data_row("radar-token-a", radar_filename)
    radar_b = _sample_data_row("radar-token-b", radar_filename)
    lidar = _sample_data_row("lidar-token", "samples/LIDAR_TOP/lidar.pcd.bin")
    sweep_camera = _sample_data_row(
        "sweep-camera-token", "sweeps/CAM_FRONT/sweep.jpg"
    )

    metadata = NuScenesMetadata(
        [camera, uppercase_camera, radar_a, radar_b, lidar, sweep_camera], [], []
    )

    assert metadata.sample_data_by_filename[camera["filename"]] is camera
    assert metadata.sample_data_by_filename[uppercase_camera["filename"]] is uppercase_camera
    assert radar_filename not in metadata.sample_data_by_filename
    assert lidar["filename"] not in metadata.sample_data_by_filename
    assert sweep_camera["filename"] not in metadata.sample_data_by_filename
    assert metadata.sample_data_by_token["radar-token-a"] is radar_a
    assert metadata.sample_data_by_token["radar-token-b"] is radar_b


def test_camera_filename_duplicate_remains_a_hard_failure():
    filename = "samples/CAM_FRONT/cam.jpg"
    camera_a = _sample_data_row("camera-token-a", filename)
    camera_b = _sample_data_row("camera-token-b", filename)

    with pytest.raises(ValueError, match="duplicate sample camera filename"):
        NuScenesMetadata([camera_a, camera_b], [], [])


def test_duplicate_keyframe_camera_channel_is_a_hard_failure():
    rows = [
        _sample_data_row("camera-token-a", "samples/CAM_FRONT/a.jpg"),
        _sample_data_row("camera-token-b", "samples/CAM_FRONT/b.jpg"),
    ]
    for row in rows:
        row["sample_token"] = "same-sample"
        row["calibrated_sensor_token"] = "calibrated-front"
    calibrated = [{"token": "calibrated-front", "sensor_token": "sensor-front"}]
    sensors = [{
        "token": "sensor-front", "channel": "CAM_FRONT", "modality": "camera"
    }]

    with pytest.raises(ValueError, match="duplicate keyframe sample camera channel"):
        NuScenesMetadata(rows, [], [], calibrated, sensors)


def _build_single_dirty_camera(tmp_path, sensor_channel, is_key_frame):
    clean_root = tmp_path / "nuscenes"
    dirty_root = tmp_path / "dirty"
    clean_filename = "samples/CAM_FRONT/foo.jpg"
    _write(clean_root / clean_filename)
    _write(dirty_root / "Dirt/0.1_dirt/CAM_FRONT/foo.jpg")
    row = _sample_data_row("camera-token", clean_filename)
    row["is_key_frame"] = is_key_frame
    row["calibrated_sensor_token"] = "calibrated-camera"
    sample = {
        "token": "sample",
        "timestamp": 1,
        "scene_token": "scene-token",
        "prev": "",
        "next": "",
    }
    metadata = NuScenesMetadata(
        [row],
        [sample],
        [{"token": "scene-token", "name": "scene-train"}],
        [{"token": "calibrated-camera", "sensor_token": "sensor-camera"}],
        [{
            "token": "sensor-camera",
            "channel": sensor_channel,
            "modality": "camera",
        }],
    )
    return _build(clean_root, dirty_root, metadata)


def test_dirty_camera_rejects_wrong_calibrated_sensor_channel(tmp_path):
    records, report = _build_single_dirty_camera(tmp_path, "CAM_BACK", True)

    assert records == []
    assert report.errors["camera_errors"]


def test_dirty_camera_rejects_non_keyframe_sample_data(tmp_path):
    records, report = _build_single_dirty_camera(tmp_path, "CAM_FRONT", False)

    assert records == []
    assert report.errors["camera_errors"]


def test_nested_and_direct_condition_discovery_and_parsing(tmp_path):
    nested = tmp_path / "Dirt/0.1_dirt/CAM_FRONT/a.JPG"
    typo_is_real = tmp_path / "Water-blur/0.3_water-blu/CAM_BACK/b.png"
    direct = tmp_path / "Scratches/CAM_FRONT/c.jpeg"
    for path in (nested, typo_is_real, direct):
        _write(path)
    images = discover_condition_images(tmp_path)
    assert [(item.raw_condition, item.corruption_type, item.severity) for item in images] == [
        ("Dirt/0.1_dirt", "Dirt", 0.1),
        ("Scratches", "Scratches", None),
        ("Water-blur/0.3_water-blu", "Water-blur", 0.3),
    ]
    assert parse_condition("Dirt/0.1_dirt") == ("Dirt", 0.1)
    assert parse_condition("Water-blur/0.3_water-blu") == ("Water-blur", 0.3)


def test_dirty_to_clean_mapping_and_metadata_lookup(tmp_path):
    clean_root, dirty_root, metadata = _synthetic_dataset(tmp_path)
    image = discover_condition_images(dirty_root)[0]
    clean_filename = clean_filename_for(image)
    assert clean_filename.startswith("samples/%s/" % image.camera)
    sample_data = metadata.sample_data_by_filename[clean_filename]
    assert metadata.samples_by_token[sample_data["sample_token"]]["scene_token"]


def test_legal_pair_has_fixed_camera_order_and_stage3_contract(tmp_path):
    clean_root, dirty_root, metadata = _synthetic_dataset(tmp_path)
    records, report = _build(clean_root, dirty_root, metadata)
    assert report.passed
    assert len(records) == 1
    record = records[0]
    assert record["previous_sample_token"] == "sample-0-0"
    assert record["current_sample_token"] == "sample-0-1"
    assert record["dt"] == 1.0
    assert record["camera_names"] == list(NUSCENES_CAMERA_ORDER)
    assert len(record["previous_clean_paths"]) == 6
    assert len(record["current_clean_paths"]) == 6
    assert len(record["current_dirty_paths"]) == 6
    assert "data" not in metadata.samples_by_token["sample-0-0"]
    assert "data" not in metadata.samples_by_token["sample-0-1"]
    assert [
        metadata.sample_camera_data[("sample-0-0", camera)]["filename"]
        for camera in NUSCENES_CAMERA_ORDER
    ] == record["previous_clean_paths"]
    assert [
        metadata.sample_camera_data[("sample-0-1", camera)]["filename"]
        for camera in NUSCENES_CAMERA_ORDER
    ] == record["current_clean_paths"]
    item = PairedOccNuScenesDataset(
        records, clean_root, dirty_root, require_files=True
    )[0]
    assert all(Path(value).is_absolute() for value in item["previous_clean_paths"])
    assert item["raw_condition"] == "Dirt/0.1_dirt"


def test_obstructed_dirty_names_build_and_audit_against_clean_names(tmp_path):
    clean_root, dirty_root, metadata = _synthetic_dataset(tmp_path)
    for path in sorted((dirty_root / "Dirt/0.1_dirt").rglob("*.jpg")):
        path.rename(path.with_name(path.stem + "_obstructed" + path.suffix))

    records, build_report = _build(clean_root, dirty_root, metadata)
    audit_report = audit_pair_records(
        records, clean_root, dirty_root, metadata, SPLITS
    )

    assert len(records) == 1
    assert build_report.passed
    assert audit_report.passed
    assert all(
        Path(path).stem.endswith("_obstructed")
        for path in records[0]["current_dirty_paths"]
    )
    assert all(
        not Path(path).stem.endswith("_obstructed")
        for path in records[0]["current_clean_paths"]
    )


def test_scene_boundary_is_rejected(tmp_path):
    clean_root, dirty_root, metadata = _synthetic_dataset(
        tmp_path, scene_names=("scene-train", "scene-val")
    )
    current = metadata.samples_by_token["sample-1-1"]
    current["prev"] = "sample-0-1"
    records, report = _build(clean_root, dirty_root, metadata)
    assert [row["scene_name"] for row in records] == ["scene-train"]
    assert report.errors["temporal_errors"]


def test_incomplete_six_camera_sample_cannot_form_pair(tmp_path):
    clean_root, dirty_root, metadata = _synthetic_dataset(tmp_path)
    missing = next((dirty_root / "Dirt/0.1_dirt/CAM_FRONT").glob("*0-1.jpg"))
    missing.unlink()
    records, report = _build(clean_root, dirty_root, metadata)
    assert records == []
    assert report.per_condition["Dirt/0.1_dirt"]["six_camera_complete_samples"] == 1


def test_audit_rejects_duplicate_and_camera_order_drift(tmp_path):
    clean_root, dirty_root, metadata = _synthetic_dataset(tmp_path)
    records, _ = _build(clean_root, dirty_root, metadata)
    damaged = copy.deepcopy(records)
    damaged[0]["camera_names"][0:2] = reversed(damaged[0]["camera_names"][0:2])
    damaged.append(copy.deepcopy(damaged[0]))
    report = audit_pair_records(
        damaged, clean_root, dirty_root, metadata, SPLITS
    )
    assert len(report.errors["duplicates"]) == 1
    assert report.errors["camera_errors"]
    assert not report.passed


def test_official_scene_level_split_isolation(tmp_path):
    clean_root, dirty_root, metadata = _synthetic_dataset(
        tmp_path, scene_names=("scene-train", "scene-val")
    )
    train, _ = _build(clean_root, dirty_root, metadata, split="mini_train")
    val, _ = _build(clean_root, dirty_root, metadata, split="mini_val")
    assert {row["scene_name"] for row in train} == {"scene-train"}
    assert {row["scene_name"] for row in val} == {"scene-val"}
    assert {row["scene_token"] for row in train}.isdisjoint(
        {row["scene_token"] for row in val}
    )


def test_official_splits_loads_train_and_val_from_devkit(monkeypatch):
    mock_splits = {
        "mini_train": ["mini-train"],
        "mini_val": ["mini-val"],
        "train": ["full-train"],
        "val": ["full-val"],
    }
    package = types.ModuleType("nuscenes")
    utils = types.ModuleType("nuscenes.utils")
    splits_module = types.ModuleType("nuscenes.utils.splits")
    splits_module.create_splits_scenes = lambda: mock_splits
    monkeypatch.setitem(sys.modules, "nuscenes", package)
    monkeypatch.setitem(sys.modules, "nuscenes.utils", utils)
    monkeypatch.setitem(sys.modules, "nuscenes.utils.splits", splits_module)

    assert official_splits() == {
        name: tuple(mock_splits[name])
        for name in ("mini_train", "mini_val", "train", "val")
    }


def test_full_train_val_filtering_conservation_and_mini_compatibility(tmp_path):
    scene_names = (
        "scene-full-train", "scene-mini-train-full-val", "scene-full-val",
    )
    clean_root, dirty_root, metadata = _synthetic_dataset(
        tmp_path, scene_names=scene_names
    )
    train, train_report = _build(
        clean_root, dirty_root, metadata, split="train", split_scene_names=FULL_SPLITS
    )
    val, val_report = _build(
        clean_root, dirty_root, metadata, split="val", split_scene_names=FULL_SPLITS
    )
    mini_train, _ = _build(
        clean_root, dirty_root, metadata, split="mini_train",
        split_scene_names=FULL_SPLITS,
    )
    mini_val, _ = _build(
        clean_root, dirty_root, metadata, split="mini_val",
        split_scene_names=FULL_SPLITS,
    )
    all_records, _ = _build(
        clean_root, dirty_root, metadata, split="all", split_scene_names=FULL_SPLITS
    )

    assert train_report.passed and val_report.passed
    assert {row["scene_name"] for row in train} == {"scene-full-train"}
    assert {row["scene_name"] for row in val} == {
        "scene-mini-train-full-val", "scene-full-val"
    }
    assert {row["scene_token"] for row in train}.isdisjoint(
        {row["scene_token"] for row in val}
    )
    assert {row["scene_name"] for row in mini_train} == {
        "scene-full-train", "scene-mini-train-full-val"
    }
    assert {row["scene_name"] for row in mini_val} == {"scene-full-val"}
    assert len(train) + len(val) == len(all_records)


def test_mini_train_scene_in_full_val_is_excluded_from_train_and_kept_in_val(tmp_path):
    clean_root, dirty_root, metadata = _synthetic_dataset(
        tmp_path, scene_names=("scene-mini-train-full-val",)
    )
    train, _ = _build(
        clean_root, dirty_root, metadata, split="train", split_scene_names=FULL_SPLITS
    )
    val, report = _build(
        clean_root, dirty_root, metadata, split="val", split_scene_names=FULL_SPLITS
    )

    assert train == []
    assert report.passed
    assert [row["scene_name"] for row in val] == ["scene-mini-train-full-val"]
    assert val[0]["split"] == "val"


@pytest.mark.parametrize(
    ("field", "root_name", "category"),
    (
        ("current_dirty_paths", "dirty", "missing_dirty"),
        ("current_clean_paths", "clean", "missing_clean"),
    ),
)
def test_audit_detects_missing_images(tmp_path, field, root_name, category):
    clean_root, dirty_root, metadata = _synthetic_dataset(tmp_path)
    records, _ = _build(clean_root, dirty_root, metadata)
    root = dirty_root if root_name == "dirty" else clean_root
    (root / records[0][field][0]).unlink()
    report = audit_pair_records(records, clean_root, dirty_root, metadata, SPLITS)
    assert report.errors[category]


def test_builder_reports_missing_metadata(tmp_path):
    clean_root, dirty_root, metadata = _synthetic_dataset(tmp_path)
    _write(dirty_root / "Dirt/0.1_dirt/CAM_FRONT/not-in-metadata.jpg")
    _, report = _build(clean_root, dirty_root, metadata)
    assert report.errors["missing_metadata"]


def test_manifest_is_byte_deterministic_and_sorted(tmp_path):
    conditions = ("Water-blur/0.3_water-blu", "Dirt/0.1_dirt")
    clean_root, dirty_root, metadata = _synthetic_dataset(tmp_path, conditions)
    records, _ = _build(clean_root, dirty_root, metadata)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_manifest(first, list(reversed(records)))
    write_manifest(second, records)
    assert first.read_bytes() == second.read_bytes()
    assert [row["corruption_type"] for row in read_manifest(first)] == [
        "Dirt", "Water-blur"
    ]


def test_corruption_type_and_exact_condition_filtering(tmp_path):
    conditions = (
        "Dirt/0.1_dirt",
        "Dirt/0.2_dirt",
        "Water-blur/0.3_water-blu",
        "Scratches",
    )
    clean_root, dirty_root, metadata = _synthetic_dataset(tmp_path, conditions)
    records, _ = _build(
        clean_root,
        dirty_root,
        metadata,
        corruption_types=("Dirt", "Water-blur"),
        conditions=("Dirt/0.2_dirt", "Water-blur/0.3_water-blu"),
    )
    assert {row["raw_condition"] for row in records} == {
        "Dirt/0.2_dirt", "Water-blur/0.3_water-blu"
    }
    assert {row["corruption_type"] for row in records} == {"Dirt", "Water-blur"}


def test_split_leakage_and_manifest_metadata_mismatch_fail_gate(tmp_path):
    clean_root, dirty_root, metadata = _synthetic_dataset(tmp_path)
    records, _ = _build(clean_root, dirty_root, metadata)
    damaged = copy.deepcopy(records)
    damaged[0]["current_sample_data_tokens"][0] = "not-a-token"
    overlapping = {"mini_train": ("scene-train",), "mini_val": ("scene-train",)}
    report = audit_pair_records(
        damaged, clean_root, dirty_root, metadata, overlapping
    )
    assert report.errors["missing_metadata"]
    assert report.errors["split_leakage"]
    assert not report.passed
