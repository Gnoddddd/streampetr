#!/usr/bin/env python3
"""One-pass full-nuScenes metadata, split, info, and blob preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import mmcv
from nuscenes.utils.splits import create_splits_scenes


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/nuscenes"
VERSION = DATA / "v1.0-trainval"
REPORT = ROOT / "reports/full_nuscenes/mechanism_confirmation/preflight"
EXPECTED_TABLE_COUNTS = {
    "scene": 850,
    "sample": 34149,
    "sample_data": 2631083,
    "sample_annotation": 1166187,
    "instance": 64386,
    "ego_pose": 2631083,
    "calibrated_sensor": 10200,
    "sensor": 12,
    "category": 23,
    "attribute": 8,
    "visibility": 4,
    "log": 68,
    "map": 4,
}
EXPECTED_INFO_SHA256 = {
    "train": "dc5e5e611badbdb1c0270a3583e022cf14a9af7b3ff8f02370434b8ec50b493d",
    "val": "a0de2828174eaa46052c416181e3019ab4e1e5e007f572b93a10cc16d97913ac",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-all-blobs",
        action="store_true",
        help="Compare every sample/sweep/map filename to sample_data/map metadata.",
    )
    return parser.parse_args()


def load_json(name: str):
    with (VERSION / f"{name}.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_blobs(sample_data: list[dict], maps: list[dict]) -> list[dict]:
    expected = defaultdict(set)
    duplicates = 0
    for source in (sample_data, maps):
        for row in source:
            relative = Path(row["filename"])
            parent = str(relative.parent)
            before = len(expected[parent])
            expected[parent].add(relative.name)
            duplicates += len(expected[parent]) == before
    rows = []
    for relative_parent in sorted(expected):
        directory = DATA / relative_parent
        if not directory.is_dir():
            actual = set()
        else:
            # These nuScenes leaves contain files only. Avoid 2.6M extra stat calls
            # on a mounted filesystem; filename equality is the integrity check.
            actual = {entry.name for entry in os.scandir(directory)}
        missing = expected[relative_parent] - actual
        extra = actual - expected[relative_parent]
        rows.append({
            "directory": relative_parent,
            "expected_files": len(expected[relative_parent]),
            "actual_files": len(actual),
            "missing_files": len(missing),
            "extra_files": len(extra),
            "exact": not missing and not extra,
            "first_missing": sorted(missing)[:1][0] if missing else "",
            "first_extra": sorted(extra)[:1][0] if extra else "",
        })
    rows.append({
        "directory": "TOTAL",
        "expected_files": sum(row["expected_files"] for row in rows),
        "actual_files": sum(row["actual_files"] for row in rows),
        "missing_files": sum(row["missing_files"] for row in rows),
        "extra_files": sum(row["extra_files"] for row in rows),
        # Official trainval metadata contains 10 cross-scene radar filename
        # references that deliberately point to an identical boundary capture.
        # Integrity is exact when the unique filename set matches on disk;
        # retain the duplicate-reference count as an explicit diagnostic.
        "exact": all(row["exact"] for row in rows),
        "first_missing": "",
        "first_extra": "",
        "duplicate_metadata_filenames": duplicates,
    })
    return rows


def info_summary(
    split: str,
    expected_scenes: set[str],
    scene_by_token: dict,
    check_referenced_paths: bool,
) -> dict:
    value = mmcv.load(str(DATA / f"nuscenes2d_temporal_infos_{split}.pkl"))
    infos = sorted(value["infos"], key=lambda row: row["timestamp"])
    seen, transitions, reentries, boundary_errors = set(), [], [], []
    tokens = set()
    previous_scene = None
    previous_frame = None
    referenced_missing = []
    for index, row in enumerate(infos):
        tokens.add(str(row["token"]))
        scene = str(row["scene_token"])
        frame = int(row["frame_idx"])
        if scene != previous_scene:
            transitions.append(scene)
            if scene in seen:
                reentries.append((index, scene, frame))
            seen.add(scene)
            if frame != 0:
                boundary_errors.append((index, scene, frame, "new_scene_not_zero"))
        elif frame != previous_frame + 1:
            boundary_errors.append((index, scene, frame, f"after_{previous_frame}"))
        previous_scene, previous_frame = scene, frame
        if check_referenced_paths:
            paths = [row["lidar_path"]] + [cam["data_path"] for cam in row["cams"].values()]
            paths += [sweep["data_path"] for sweep in row.get("sweeps", [])]
            for path in paths:
                normalized = str(path)
                if normalized.startswith("./"):
                    normalized = normalized[2:]
                resolved = ROOT / normalized
                if not resolved.is_file():
                    referenced_missing.append(str(path))
    scene_names = {scene_by_token[token] for token in seen}
    return {
        "split": split,
        "frames": len(infos),
        "unique_sample_tokens": len(tokens),
        "scenes": len(seen),
        "scene_transitions": len(transitions),
        "scene_reentries": len(reentries),
        "frame_boundary_errors": len(boundary_errors),
        "split_scene_missing": len(expected_scenes - scene_names),
        "split_scene_extra": len(scene_names - expected_scenes),
        "referenced_path_missing": len(referenced_missing),
        "metadata_version": value["metadata"].get("version"),
        "passed": (
            len(reentries) == 0
            and len(boundary_errors) == 0
            and scene_names == expected_scenes
            and not referenced_missing
        ),
    }


def main() -> None:
    args = parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    required = [VERSION, DATA / "samples", DATA / "sweeps", DATA / "maps"]
    required_ok = all(path.is_dir() for path in required)
    tables = {}
    loaded = {}
    for name, expected in EXPECTED_TABLE_COUNTS.items():
        rows = load_json(name)
        if name in {"scene", "sample_data", "map"}:
            loaded[name] = rows
        tables[name] = {"actual": len(rows), "expected": expected,
                        "passed": len(rows) == expected}
        print(f"table {name}: {len(rows)}/{expected}", flush=True)
    scenes = loaded["scene"]
    scene_by_token = {row["token"]: row["name"] for row in scenes}
    split_names = create_splits_scenes()
    train_names, val_names = set(split_names["train"]), set(split_names["val"])
    info_rows = [
        info_summary("train", train_names, scene_by_token, not args.verify_all_blobs),
        info_summary("val", val_names, scene_by_token, not args.verify_all_blobs),
    ]
    info_hashes = {}
    for split, expected_hash in EXPECTED_INFO_SHA256.items():
        path = DATA / f"nuscenes2d_temporal_infos_{split}.pkl"
        actual_hash = sha256(path)
        info_hashes[split] = {
            "path": str(path), "sha256": actual_hash,
            "expected_sha256": expected_hash, "passed": actual_hash == expected_hash,
        }
    with (REPORT / "info_split.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(info_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(info_rows)

    blob_rows = []
    if args.verify_all_blobs:
        blob_rows = inventory_blobs(loaded["sample_data"], loaded["map"])
        with (REPORT / "blob_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = []
            for row in blob_rows:
                for key in row:
                    if key not in fields:
                        fields.append(key)
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(blob_rows)
    detector = ROOT / "repos/StreamPETR/projects/mmdet3d_plugin/models/detectors/petr3d.py"
    detector_text = detector.read_text(encoding="utf-8")
    memory_reset_ok = all(text in detector_text for text in (
        "if img_metas[0]['scene_token'] != self.prev_scene_token:",
        "data['prev_exists'] = data['img'].new_zeros(1)",
        "self.pts_bbox_head.reset_memory()",
    ))
    result = {
        "data_root": str(DATA.resolve()),
        "required_directories_present": required_ok,
        "table_counts": tables,
        "train_val_scene_overlap": len(train_names & val_names),
        "info_splits": info_rows,
        "official_info_sha256": info_hashes,
        "scene_memory_reset_static_check": memory_reset_ok,
        "all_blob_filenames_checked": bool(args.verify_all_blobs),
        "blob_inventory_exact": bool(blob_rows and blob_rows[-1]["exact"]),
    }
    result["passed"] = bool(
        required_ok
        and all(value["passed"] for value in tables.values())
        and not result["train_val_scene_overlap"]
        and all(row["passed"] for row in info_rows)
        and all(value["passed"] for value in info_hashes.values())
        and memory_reset_ok
        and (not args.verify_all_blobs or result["blob_inventory_exact"])
    )
    (REPORT / "data_preflight.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
