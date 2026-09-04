#!/usr/bin/env python3
"""Freeze scene and feature manifests before prospective extraction."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import mmcv
import pandas as pd
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import train as official_train


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/full_nuscenes/prospective_failure_decodability"
INFO = ROOT / "data/nuscenes/nuscenes2d_temporal_infos_train.pkl"
DATA = ROOT / "data/nuscenes"
DISCOVERY = ROOT / "reports/full_nuscenes/ctep_method_activation/scene_list.csv"
CONFIG = ROOT / "configs/full_nuscenes/stream_petr_r50_90e_ctep_train_audit.py"
CHECKPOINT = ROOT / "checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth"
PROTOCOLS = {
    "blur_back": ROOT / "protocols/presets/motion_blur_back_10f_s09.json",
    "crash_back": ROOT / "protocols/presets/camera_crash_back_10f.json",
    "dark_back": ROOT / "protocols/presets/dark_back_10f_s09.json",
}
EXPECTED = {
    INFO: "dc5e5e611badbdb1c0270a3583e022cf14a9af7b3ff8f02370434b8ec50b493d",
    CONFIG: "927ba2518a4ca460d2f7f6b3ba74dab620ac8e2995ee7e9aadbcbebf2d7c64a6",
    CHECKPOINT: "e6323ae5c31adf1eedd46d6dd4fd3c73d95aa26f18cc8aa23c196494b7de3451",
    DISCOVERY: "7bbd389f8ec1f02e75d3d8a7feb773f109fdf24d5824b65db89fc7544e425fb3",
}
SALT = "prospective_failure_decodability:v1:"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict]):
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def split_for(token: str) -> tuple[str, str, int]:
    digest = hashlib.sha256((SALT + token).encode()).hexdigest()
    bucket = int(digest[:16], 16) % 10000
    split = "probe_train" if bucket < 6000 else ("probe_val" if bucket < 8000 else "probe_test")
    return split, digest, bucket


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    observed = {str(path.relative_to(ROOT)): sha256(path) for path in EXPECTED}
    if observed != {str(path.relative_to(ROOT)): digest for path, digest in EXPECTED.items()}:
        raise RuntimeError(f"immutable source mismatch: {observed}")
    info_object = mmcv.load(str(INFO))
    infos = info_object["infos"] if isinstance(info_object, dict) else info_object
    by_scene = defaultdict(list)
    for index, info in enumerate(infos):
        by_scene[str(info["scene_token"])].append((int(info["frame_idx"]), index, str(info["token"])))
    if len(infos) != 28130 or len(by_scene) != 700:
        raise RuntimeError("full train info coverage changed")
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(DATA), verbose=False)
    names = {str(scene["name"]): str(scene["token"]) for scene in nusc.scene}
    official_tokens = {names[name] for name in official_train}
    if official_tokens != set(by_scene):
        raise RuntimeError("info scenes differ from official train split")
    discovery = set(pd.read_csv(DISCOVERY).scene_token.astype(str))
    if len(discovery) != 16 or not discovery <= official_tokens:
        raise RuntimeError("discovery exclusion set changed")
    token_to_name = {token: name for name, token in names.items()}
    rows = []
    for token in sorted(official_tokens - discovery):
        ordered = sorted(by_scene[token])
        if len(ordered) < 13 or [item[0] for item in ordered[:13]] != list(range(13)):
            raise RuntimeError(f"scene lacks ordered frames 0..12: {token}")
        split, digest, bucket = split_for(token)
        rows.append({
            "scene_token": token, "scene_name": token_to_name[token], "split": split,
            "split_sha256": digest, "split_bucket": bucket, "num_scene_frames": len(ordered),
            "dataset_indices_0_12": json.dumps([item[1] for item in ordered[:13]], separators=(",", ":")),
            "sample_tokens_0_12": json.dumps([item[2] for item in ordered[:13]], separators=(",", ":")),
        })
    counts = Counter(row["split"] for row in rows)
    if len(rows) != 684 or counts != {"probe_train": 419, "probe_val": 133, "probe_test": 132}:
        raise RuntimeError(f"frozen hash split changed: {counts}")
    scene_path = REPORT / "frozen_scene_manifest.csv"
    if scene_path.exists():
        existing = pd.read_csv(scene_path)
        candidate = pd.DataFrame(rows)
        if not existing.equals(candidate):
            raise RuntimeError("existing scene manifest differs")
    else:
        atomic_csv(scene_path, rows)
    features = [
        ("observable", "prediction_score", 1), ("observable", "query_top1_score", 1),
        ("observable", "query_top1_top2_margin", 1), ("observable", "deployment_flat_rank", 1),
        ("observable", "prediction_class_one_hot", 10), ("observable", "box_width_m", 1),
        ("observable", "box_length_m", 1), ("observable", "box_height_m", 1),
        ("observable", "box_radial_distance_m", 1), ("observable", "box_speed_mps", 1),
        ("observable", "query_is_propagated", 1), ("observable", "query_source_age_seconds", 1),
        ("representation", "temporal_alignment_query_state", 512),
        ("representation", "decoder_layer5_temporal_self_attn_output", 256),
        ("representation", "final_decoder_pre_cls_query", 256),
    ]
    feature_rows, offsets = [], defaultdict(int)
    for group, name, dimension in features:
        offset = offsets[group]
        feature_rows.append({"group": group, "feature": name, "dimension": dimension,
                             "start_offset": offset, "stop_offset": offset + dimension})
        offsets[group] += dimension
    feature_path = REPORT / "feature_manifest.csv"
    feature_frame = pd.DataFrame(feature_rows)
    if feature_path.exists():
        if not pd.read_csv(feature_path).equals(feature_frame):
            raise RuntimeError("existing feature manifest differs")
    else:
        atomic_csv(feature_path, feature_rows)
    validation = {
        "status": "VALIDATED_BEFORE_FORWARD", "source_hashes": observed,
        "preregistration_sha256": sha256(REPORT / "PRE_REGISTRATION.md"),
        "scene_manifest_sha256": sha256(scene_path), "feature_manifest_sha256": sha256(feature_path),
        "scene_counts": dict(counts), "excluded_discovery_scenes": len(discovery),
        "protocol_sha256": {key: sha256(path) for key, path in PROTOCOLS.items()},
        "stream_petr_commit": "95f64702306ccdb7a78889578b2a55b5deb35b2a",
    }
    atomic_json(REPORT / "source_validation.json", validation)
    initial = {"schema_version": 1, "status": "PREPARED_P0_EXTRACTION_PENDING",
               "scene_manifest_sha256": validation["scene_manifest_sha256"],
               "feature_manifest_sha256": validation["feature_manifest_sha256"],
               "stages": {"P0_extraction": {}, "P0_probe": "LOCKED",
                          "P1": "LOCKED_PENDING_P0", "P2": "LOCKED_PENDING_P0_P1",
                          "P3": "LOCKED_PENDING_P0_P2"},
               "detector_training": "PROHIBITED", "repository_modification": "PROHIBITED"}
    progress_path = REPORT / "progress_manifest.json"
    if progress_path.exists():
        current = json.loads(progress_path.read_text())
        for key in ("scene_manifest_sha256", "feature_manifest_sha256"):
            if current.get(key) != initial[key]:
                raise RuntimeError(f"resume mismatch: {key}")
    else:
        atomic_json(progress_path, initial)
    (REPORT / "PARTIAL_STATUS.md").write_text(
        "# PARTIAL STATUS\n\n`PREPARED_P0_EXTRACTION_PENDING`\n\n"
        "The outcome-blind scene split, feature fields, labels, probe hyperparameters and gates are frozen. "
        "No new trajectory forward has run; no Go/No-Go is permitted.\n\nResume:\n\n```bash\n"
        "python scripts/run_prospective_failure_features.py\n"
        "python scripts/analyze_prospective_failure_decodability.py\n```\n")
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
