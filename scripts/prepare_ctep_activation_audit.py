#!/usr/bin/env python3
"""Freeze the deterministic train-scene list and CTEP audit manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import mmcv


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/full_nuscenes/ctep_method_activation"
INFO = ROOT / "data/nuscenes/nuscenes2d_temporal_infos_train.pkl"
PREREG = REPORT / "PRE_REGISTRATION.md"
SEED_TEXT = "ctep-train-scenes-v1-20260902"
SCENE_COUNT = 16
PRESETS = {
    "blur_back": ROOT / "protocols/presets/motion_blur_back_10f_s09.json",
    "crash_back": ROOT / "protocols/presets/camera_crash_back_10f.json",
    "dark_back": ROOT / "protocols/presets/dark_back_10f_s09.json",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    if (REPORT / "scene_list.csv").exists():
        raise RuntimeError("scene_list.csv already exists; refusing to redefine scenes")
    infos = mmcv.load(str(INFO))["infos"]
    by_scene: dict[str, list[tuple[int, dict]]] = {}
    for index, info in enumerate(infos):
        by_scene.setdefault(str(info["scene_token"]), []).append((index, info))
    eligible = []
    for token, entries in by_scene.items():
        frames = {int(info["frame_idx"]) for _, info in entries}
        if set(range(13)).issubset(frames):
            key = hashlib.sha256((SEED_TEXT + token).encode()).hexdigest()
            eligible.append((key, token, entries))
    eligible.sort(key=lambda item: (item[0], item[1]))
    selected = eligible[:SCENE_COUNT]
    rows = []
    for rank, (key, token, entries) in enumerate(selected, 1):
        entries = sorted(entries, key=lambda item: int(item[1]["frame_idx"]))
        first13 = {int(info["frame_idx"]): (index, info) for index, info in entries}
        rows.append({
            "scene_rank": rank,
            "scene_token": token,
            "selection_sha256": key,
            "num_scene_frames": len(entries),
            "first_dataset_index": entries[0][0],
            "last_dataset_index": entries[-1][0],
            "replay_indices_0_12": json.dumps([first13[i][0] for i in range(13)]),
            "sample_tokens_0_12": json.dumps([str(first13[i][1]["token"]) for i in range(13)]),
        })
    path = REPORT / "scene_list.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    manifest = {
        "schema_version": 1,
        "split": "train",
        "info_file": str(INFO.relative_to(ROOT)),
        "info_size": INFO.stat().st_size,
        "selection_seed_text": SEED_TEXT,
        "eligible_scenes": len(eligible),
        "selected_scenes": len(rows),
        "active_frames": [3, 12],
        "scene_list_sha256": digest(path),
        "preregistration_sha256": digest(PREREG),
        "protocols": {
            key: {"path": str(value.relative_to(ROOT)), "sha256": digest(value)}
            for key, value in PRESETS.items()
        },
    }
    (REPORT / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (REPORT / "progress_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "scene_list_sha256": manifest["scene_list_sha256"],
        "p0": {key: {"completed_scenes": [], "rows": 0} for key in PRESETS},
        "p1": {key: {"completed_scenes": [], "rows": 0} for key in PRESETS},
        "status": "PREREGISTERED_P0_PENDING",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (REPORT / "PARTIAL_STATUS.md").write_text(
        "# PARTIAL STATUS\n\n`PREREGISTERED_P0_PENDING`\n\n"
        "No forward has run. Resume with:\n\n"
        "```bash\npython scripts/run_ctep_p0.py --protocol blur_back\n"
        "python scripts/run_ctep_p0.py --protocol crash_back\n"
        "python scripts/run_ctep_p0.py --protocol dark_back\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
