#!/usr/bin/env python3
"""Freeze and validate the prospective stress-reserve audit before forward."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/full_nuscenes/fault_stress_reserve_audit"
SOURCE = ROOT / "reports/full_nuscenes/temporal_utility_audit"
FORMAL = {
    "blur_back": ROOT / "protocols/presets/motion_blur_back_10f_s09.json",
    "crash_back": ROOT / "protocols/presets/camera_crash_back_10f.json",
    "dark_back": ROOT / "protocols/presets/dark_back_10f_s09.json",
}
PROBES = {
    "blur_back": REPORT / "probes/cam_back_blur_09_frame2.json",
    "crash_back": REPORT / "probes/cam_back_crash_frame2.json",
    "dark_back": REPORT / "probes/cam_back_dark_09_frame2.json",
}
EXPECTED = {
    SOURCE / "cohort_manifest.csv": "243fc78212bb5a2884b84e933c5a5529d9ef40d7cdab1509c42ac3243ec8f31a",
    SOURCE / "trajectory_outcomes.csv": "0321e7fd4f32ab2ef6e36631958c7fefdab4f9508bd85bc32d62df8d86776d09",
    SOURCE / "per_gt_frame_cohort.csv": "79fe11f2a90746caf8923031d634e271855b108dec8f0ab2333945e276d30a0b",
    ROOT / "reports/full_nuscenes/ctep_method_activation/scene_list.csv": "7bbd389f8ec1f02e75d3d8a7feb773f109fdf24d5824b65db89fc7544e425fb3",
    ROOT / "configs/full_nuscenes/stream_petr_r50_90e_ctep_train_audit.py": "927ba2518a4ca460d2f7f6b3ba74dab620ac8e2995ee7e9aadbcbebf2d7c64a6",
    ROOT / "checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth": "e6323ae5c31adf1eedd46d6dd4fd3c73d95aa26f18cc8aa23c196494b7de3451",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def episode(path: Path) -> dict:
    value = json.loads(path.read_text())
    episodes = value.get("scenes", {}).get("*")
    if value.get("version") != 1 or not isinstance(episodes, list) or len(episodes) != 1:
        raise RuntimeError(f"invalid schedule: {path}")
    return dict(episodes[0])


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    observed = {str(path.relative_to(ROOT)): sha256(path) for path in EXPECTED}
    expected = {str(path.relative_to(ROOT)): value for path, value in EXPECTED.items()}
    if observed != expected:
        raise RuntimeError(f"immutable source hash mismatch: {observed}")
    for protocol in FORMAL:
        formal, probe = episode(FORMAL[protocol]), episode(PROBES[protocol])
        if (formal.pop("start_frame"), formal.pop("end_frame")) != (3, 12):
            raise RuntimeError(f"formal episode boundary changed: {protocol}")
        if (probe.pop("start_frame"), probe.pop("end_frame")) != (2, 2):
            raise RuntimeError(f"probe boundary changed: {protocol}")
        if formal != probe:
            raise RuntimeError(f"probe differs from formal protocol: {protocol}")
    empty = episode(REPORT / "probes/disabled_empty_frame2.json")
    if empty != {"start_frame": 2, "end_frame": 2, "failed_cameras": [],
                 "lost_cameras": [], "dark": {}, "fog": {}, "motion_blur": {}}:
        raise RuntimeError("disabled schedule is not an exact empty operation")

    source = pd.read_csv(SOURCE / "cohort_manifest.csv")
    required = {"trajectory_id", "protocol", "scene_token", "instance_token",
                "gt_class", "gt_token", "distance_m", "visibility_token",
                "alternative_view_count", "A_tp"}
    if set(source.columns) != required:
        raise RuntimeError(f"cohort schema changed: {source.columns.tolist()}")
    sizes = source.groupby("protocol").size().to_dict()
    if len(source) != 1323 or sizes != {"blur_back": 441, "crash_back": 441, "dark_back": 441}:
        raise RuntimeError(f"cohort coverage changed: {sizes}")
    if not source.A_tp.astype(bool).all() or source.scene_token.nunique() != 16:
        raise RuntimeError("cohort is not all frame-2 Clean TP across 16 frozen scenes")
    if source.groupby(["protocol", "scene_token", "instance_token"]).size().max() != 1:
        raise RuntimeError("duplicate prospective trajectory")

    frozen = REPORT / "frozen_cohort.csv"
    if frozen.exists() and sha256(frozen) != EXPECTED[SOURCE / "cohort_manifest.csv"]:
        raise RuntimeError("existing frozen cohort differs from immutable source")
    if not frozen.exists():
        shutil.copyfile(SOURCE / "cohort_manifest.csv", frozen)
    validation = {
        "status": "VALIDATED_BEFORE_FORWARD",
        "source_hashes": observed,
        "frozen_cohort_sha256": sha256(frozen),
        "preregistration_sha256": sha256(REPORT / "PRE_REGISTRATION.md"),
        "formal_protocol_sha256": {key: sha256(path) for key, path in FORMAL.items()},
        "probe_sha256": {key: sha256(path) for key, path in PROBES.items()},
        "disabled_probe_sha256": sha256(REPORT / "probes/disabled_empty_frame2.json"),
        "cohort_rows": len(source), "scenes": int(source.scene_token.nunique()),
        "rows_per_protocol": sizes,
        "stream_petr_commit": "95f64702306ccdb7a78889578b2a55b5deb35b2a",
    }
    atomic_json(REPORT / "source_validation.json", validation)
    initial = {
        "schema_version": 1, "status": "PREPARED_P0_PENDING",
        "frozen_cohort_sha256": validation["frozen_cohort_sha256"],
        "preregistration_sha256": validation["preregistration_sha256"],
        "stages": {"P0_forward": {}, "P0_analysis": "LOCKED",
                   "P1": "LOCKED_PENDING_P0", "P2": "LOCKED_PENDING_P0_P1"},
        "training": "PROHIBITED", "repository_modification": "PROHIBITED",
    }
    progress_path = REPORT / "progress_manifest.json"
    if progress_path.exists():
        current = json.loads(progress_path.read_text())
        for key in ("frozen_cohort_sha256", "preregistration_sha256"):
            if current.get(key) != initial[key]:
                raise RuntimeError(f"resume identity mismatch: {key}")
    else:
        atomic_json(progress_path, initial)
    (REPORT / "PARTIAL_STATUS.md").write_text(
        "# PARTIAL STATUS\n\n`PREPARED_P0_PENDING`\n\n"
        "The prospective frame-2 cohort and exact probe definitions are frozen. "
        "No new forward has run, so no Go/No-Go is permitted.\n\nResume:\n\n```bash\n"
        "python scripts/run_fault_stress_reserve_p0.py\n"
        "python scripts/analyze_fault_stress_reserve_p0.py\n```\n"
    )
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
