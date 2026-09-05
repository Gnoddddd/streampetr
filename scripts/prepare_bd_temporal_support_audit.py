#!/usr/bin/env python3
"""Freeze BD temporal-support audit inputs before any new model forward."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "reports/full_nuscenes/temporal_representation_localization_audit"
SOURCE = ROOT / "reports/full_nuscenes/ctep_method_activation"
REPORT = ROOT / "reports/full_nuscenes/bd_temporal_support_audit"
EXPECTED = {
    PARENT / "population.csv": "08305128a6eae685b6c5e1c2af4d1508f0dfe93419350d812c9233a523d06452",
    SOURCE / "scene_list.csv": "7bbd389f8ec1f02e75d3d8a7feb773f109fdf24d5824b65db89fc7544e425fb3",
    ROOT / "configs/full_nuscenes/stream_petr_r50_90e_ctep_train_audit.py": "927ba2518a4ca460d2f7f6b3ba74dab620ac8e2995ee7e9aadbcbebf2d7c64a6",
    ROOT / "checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth": "e6323ae5c31adf1eedd46d6dd4fd3c73d95aa26f18cc8aa23c196494b7de3451",
    PARENT / "per_gt_causal_patch.csv": "30cfcd75c0d0db7b7b39d04881b7fee01a712303729302eab4fa0769aedb35bd",
}
EXPECTED_COUNTS = {
    "blur_back": {"lost": 128, "retained": 4473},
    "crash_back": {"lost": 424, "retained": 4177},
    "dark_back": {"lost": 171, "retained": 4430},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    observed = {str(path.relative_to(ROOT)): sha256(path) for path in EXPECTED}
    expected = {str(path.relative_to(ROOT)): value for path, value in EXPECTED.items()}
    if observed != expected:
        raise RuntimeError(f"immutable source hash mismatch: {observed}")
    source = pd.read_csv(PARENT / "population.csv")
    counts = {
        protocol: {
            "lost": int(group.lost.astype(bool).sum()),
            "retained": int(group.retained.astype(bool).sum()),
        }
        for protocol, group in source.groupby("protocol", sort=False)
    }
    if counts != EXPECTED_COUNTS or len(source) != 13803:
        raise RuntimeError(f"population changed: rows={len(source)}, counts={counts}")
    target = REPORT / "population.csv"
    if target.exists():
        if not pd.read_csv(target).equals(source):
            raise RuntimeError("existing frozen population differs")
    else:
        source.to_csv(target, index=False)
    validation = {
        "status": "VALIDATED_BEFORE_FORWARD",
        "source_hashes": observed,
        "population_sha256": sha256(target),
        "preregistration_sha256": sha256(REPORT / "PRE_REGISTRATION.md"),
        "support_tap_manifest_sha256": sha256(REPORT / "support_tap_manifest.csv"),
        "population_rows": len(source),
        "population_counts": counts,
        "scene_count": int(pd.read_csv(SOURCE / "scene_list.csv").scene_token.nunique()),
    }
    atomic_json(REPORT / "source_validation.json", validation)
    manifest_path = REPORT / "progress_manifest.json"
    initial = {
        "schema_version": 1,
        "status": "PREPARED_NO_FORWARD_YET",
        "population_sha256": validation["population_sha256"],
        "preregistration_sha256": validation["preregistration_sha256"],
        "support_tap_manifest_sha256": validation["support_tap_manifest_sha256"],
        "stages": {"P0": {}, "P1": "LOCKED", "P2": "LOCKED"},
        "training": "PROHIBITED",
    }
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text())
        for key in ("population_sha256", "preregistration_sha256", "support_tap_manifest_sha256"):
            if current.get(key) != initial[key]:
                raise RuntimeError(f"resume manifest mismatch: {key}")
    else:
        atomic_json(manifest_path, initial)
    (REPORT / "PARTIAL_STATUS.md").write_text(
        "# PARTIAL STATUS\n\n`PREPARED_NO_FORWARD_YET`\n\n"
        "Frozen inputs and preregistration are complete; no new model forward has run.\n\n"
        "Resume:\n\n```bash\n"
        "python scripts/run_bd_temporal_support_p0.py --protocol blur_back\n"
        "python scripts/run_bd_temporal_support_p0.py --protocol crash_back\n"
        "python scripts/run_bd_temporal_support_p0.py --protocol dark_back\n"
        "python scripts/analyze_bd_temporal_support_p0.py\n```\n"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
