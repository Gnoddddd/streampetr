#!/usr/bin/env python3
"""Validate immutable utility-audit inputs before supplemental forward."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/full_nuscenes/temporal_utility_audit"
EXPECTED = {
    ROOT / "reports/full_nuscenes/ctep_method_activation/per_gt_p0.csv": "0d4d962af9435234a90076eed6dc840603e031936188c86422fe622a2844ae79",
    ROOT / "reports/full_nuscenes/bd_temporal_support_audit/per_gt_nohistory.csv": "44d0f4f5223b1dbe71691f31bec8e3704b6bd6f973bc052b0dea469592cb26f5",
    ROOT / "reports/full_nuscenes/ctep_method_activation/scene_list.csv": "7bbd389f8ec1f02e75d3d8a7feb773f109fdf24d5824b65db89fc7544e425fb3",
    ROOT / "configs/full_nuscenes/stream_petr_r50_90e_ctep_train_audit.py": "927ba2518a4ca460d2f7f6b3ba74dab620ac8e2995ee7e9aadbcbebf2d7c64a6",
    ROOT / "checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth": "e6323ae5c31adf1eedd46d6dd4fd3c73d95aa26f18cc8aa23c196494b7de3451",
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    observed = {str(path.relative_to(ROOT)): sha256(path) for path in EXPECTED}
    expected = {str(path.relative_to(ROOT)): digest for path, digest in EXPECTED.items()}
    if observed != expected:
        raise RuntimeError(f"source hash mismatch: {observed}")
    source = pd.read_csv(next(path for path in EXPECTED if path.name == "per_gt_p0.csv"))
    if len(source) != 17466 or source.groupby("protocol").size().to_dict() != {
        "blur_back": 5822, "crash_back": 5822, "dark_back": 5822,
    }:
        raise RuntimeError("A/B/C/D all-GT cache coverage changed")
    split = pd.read_csv(REPORT / "scene_split.csv")
    if len(split) != 16 or split.fold.value_counts().sort_index().to_dict() != {0: 4, 1: 4, 2: 4, 3: 4}:
        raise RuntimeError("fixed scene split invalid")
    validation = {
        "status": "VALIDATED_BEFORE_SUPPLEMENT_FORWARD",
        "source_hashes": observed,
        "source_abcd_rows": len(source),
        "scene_split_sha256": sha256(REPORT / "scene_split.csv"),
        "preregistration_sha256": sha256(REPORT / "PRE_REGISTRATION.md"),
        "existing_disabled_summary_sha256": sha256(
            ROOT / "reports/full_nuscenes/bd_temporal_support_audit/disabled_equivalence_summary.json"),
    }
    atomic_json(REPORT / "source_validation.json", validation)
    progress_path = REPORT / "progress_manifest.json"
    initial = {
        "schema_version": 1,
        "status": "PREPARED_SUPPLEMENT_PENDING",
        "scene_split_sha256": validation["scene_split_sha256"],
        "preregistration_sha256": validation["preregistration_sha256"],
        "stages": {"supplement": {}, "P0": "LOCKED", "P1": "LOCKED", "P2": "LOCKED"},
        "training": "PROHIBITED",
    }
    if progress_path.exists():
        current = json.loads(progress_path.read_text())
        for key in ("scene_split_sha256", "preregistration_sha256"):
            if current.get(key) != initial[key]:
                raise RuntimeError(f"resume mismatch: {key}")
    else:
        atomic_json(progress_path, initial)
    (REPORT / "PARTIAL_STATUS.md").write_text(
        "# PARTIAL STATUS\n\n`PREPARED_SUPPLEMENT_PENDING`\n\n"
        "Pre-registration and the outcome-independent scene split are frozen. "
        "No supplemental forward has run.\n\nResume:\n\n```bash\n"
        "python scripts/run_temporal_utility_supplement.py --protocol blur_back\n"
        "python scripts/run_temporal_utility_supplement.py --protocol crash_back\n"
        "python scripts/run_temporal_utility_supplement.py --protocol dark_back\n"
        "python scripts/analyze_temporal_utility.py\n```\n"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
