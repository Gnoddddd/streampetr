#!/usr/bin/env python3
"""Audit an OccNuScenes paired temporal manifest and enforce its gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.paired_occ_nuscenes import (  # noqa: E402
    NuScenesMetadata,
    audit_pair_records,
    format_integrity_report,
    official_splits,
    read_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--nuscenes-root", required=True)
    parser.add_argument("--dirty-root", required=True)
    parser.add_argument("--metadata-version", default="v1.0-trainval")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    nuscenes_root = Path(args.nuscenes_root)
    metadata_version = Path(args.metadata_version)
    metadata_directory = (
        metadata_version if metadata_version.is_absolute()
        else nuscenes_root / metadata_version
    )
    report = audit_pair_records(
        read_manifest(args.manifest),
        nuscenes_root,
        args.dirty_root,
        NuScenesMetadata.from_directory(metadata_directory),
        official_splits(),
    )
    print(format_integrity_report(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
