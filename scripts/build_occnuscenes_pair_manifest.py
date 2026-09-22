#!/usr/bin/env python3
"""Build a deterministic OccNuScenes paired temporal JSONL manifest."""

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
    build_pair_records,
    format_integrity_report,
    official_splits,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuscenes-root", required=True)
    parser.add_argument("--dirty-root", required=True)
    parser.add_argument("--metadata-version", default="v1.0-trainval")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--split", choices=("mini_train", "mini_val", "train", "val", "all"), default="all"
    )
    parser.add_argument(
        "--corruption-types", nargs="+", default=("Dirt", "Water-blur")
    )
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument(
        "--strict", action="store_true",
        help="Audit the written records and fail on any discovery or integrity error.",
    )
    return parser.parse_args()


def _metadata_directory(root: Path, version: str) -> Path:
    candidate = Path(version)
    return candidate if candidate.is_absolute() else root / candidate


def main() -> int:
    args = parse_args()
    nuscenes_root = Path(args.nuscenes_root)
    dirty_root = Path(args.dirty_root)
    metadata = NuScenesMetadata.from_directory(
        _metadata_directory(nuscenes_root, args.metadata_version)
    )
    splits = official_splits()
    records, build_report = build_pair_records(
        nuscenes_root=nuscenes_root,
        dirty_root=dirty_root,
        metadata=metadata,
        split=args.split,
        corruption_types=args.corruption_types,
        conditions=args.conditions,
        split_scene_names=splits,
    )
    write_manifest(args.output, records)
    report = build_report
    if args.strict:
        audit_report = audit_pair_records(
            records, nuscenes_root, dirty_root, metadata, splits
        )
        for category, messages in audit_report.errors.items():
            report.errors[category].extend(messages)
    print(format_integrity_report(report))
    return 1 if args.strict and not report.passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
