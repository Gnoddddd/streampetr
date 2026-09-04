#!/usr/bin/env python3
"""Freeze the source population and real-graph tap inventory before forward."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/full_nuscenes/ctep_method_activation"
REPORT = ROOT / "reports/full_nuscenes/temporal_representation_localization_audit"
EXPECTED = {
    "per_gt_p0.csv": "0d4d962af9435234a90076eed6dc840603e031936188c86422fe622a2844ae79",
    "scene_list.csv": "7bbd389f8ec1f02e75d3d8a7feb773f109fdf24d5824b65db89fc7544e425fb3",
    "gradient_units.csv": "eb733e626930032bcf952a6a609e56ae113ae28f180f126233580c129cb5737f",
    "p0_decision.json": "b76fddedf8057c9fa4bdea96e422498ae966002a6db82e8f9473bf72fac31d5e",
}
PROTOCOLS = ("blur_back", "crash_back", "dark_back")
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
    observed = {name: sha256(SOURCE / name) for name in EXPECTED}
    if observed != EXPECTED:
        raise RuntimeError(f"source hash mismatch: {observed}")
    source = pd.read_csv(SOURCE / "per_gt_p0.csv")
    population = source[source.lost.astype(bool) | source.retained.astype(bool)].copy()
    counts = {
        protocol: {
            "lost": int(population[population.protocol == protocol].lost.astype(bool).sum()),
            "retained": int(population[population.protocol == protocol].retained.astype(bool).sum()),
        }
        for protocol in PROTOCOLS
    }
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"frozen population count mismatch: {counts}")
    population_path = REPORT / "population.csv"
    if population_path.exists():
        existing = pd.read_csv(population_path)
        if not existing.equals(population):
            raise RuntimeError("existing population manifest differs from source filter")
    else:
        population.to_csv(population_path, index=False)

    taps = pd.DataFrame([
        {
            "tap_id": "temporal_alignment_query_state",
            "readout_proximity_rank": 1,
            "module_path": "pts_bbox_head.temporal_alignment return",
            "source_file": "projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py",
            "source_lines": "420-449;603",
            "tensor": "concat(tgt,query_pos)",
            "feature_dim": 512,
            "patch_semantics": "restore both tgt and query_pos components",
        },
        {
            "tap_id": "decoder_layer5_temporal_self_attn_output",
            "readout_proximity_rank": 2,
            "module_path": "pts_bbox_head.transformer.decoder.layers.5.attentions.0",
            "source_file": "projects/mmdet3d_plugin/models/utils/petr_transformer.py",
            "source_lines": "707-723",
            "tensor": "final-layer temporal self-attention output",
            "feature_dim": 256,
            "patch_semantics": "restore matched output query rows",
        },
        {
            "tap_id": "final_decoder_pre_cls_query",
            "readout_proximity_rank": 3,
            "module_path": "pts_bbox_head.cls_branches[5] input",
            "source_file": "projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py",
            "source_lines": "606-615",
            "tensor": "outs_dec[5]",
            "feature_dim": 256,
            "patch_semantics": "restore matched pre-classification query rows",
        },
    ])
    tap_path = REPORT / "tap_manifest.csv"
    if tap_path.exists():
        if not pd.read_csv(tap_path).equals(taps):
            raise RuntimeError("existing tap manifest differs")
    else:
        taps.to_csv(tap_path, index=False)

    gradient_units = pd.read_csv(SOURCE / "gradient_units.csv")
    gradient_counts = (
        gradient_units.groupby(["protocol", "gradient_stratum"]).size().unstack(fill_value=0).to_dict("index")
    )
    validation = {
        "status": "VALIDATED_BEFORE_FORWARD",
        "source_hashes": observed,
        "population_sha256": sha256(population_path),
        "tap_manifest_sha256": sha256(tap_path),
        "population_counts": counts,
        "population_rows": len(population),
        "scene_count": int(pd.read_csv(SOURCE / "scene_list.csv").scene_token.nunique()),
        "p2_prefrozen_gradient_counts": gradient_counts,
    }
    atomic_json(REPORT / "source_validation.json", validation)
    manifest = {
        "schema_version": 1,
        "status": "PREPARED_NO_FORWARD_YET",
        "population_sha256": validation["population_sha256"],
        "tap_manifest_sha256": validation["tap_manifest_sha256"],
        "stages": {"P0": {}, "P1": "LOCKED", "P2": "LOCKED"},
        "conditional_training": "PROHIBITED_IN_THIS_AUDIT",
    }
    progress_path = REPORT / "progress_manifest.json"
    if progress_path.exists():
        existing = json.loads(progress_path.read_text())
        for key in ("population_sha256", "tap_manifest_sha256"):
            if existing.get(key) != manifest[key]:
                raise RuntimeError(f"resume {key} mismatch")
    else:
        atomic_json(progress_path, manifest)
    (REPORT / "PARTIAL_STATUS.md").write_text(
        "# PARTIAL STATUS\n\n`PREPARED_NO_FORWARD_YET`\n\n"
        "Frozen population and the three real-graph taps are validated. No new forward has run.\n\n"
        "Resume:\n\n```bash\n"
        "python scripts/run_temporal_representation_p0.py --protocol blur_back\n"
        "python scripts/run_temporal_representation_p0.py --protocol crash_back\n"
        "python scripts/run_temporal_representation_p0.py --protocol dark_back\n"
        "python scripts/analyze_temporal_representation_p0.py\n```\n"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()

