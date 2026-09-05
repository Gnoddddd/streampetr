#!/usr/bin/env python3
"""Validate and freeze inputs for the representation-side CTEP routing audit."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/full_nuscenes/ctep_method_activation"
REPORT = ROOT / "reports/full_nuscenes/ctep_gradient_routing_audit"
EXPECTED = {
    "gradient_units.csv": "eb733e626930032bcf952a6a609e56ae113ae28f180f126233580c129cb5737f",
    "per_gt_p0.csv": "0d4d962af9435234a90076eed6dc840603e031936188c86422fe622a2844ae79",
    "p0_decision.json": "b76fddedf8057c9fa4bdea96e422498ae966002a6db82e8f9473bf72fac31d5e",
    "p1_activation_cluster_ci.csv": "14602a04d44ed6405e10ed6110817c3f13999406434a25fc773722afef764f68",
    "activation_decision.json": "85e2da15e920d445fb891d663ef02a80ba184b615c1488ead3f35437f7f5a93e",
}
PROTOCOLS = ("blur_back", "crash_back", "dark_back")
EXPECTED_TERMS = {"blur_back": 27, "crash_back": 29, "dark_back": 31}


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
        raise RuntimeError(f"immutable source hash mismatch: {observed}")
    p0 = json.loads((SOURCE / "p0_decision.json").read_text())
    activation = json.loads((SOURCE / "activation_decision.json").read_text())
    if not p0.get("train_mechanism_reproduced"):
        raise RuntimeError("source train mechanism did not pass")
    enrichment = activation.get("history_sensitive_enrichment_each_protocol", {})
    if not all(all(enrichment.get(protocol, {}).values()) for protocol in PROTOCOLS):
        raise RuntimeError("source history-sensitive enrichment did not pass")
    if activation.get("verdict") != "NO_GO_CTEP_ACTIVATION":
        raise RuntimeError("source audit is not the expected activation No-Go")
    source_units = SOURCE / "gradient_units.csv"
    frozen_units = REPORT / "gradient_units.csv"
    if frozen_units.exists() and sha256(frozen_units) != EXPECTED["gradient_units.csv"]:
        raise RuntimeError("existing frozen gradient units differ from source")
    if not frozen_units.exists():
        shutil.copyfile(source_units, frozen_units)
    if sha256(frozen_units) != EXPECTED["gradient_units.csv"]:
        raise RuntimeError("frozen gradient-unit copy is not byte-identical")
    units = pd.read_csv(frozen_units)
    term_counts = {
        protocol: int(
            units.loc[units.protocol == protocol, "active_terms"]
            .map(lambda value: len(json.loads(value)))
            .sum()
        )
        for protocol in PROTOCOLS
    }
    scene_counts = {
        protocol: int(units.loc[units.protocol == protocol, "scene_token"].nunique())
        for protocol in PROTOCOLS
    }
    if term_counts != EXPECTED_TERMS or any(value != 4 for value in scene_counts.values()):
        raise RuntimeError(f"frozen term inventory mismatch: {term_counts}/{scene_counts}")
    validation = {
        "status": "VALIDATED_BEFORE_FORWARD",
        "source_directory": str(SOURCE.relative_to(ROOT)),
        "source_hashes": observed,
        "frozen_gradient_units_sha256": sha256(frozen_units),
        "p0_train_mechanism_reproduced": True,
        "history_sensitive_enrichment_each_protocol": enrichment,
        "active_term_counts": term_counts,
        "scene_counts": scene_counts,
    }
    atomic_json(REPORT / "source_validation.json", validation)
    manifest_path = REPORT / "progress_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("frozen_gradient_units_sha256") != sha256(frozen_units):
            raise RuntimeError("resume manifest frozen-unit hash mismatch")
    else:
        manifest = {
            "schema_version": 1,
            "status": "PREPARED_NO_FORWARD_YET",
            "frozen_gradient_units_sha256": sha256(frozen_units),
            "protocols": {
                protocol: {"expected_scenes": scene_counts[protocol], "completed_scenes": []}
                for protocol in PROTOCOLS
            },
            "conditional_stages": {
                "single_batch_overfit": "LOCKED",
                "two_iter_smoke": "LOCKED",
                "short_training": "LOCKED",
            },
        }
        atomic_json(manifest_path, manifest)
    status = """# PARTIAL STATUS

`PREPARED_NO_FORWARD_YET`

The immutable source hashes, train mechanism, history-sensitive enrichment,
and exact 87-term/12-scene inventory are validated.  No new forward has run.

Resume in priority order:

```bash
python scripts/run_ctep_gradient_routing_audit.py --protocol blur_back
python scripts/run_ctep_gradient_routing_audit.py --protocol crash_back
python scripts/run_ctep_gradient_routing_audit.py --protocol dark_back
python scripts/analyze_ctep_gradient_routing_audit.py
```
"""
    (REPORT / "PARTIAL_STATUS.md").write_text(status)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()

