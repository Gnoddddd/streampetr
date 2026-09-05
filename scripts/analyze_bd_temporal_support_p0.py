#!/usr/bin/env python3
"""Aggregate P0 NoHistory anchors and apply the preregistered mechanism gate."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.bd_temporal_support import (
    classify_protocol, two_group_cluster_contrast, two_stage_cluster_bootstrap,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/full_nuscenes/bd_temporal_support_audit"
PROTOCOLS = ("blur_back", "crash_back", "dark_back")
METRICS = (
    "G_A", "G_C", "G_B", "G_D", "current_only_deficit",
    "AC_gap_from_anchor", "BD_gap_from_anchor",
)
BOOTSTRAPS = 5000


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    population = pd.read_csv(REPORT / "population.csv")
    frames, equivalence, coverage = [], [], []
    all_complete = True
    for protocol in PROTOCOLS:
        expected_scenes = set(population.loc[population.protocol == protocol, "scene_token"].astype(str))
        directory = REPORT / "incremental/P0" / protocol
        completed = {}
        for path in directory.glob("*.complete.json"):
            meta = json.loads(path.read_text())
            if meta.get("complete") and meta.get("schema_version") == 1:
                completed[str(meta["scene_token"])] = meta
        valid = expected_scenes & set(completed)
        missing = expected_scenes - valid
        protocol_rows = []
        protocol_exact = []
        for scene in sorted(valid):
            protocol_rows.append(pd.read_csv(directory / f"{scene}.csv"))
            protocol_exact.append(pd.read_csv(directory / f"{scene}.equivalence.csv"))
        if protocol_rows:
            frame = pd.concat(protocol_rows, ignore_index=True)
            frames.append(frame)
            equivalence.append(pd.concat(protocol_exact, ignore_index=True))
            observed_units = set(frame.unit_id)
        else:
            observed_units = set()
        expected_units = set(population.loc[population.protocol == protocol, "unit_id"])
        coverage.append({
            "protocol": protocol,
            "completed_scenes": len(valid),
            "expected_scenes": len(expected_scenes),
            "rows": len(observed_units),
            "expected_rows": len(expected_units),
            "missing_scenes": json.dumps(sorted(missing)),
            "duplicate_units": (0 if not protocol_rows else int(frame.unit_id.duplicated().sum())),
            "complete": not missing and observed_units == expected_units,
        })
        all_complete &= coverage[-1]["complete"]
    write_csv(REPORT / "p0_coverage.csv", coverage)
    if not all_complete:
        progress = json.loads((REPORT / "progress_manifest.json").read_text())
        progress["status"] = "PARTIAL_INSUFFICIENT_COVERAGE"
        atomic_json(REPORT / "progress_manifest.json", progress)
        raise RuntimeError("P0 coverage incomplete; final decision prohibited")
    data = pd.concat(frames, ignore_index=True)
    if len(data) != len(population) or data.unit_id.duplicated().any():
        raise RuntimeError("P0 row identity mismatch")
    data["AC_gap_from_anchor"] = data["G_A"] - data["G_C"]
    data["BD_gap_from_anchor"] = data["G_B"] - data["G_D"]
    data.to_csv(REPORT / "per_gt_nohistory.csv", index=False)
    exact = pd.concat(equivalence, ignore_index=True)
    exact.to_csv(REPORT / "p0_disabled_equivalence.csv", index=False)
    disabled = exact[exact.check == "disabled_wrapper_B0"]
    path_checks = exact[exact.check == "nohistory_current_path"]
    exact_ok = bool(
        len(disabled) == 48
        and disabled.output_bitwise_equal.astype(bool).all()
        and disabled.memory_bitwise_equal.astype(bool).all()
        and (disabled.output_max_abs_diff.astype(float) == 0).all()
        and (disabled.memory_max_abs_diff.astype(float) == 0).all()
        and (path_checks.nohistory_query_count.astype(int) == 644).all()
        and path_checks.E_temp_memory_is_none.astype(bool).all()
        and path_checks.F_temp_memory_is_none.astype(bool).all()
    )
    if not exact_ok:
        raise RuntimeError("NoHistory/disabled exactness gate failed")

    summaries, cis, decisions = [], [], []
    by_key = {}
    for protocol_index, protocol in enumerate(PROTOCOLS):
        selected = data[data.protocol == protocol]
        for metric_index, metric in enumerate(METRICS):
            for group_index, group in enumerate(("lost", "retained")):
                result = two_stage_cluster_bootstrap(
                    selected, metric, group, n_boot=BOOTSTRAPS,
                    seed=626262 + protocol_index * 100 + metric_index * 10 + group_index,
                )
                row = {"protocol": protocol, "population": group, "metric": metric, **result}
                cis.append(row)
                by_key[(protocol, group, metric)] = row
            contrast = two_group_cluster_contrast(
                selected, metric, n_boot=BOOTSTRAPS,
                seed=636363 + protocol_index * 100 + metric_index,
            )
            row = {"protocol": protocol, "population": "lost_minus_retained", "metric": metric, **contrast}
            cis.append(row)
            by_key[(protocol, "lost_minus_retained", metric)] = row
        lost = selected[selected.population == "lost"]
        retained = selected[selected.population == "retained"]
        eligibility = {
            "lost_finite": int((np.isfinite(lost.G_B) & np.isfinite(lost.G_D)).sum()),
            "lost_scenes": int(lost.loc[np.isfinite(lost.G_B) & np.isfinite(lost.G_D), "scene_token"].nunique()),
            "retained_finite": int((np.isfinite(retained.G_B) & np.isfinite(retained.G_D)).sum()),
            "retained_scenes": int(retained.loc[np.isfinite(retained.G_B) & np.isfinite(retained.G_D), "scene_token"].nunique()),
        }
        gate = classify_protocol(
            by_key[(protocol, "lost", "G_B")],
            by_key[(protocol, "lost", "G_D")],
            by_key[(protocol, "lost_minus_retained", "G_B")],
            by_key[(protocol, "lost_minus_retained", "G_D")],
        )
        gate["coverage_pass"] = bool(
            eligibility["lost_finite"] >= 20 and eligibility["lost_scenes"] >= 6
            and eligibility["retained_finite"] >= 50 and eligibility["retained_scenes"] >= 8)
        gate["protocol"] = protocol
        gate.update(eligibility)
        decisions.append(gate)
        summary = {"protocol": protocol, **eligibility, **gate}
        for metric in METRICS:
            for group in ("lost", "retained", "lost_minus_retained"):
                item = by_key[(protocol, group, metric)]
                prefix = f"{group}_{metric}"
                summary[f"{prefix}_estimate"] = item["estimate"]
                summary[f"{prefix}_ci_low"] = item["ci_low"]
                summary[f"{prefix}_ci_high"] = item["ci_high"]
        summaries.append(summary)
    write_csv(REPORT / "p0_cluster_ci.csv", cis)
    write_csv(REPORT / "p0_summary.csv", summaries)
    write_csv(REPORT / "p0_protocol_decisions.csv", decisions)

    mechanisms = [row["mechanism"] for row in decisions]
    component_counts = {
        "compensation": sum(value in {"clean_history_compensation", "both"} for value in mechanisms),
        "contamination": sum(value in {"fault_history_contamination", "both"} for value in mechanisms),
    }
    p0_pass = bool(
        exact_ok and all(row["coverage_pass"] and row["pattern_pass"] for row in decisions)
        and max(component_counts.values()) >= 2
    )
    verdict = "P0_GO_P1_REQUIRED" if p0_pass else "NO_GO_LOCALIZABLE_BD_TEMPORAL_SUPPORT"
    decision = {
        "verdict": verdict,
        "p0_pass": p0_pass,
        "protocol_mechanisms": {row["protocol"]: row["mechanism"] for row in decisions},
        "component_protocol_counts": component_counts,
        "disabled_exact": exact_ok,
        "coverage_complete": True,
        "bootstrap": {"unit": "instance-trajectory within scene", "replicates": BOOTSTRAPS},
    }
    atomic_json(REPORT / "p0_decision.json", decision)
    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    progress["stages"]["P0"] = {"status": "COMPLETE", "decision": verdict}
    if p0_pass:
        progress["status"] = "P0_GO_P1_PENDING"
        progress["stages"]["P1"] = "UNLOCKED_PENDING"
    else:
        progress["status"] = "FINAL_NO_GO_LOCALIZABLE_BD_TEMPORAL_SUPPORT"
        progress["stages"]["P1"] = "LOCKED_P0_FAILED"
        progress["stages"]["P2"] = "LOCKED_P0_FAILED"
        atomic_json(REPORT / "p1_status.json", {"status": "LOCKED_NOT_RUN", "reason": verdict})
        atomic_json(REPORT / "p2_status.json", {"status": "LOCKED_NOT_RUN", "reason": verdict})
        atomic_json(REPORT / "final_decision.json", {
            "decision": verdict,
            "failed_stage": "P0",
            "reason": "NoHistory anchor did not satisfy a preregistered BD mechanism pattern with lost enrichment in every protocol.",
            "protocol_mechanisms": decision["protocol_mechanisms"],
            "P1": "LOCKED_NOT_RUN",
            "P2": "LOCKED_NOT_RUN",
            "training": "NOT_RUN_PROHIBITED",
        })
    atomic_json(REPORT / "progress_manifest.json", progress)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
