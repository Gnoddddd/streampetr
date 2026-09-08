#!/usr/bin/env python3
"""Analyze the 419-scene CARE-3D P2-A2-R0 development pre-gate."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.care3d_p2a2_lineage import (
    BOOTSTRAP_REPETITIONS,
    FROZEN_P2A0_CONFIG,
    PRIMARY_METHOD,
    paired_delta_bootstrap,
    protocol_metrics,
)
from analysis.care3d_p2a_association import PROTOCOLS


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/care3d/p2a2_memory_lineage_r0"
SCHEMA = 1


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def load_probe_train_rows() -> pd.DataFrame:
    validation = json.loads((REPORT / "source_validation.json").read_text())
    if validation.get("probe_val_read") is not False or validation.get("probe_test_read") is not False:
        raise RuntimeError("P2-A2-R0 source validation indicates held-out access")
    manifest = pd.read_csv(REPORT / "probe_train_manifest.csv")
    if len(manifest) != 419 or set(manifest.split.astype(str)) != {"probe_train"}:
        raise RuntimeError("P2-A2-R0 analysis requires the frozen 419-scene train manifest")
    frames = []
    for scene in manifest.scene_token.astype(str):
        prefix = REPORT / "incremental/probe_train" / scene
        marker_path = prefix.with_suffix(".complete.json")
        rows_path = prefix.with_suffix(".rows.csv")
        if not marker_path.exists() or not rows_path.exists():
            raise RuntimeError(f"missing P2-A2-R0 probe-train scene: {scene}")
        marker = json.loads(marker_path.read_text())
        valid = (
            marker.get("complete")
            and marker.get("split") == "probe_train"
            and marker.get("p2a0_selection_sha256") == validation["p2a0_selection_sha256"]
            and marker.get("probe_val_read") is False
            and marker.get("probe_test_read") is False
            and marker.get("gt_used_as_input") is False
            and marker.get("clean_future_used_as_input") is False
            and marker.get("oracle_query_used_as_input") is False
        )
        if not valid:
            raise RuntimeError(f"invalid P2-A2-R0 marker: {scene}")
        frame = pd.read_csv(rows_path)
        if len(frame) != int(marker.get("rows", -1)):
            raise RuntimeError(f"P2-A2-R0 marker/row mismatch: {scene}")
        if len(frame):
            frames.append(frame)
    if not frames:
        raise RuntimeError("P2-A2-R0 probe-train contains no rows")
    rows = pd.concat(frames, ignore_index=True)
    if set(rows.protocol.astype(str)) != set(PROTOCOLS):
        raise RuntimeError("P2-A2-R0 protocol set changed")
    for flag in (
        "gt_used_as_association_input",
        "clean_future_used_as_association_input",
        "oracle_query_used_as_association_input",
    ):
        if rows[flag].astype(bool).any():
            raise RuntimeError(f"forbidden association input flag set: {flag}")
    if not set(rows.hybrid_source.astype(str)) <= {"lineage", "p2a0_fallback", "unmatched"}:
        raise RuntimeError("P2-A2-R0 hybrid source changed")
    return rows


def diagnostic_rows(frame: pd.DataFrame, protocol: str) -> list[dict]:
    output = []
    for column in ("anchor_query_origin", "oracle_query_origin", "target_frame_idx"):
        for value, group in frame.groupby(column, sort=True):
            metrics = protocol_metrics(group)
            output.append({
                "protocol": protocol,
                "stratifier": column,
                "stratum": str(value),
                **metrics,
            })
    return output


def main() -> None:
    rows = load_probe_train_rows()
    metric_rows = []
    ci_rows = []
    diagnostics = []
    gate_rows = []
    for protocol_index, protocol in enumerate(PROTOCOLS):
        group = rows[rows.protocol.astype(str) == protocol].reset_index(drop=True)
        metrics = protocol_metrics(group)
        metric_rows.append({"protocol": protocol, **metrics})
        bootstrap_by_cluster = {}
        for cluster_index, cluster in enumerate(("scene_token", "instance_token")):
            result = paired_delta_bootstrap(
                group,
                cluster_column=cluster,
                repetitions=BOOTSTRAP_REPETITIONS,
                seed=930000 + protocol_index * 100 + cluster_index * 10,
            )
            bootstrap_by_cluster[cluster] = result
            for metric, values in result.items():
                ci_rows.append({
                    "protocol": protocol,
                    "cluster": cluster,
                    "metric": metric,
                    **values,
                })
        diagnostics.extend(diagnostic_rows(group, protocol))
        wrong_pass = bool(metrics["hybrid_wrong_match_rate"] <= 0.10)
        scene_exact_pass = bool(
            bootstrap_by_cluster["scene_token"]["delta_exact"]["ci_low"] > 0.0
        )
        instance_exact_pass = bool(
            bootstrap_by_cluster["instance_token"]["delta_exact"]["ci_low"] > 0.0
        )
        unmatched_pass = bool(metrics["hybrid_unmatched_rate"] <= 0.01)
        heldout_pass = True
        gate_rows.append({
            "protocol": protocol,
            "hybrid_wrong_match_rate_pass": wrong_pass,
            "scene_delta_exact_ci_low_gt_zero": scene_exact_pass,
            "instance_delta_exact_ci_low_gt_zero": instance_exact_pass,
            "hybrid_unmatched_rate_pass": unmatched_pass,
            "probe_val_read_false": heldout_pass,
            "probe_test_read_false": heldout_pass,
            "protocol_pass": bool(
                wrong_pass and scene_exact_pass and instance_exact_pass
                and unmatched_pass and heldout_pass
            ),
        })

    metrics_frame = pd.DataFrame(metric_rows)
    ci_frame = pd.DataFrame(ci_rows)
    diagnostics_frame = pd.DataFrame(diagnostics)
    gate_frame = pd.DataFrame(gate_rows)
    atomic_csv(REPORT / "p2a2_r0_protocol_metrics.csv", metrics_frame)
    atomic_csv(REPORT / "p2a2_r0_paired_cluster_ci.csv", ci_frame)
    atomic_csv(REPORT / "p2a2_r0_diagnostic_strata.csv", diagnostics_frame)
    atomic_csv(REPORT / "p2a2_r0_gate_summary.csv", gate_frame)

    passing = gate_frame.loc[gate_frame.protocol_pass.astype(bool), "protocol"].tolist()
    go = len(passing) >= 2
    decision = {
        "schema_version": SCHEMA,
        "decision": (
            "GO_P2A2_R0_MEMORY_LINEAGE" if go else "NO_GO_P2A2_R0_MEMORY_LINEAGE"
        ),
        "development_pre_gate": True,
        "confirmatory_validation": False,
        "primary_method": PRIMARY_METHOD,
        "baselines": ["p2a0_frozen", "lineage_only"],
        "selected_config": FROZEN_P2A0_CONFIG.as_dict(),
        "passing_protocols": passing,
        "required_passing_protocols": 2,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "probe_train_scenes": 419,
        "probe_val_read": False,
        "probe_test_read": False,
        "P2A0_gate_modified": False,
    }
    atomic_json(REPORT / "decision.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
