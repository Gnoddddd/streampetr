#!/usr/bin/env python3
"""Final preregistered CARE-3D P2-A0 probe-val Go/No-Go analysis."""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.care3d_p1 import cluster_bootstrap_mean
from analysis.care3d_p2a_association import (
    P2A_QUERY_COLLISION_POLICY,
    PROTOCOLS,
    aggregate_summary,
    baseline_configs,
    p2a_protocol_gate,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/care3d/p2a_online_query_association"
CONFIG = ROOT / "configs/care3d/p2a_online_query_association.py"
SCHEMA = 1


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_rows(validation: dict, selection: dict) -> pd.DataFrame:
    manifest = pd.read_csv(REPORT / "frozen_scene_manifest.csv")
    scenes = manifest[manifest.split.astype(str) == "probe_val"].scene_token.astype(str).tolist()
    if len(scenes) != 133:
        raise RuntimeError(f"P2-A probe-val scene count changed: {len(scenes)}")
    frames = []
    selected_id = str(selection["selected"]["config_id"])
    selected_max_cost = float(selection["selected"]["max_cost"])
    expected_methods = {"selected_full", "geometry_only", "embedding_only", "class_geometry"}
    baseline_ids = {
        name: config.config_id for name, config in baseline_configs(selected_max_cost)
    }
    for scene in scenes:
        prefix = REPORT / "incremental/probe_val" / scene
        marker_path = prefix.with_suffix(".complete.json")
        rows_path = prefix.with_suffix(".rows.csv")
        if not marker_path.exists() or not rows_path.exists():
            raise RuntimeError(f"missing P2-A probe-val scene: {scene}")
        marker = json.loads(marker_path.read_text())
        valid = (
            marker.get("complete")
            and marker.get("schema_version") == SCHEMA
            and marker.get("scene_manifest_sha256") == validation["scene_manifest_sha256"]
            and marker.get("split") == "probe_val"
            and marker.get("query_collision_policy") == P2A_QUERY_COLLISION_POLICY
            and marker.get("probe_test_read") is False
        )
        if not valid:
            raise RuntimeError(f"invalid P2-A probe-val marker: {scene}")
        frame = pd.read_csv(rows_path)
        if len(frame) == 0:
            if int(marker.get("eligible_rows", -1)) != 0:
                raise RuntimeError(f"empty P2-A val CSV but marker has rows: {scene}")
            continue
        methods = set(frame.method.astype(str))
        if methods != expected_methods:
            raise RuntimeError(f"P2-A val method set changed for {scene}: {methods}")
        selected_rows = frame[frame.method.astype(str) == "selected_full"]
        if set(selected_rows.config_id.astype(str)) != {selected_id}:
            raise RuntimeError(f"P2-A selected config changed in val scene: {scene}")
        for name, expected_id in baseline_ids.items():
            ids = set(frame[frame.method.astype(str) == name].config_id.astype(str))
            if ids != {expected_id}:
                raise RuntimeError(f"P2-A baseline config changed: {name} {scene}")
        for flag in (
            "gt_used_as_association_input",
            "clean_future_used_as_association_input",
            "oracle_query_used_as_association_input",
        ):
            if frame[flag].astype(bool).any():
                raise RuntimeError(f"P2-A forbidden association input flag set: {flag}")
        frames.append(frame)
    if not frames:
        raise RuntimeError("P2-A probe-val contains no eligible association rows")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    validation = json.loads((REPORT / "source_validation.json").read_text())
    selection = json.loads((REPORT / "selection.json").read_text())
    progress_path = REPORT / "progress_manifest.json"
    progress = json.loads(progress_path.read_text())
    if validation.get("probe_test_read") is not False or progress.get("probe_test_read") is not False:
        raise RuntimeError("P2-A probe-test lock was violated")
    val_stage = progress.get("stages", {}).get("probe_val_extraction", {})
    if not isinstance(val_stage, dict) or int(val_stage.get("completed_scenes", -1)) != 133:
        raise RuntimeError("all 133 P2-A probe-val scenes are required")
    if selection.get("status") != "P2A_GLOBAL_ASSOCIATION_CONFIG_FROZEN":
        raise RuntimeError("P2-A config selection is not frozen")

    cfg = runpy.run_path(str(CONFIG))
    gate = dict(cfg["gate"])
    repetitions = int(gate["bootstrap_repetitions"])
    if repetitions != 5000:
        raise RuntimeError("formal P2-A analysis requires frozen 5000 bootstraps")
    rows = load_rows(validation, selection)

    metric_rows = []
    cluster_rows = []
    gate_rows = []
    methods = ("selected_full", "geometry_only", "embedding_only", "class_geometry")
    for protocol_index, protocol in enumerate(PROTOCOLS):
        for method_index, method in enumerate(methods):
            group = rows[
                (rows.protocol.astype(str) == protocol)
                & (rows.method.astype(str) == method)
            ].reset_index(drop=True)
            if len(group) == 0:
                raise RuntimeError(f"empty P2-A val result protocol={protocol} method={method}")
            point = aggregate_summary(group)
            point.update({
                "protocol": protocol,
                "method": method,
                "oracle_rank_mean": float(group.oracle_rank.astype(float).mean()),
                "oracle_rank_median": float(group.oracle_rank.astype(float).median()),
                "correct_vs_best_wrong_margin_mean": float(
                    group.correct_vs_best_wrong_margin.astype(float).replace([np.inf, -np.inf], np.nan).mean()
                ),
                "oracle_geometry_eligible_rate": float(
                    group.oracle_geometry_eligible.astype(float).mean()
                ),
            })
            metric_rows.append(point)
            if method == "selected_full":
                scene = cluster_bootstrap_mean(
                    group.exact_match.to_numpy(dtype=float),
                    group.scene_token.astype(str).tolist(),
                    repetitions,
                    seed=910000 + protocol_index * 100,
                )
                instance = cluster_bootstrap_mean(
                    group.exact_match.to_numpy(dtype=float),
                    group.instance_token.astype(str).tolist(),
                    repetitions,
                    seed=920000 + protocol_index * 100,
                )
                cluster_rows.extend([
                    {"protocol": protocol, "cluster": "scene_token", **scene},
                    {"protocol": protocol, "cluster": "instance_token", **instance},
                ])
                flags = p2a_protocol_gate(
                    point,
                    scene,
                    instance,
                    min_exact_recall=float(gate["min_exact_recall"]),
                    min_cluster_ci_low=float(gate["min_scene_cluster_ci_low"]),
                    max_wrong_match_rate=float(gate["max_wrong_match_rate"]),
                )
                # The frozen scene and instance thresholds are currently equal,
                # but keep an explicit check so a later config edit cannot silently
                # reuse the scene threshold for the instance gate.
                flags["instance_cluster_pass"] = bool(
                    float(instance["ci_low"]) > float(gate["min_instance_cluster_ci_low"])
                )
                flags["protocol_pass"] = bool(
                    flags["exact_recall_pass"]
                    and flags["scene_cluster_pass"]
                    and flags["instance_cluster_pass"]
                    and flags["wrong_match_pass"]
                )
                gate_rows.append({"protocol": protocol, **flags})

    metric_frame = pd.DataFrame(metric_rows)
    cluster_frame = pd.DataFrame(cluster_rows)
    gate_frame = pd.DataFrame(gate_rows)
    metric_frame.to_csv(REPORT / "p2a_val_metrics.csv", index=False)
    cluster_frame.to_csv(REPORT / "p2a_val_cluster_ci.csv", index=False)
    gate_frame.to_csv(REPORT / "p2a_val_gate_summary.csv", index=False)

    protocol_pass = {
        protocol: bool(
            gate_frame[gate_frame.protocol.astype(str) == protocol]
            .protocol_pass.astype(bool).all()
        )
        for protocol in PROTOCOLS
    }
    passing = [protocol for protocol, value in protocol_pass.items() if value]
    go = len(passing) >= int(gate["min_passing_fault_families"])
    decision = (
        "GO_CARE3D_P2A_ONLINE_QUERY_ASSOCIATION"
        if go else "NO_GO_CARE3D_P2A_ONLINE_QUERY_ASSOCIATION"
    )
    result = {
        "schema_version": SCHEMA,
        "decision": decision,
        "protocol_pass": protocol_pass,
        "passing_protocols": passing,
        "required_passing_fault_families": int(gate["min_passing_fault_families"]),
        "bootstrap_repetitions": repetitions,
        "selected": selection["selected"],
        "probe_train_scenes": 419,
        "probe_val_scenes": 133,
        "probe_test_read": False,
        "association_neural_training": False,
        "P2A1_status": "ELIGIBLE" if go else "LOCKED_P2A0_NO_GO",
        "recommended_next_step": (
            "P2A1_FROZEN_ONLINE_ROUTER_EVALUATION"
            if go else "REVIEW_ASSOCIATION_FAILURE_WITHOUT_OPENING_PROBE_TEST"
        ),
    }
    atomic_json(REPORT / "decision.json", result)
    progress["status"] = decision
    progress["stages"]["probe_val_analysis"] = "COMPLETE"
    progress["stages"]["P2A1"] = result["P2A1_status"]
    progress["probe_test_read"] = False
    atomic_json(progress_path, progress)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
