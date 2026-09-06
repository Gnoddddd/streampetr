#!/usr/bin/env python3
"""Final preregistered CARE-3D P1 recovery/no-harm gate."""

from __future__ import annotations

import argparse
import json
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.care3d_p1 import (
    PROTOCOLS,
    cluster_bootstrap_fp_inflation,
    cluster_bootstrap_mean,
    p1_gate_flags,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/care3d/p1_sparse_evidence_router"
CONFIG = ROOT / "configs/care3d/p1_sparse_evidence_router.py"
SEEDS = (42, 2027, 2028)
SCHEMA = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstraps", type=int)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def require_complete() -> tuple[dict, dict]:
    validation = json.loads((REPORT / "source_validation.json").read_text())
    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    coverage = progress.get("stages", {}).get("probe_test_evaluation")
    if not isinstance(coverage, dict) \
            or int(coverage.get("completed_scenes", -1)) != 132 \
            or int(coverage.get("expected_scenes", -1)) != 132:
        raise RuntimeError("all 132 frozen P1 probe-test scenes are required")
    if progress.get("status") not in (
        "P1_TEST_EVALUATION_COMPLETE_ANALYSIS_PENDING",
        "GO_CARE3D_P1_SPARSE_EVIDENCE_ROUTER",
        "NO_GO_CARE3D_P1_SPARSE_EVIDENCE_ROUTER",
    ):
        raise RuntimeError(f"P1 analysis is not eligible: {progress.get('status')}")
    return validation, progress


def load_evaluation(validation: dict):
    manifest = pd.read_csv(REPORT / "frozen_scene_manifest.csv")
    scenes = manifest[manifest.split.astype(str) == "probe_test"].scene_token.astype(str).tolist()
    objects, frames, clean_identity = [], [], []
    for scene in scenes:
        prefix = REPORT / "evaluation/probe_test" / scene
        marker_path = prefix.with_suffix(".complete.json")
        object_path = prefix.with_suffix(".objects.csv")
        frame_path = prefix.with_suffix(".frames.csv")
        if not marker_path.exists() or not object_path.exists() or not frame_path.exists():
            raise RuntimeError(f"missing P1 evaluation scene: {scene}")
        marker = json.loads(marker_path.read_text())
        if not marker.get("complete") or marker.get("schema_version") != SCHEMA:
            raise RuntimeError(f"invalid P1 evaluation marker: {scene}")
        if marker.get("scene_manifest_sha256") != validation["scene_manifest_sha256"]:
            raise RuntimeError(f"P1 evaluation cohort mismatch: {scene}")
        clean_identity.append(bool(marker.get("clean_identity_pass")))
        objects.append(pd.read_csv(object_path))
        frames.append(pd.read_csv(frame_path))
    return (
        pd.concat(objects, ignore_index=True),
        pd.concat(frames, ignore_index=True),
        bool(all(clean_identity)),
    )


def safe_mean(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def bootstrap_pack(group: pd.DataFrame, cluster: str, repetitions: int, seed: int) -> dict:
    result = {"cluster": cluster, "bootstrap_repetitions": repetitions}
    specifications = [
        ("lost_recovery", group[group.base_tp == 0], "lost_recovered"),
        ("net_tp_delta", group, "tp_delta"),
        ("cross_topk_recovery", group[group.base_cross_topk == 1], "cross_topk_recovered"),
        ("target_score_delta", group[group.base_cross_topk == 1], "target_score_delta"),
        ("retained_damage", group[group.base_tp == 1], "retained_damaged"),
    ]
    for offset, (name, frame, column) in enumerate(specifications):
        stats = cluster_bootstrap_mean(
            frame[column].to_numpy(dtype=float),
            frame[cluster].astype(str).tolist(),
            repetitions,
            seed + offset * 17,
        )
        result[f"{name}_estimate"] = stats["estimate"]
        result[f"{name}_ci_low"] = stats["ci_low"]
        result[f"{name}_ci_high"] = stats["ci_high"]
        result[f"{name}_finite_bootstraps"] = stats["finite_bootstraps"]
    return result


def main() -> None:
    args = parse_args()
    validation, progress = require_complete()
    cfg = runpy.run_path(str(CONFIG))
    gate = dict(cfg["gate"])
    repetitions = int(args.bootstraps or gate["bootstrap_repetitions"])
    if args.bootstraps is None and repetitions != 5000:
        raise RuntimeError("formal P1 analysis requires frozen 5000 bootstraps")
    objects, frames, clean_identity = load_evaluation(validation)
    metric_rows, cluster_rows, fp_rows, gate_rows = [], [], [], []

    for seed_index, seed in enumerate(SEEDS):
        for protocol_index, protocol in enumerate(PROTOCOLS):
            group = objects[
                (objects.seed.astype(int) == seed)
                & (objects.protocol.astype(str) == protocol)
            ].reset_index(drop=True)
            frame_group = frames[
                (frames.seed.astype(int) == seed)
                & (frames.protocol.astype(str) == protocol)
            ].reset_index(drop=True)
            if len(group) == 0 or len(frame_group) == 0:
                raise RuntimeError(f"empty P1 result for seed={seed}, protocol={protocol}")
            lost = group[group.base_tp == 0]
            retained = group[group.base_tp == 1]
            crossing = group[group.base_cross_topk == 1]
            if len(lost) == 0 or len(retained) == 0 or len(crossing) == 0:
                raise RuntimeError(
                    f"P1 gate cohort degenerate seed={seed}, protocol={protocol}: "
                    f"lost={len(lost)} retained={len(retained)} crossing={len(crossing)}"
                )
            base_fp = float(frame_group.base_fp.sum())
            patched_fp = float(frame_group.patched_fp.sum())
            fp_inflation = (patched_fp - base_fp) / max(base_fp, 1.0)
            point = {
                "seed": seed,
                "protocol": protocol,
                "rows": len(group),
                "lost_n": len(lost),
                "lost_recovered_n": int(lost.lost_recovered.sum()),
                "lost_recovery_rate": safe_mean(lost.lost_recovered),
                "retained_n": len(retained),
                "retained_damaged_n": int(retained.retained_damaged.sum()),
                "retained_damage_rate": safe_mean(retained.retained_damaged),
                "net_tp_delta": safe_mean(group.tp_delta),
                "net_tp_gain_count": int(group.tp_delta.sum()),
                "cross_topk_n": len(crossing),
                "cross_topk_recovered_n": int(crossing.cross_topk_recovered.sum()),
                "cross_topk_recovery_rate": safe_mean(crossing.cross_topk_recovered),
                "target_score_delta_on_cross": safe_mean(crossing.target_score_delta),
                "mean_risk_probability": safe_mean(group.risk_probability),
                "base_frame_tp": int(frame_group.base_tp.sum()),
                "patched_frame_tp": int(frame_group.patched_tp.sum()),
                "base_frame_fp": int(base_fp),
                "patched_frame_fp": int(patched_fp),
                "fp_inflation_rate": float(fp_inflation),
                "clean_identity_pass": bool(clean_identity),
            }
            metric_rows.append(point)

            scene = bootstrap_pack(
                group,
                "scene_token",
                repetitions,
                seed=810000 + seed_index * 1000 + protocol_index * 100,
            )
            instance = bootstrap_pack(
                group,
                "instance_token",
                repetitions,
                seed=820000 + seed_index * 1000 + protocol_index * 100,
            )
            cluster_rows.extend([
                {"seed": seed, "protocol": protocol, **scene},
                {"seed": seed, "protocol": protocol, **instance},
            ])
            fp_ci = cluster_bootstrap_fp_inflation(
                frame_group,
                repetitions,
                seed=830000 + seed_index * 1000 + protocol_index * 100,
            )
            fp_rows.append({"seed": seed, "protocol": protocol, **fp_ci})
            flags = p1_gate_flags(
                point,
                scene,
                instance,
                fp_ci,
                max_retained_damage_rate=float(gate["max_retained_damage_rate"]),
                max_retained_damage_ci_high=float(gate["max_retained_damage_ci_high"]),
                max_fp_inflation_rate=float(gate["max_fp_inflation_rate"]),
                max_fp_inflation_ci_high=float(gate["max_fp_inflation_ci_high"]),
            )
            gate_rows.append({"seed": seed, "protocol": protocol, **flags})

    metric_frame = pd.DataFrame(metric_rows)
    cluster_frame = pd.DataFrame(cluster_rows)
    fp_frame = pd.DataFrame(fp_rows)
    gate_frame = pd.DataFrame(gate_rows)
    metric_frame.to_csv(REPORT / "p1_metrics.csv", index=False)
    cluster_frame.to_csv(REPORT / "p1_cluster_ci.csv", index=False)
    fp_frame.to_csv(REPORT / "p1_fp_ci.csv", index=False)
    gate_frame.to_csv(REPORT / "p1_gate_summary.csv", index=False)

    protocol_pass = {
        protocol: bool(
            gate_frame[gate_frame.protocol.astype(str) == protocol]
            .seed_protocol_pass.astype(bool).all()
        )
        for protocol in PROTOCOLS
    }
    passing = [protocol for protocol, value in protocol_pass.items() if value]
    go = len(passing) >= int(gate["min_passing_fault_families"])
    decision = (
        "GO_CARE3D_P1_SPARSE_EVIDENCE_ROUTER"
        if go else "NO_GO_CARE3D_P1_SPARSE_EVIDENCE_ROUTER"
    )
    result = {
        "schema_version": SCHEMA,
        "decision": decision,
        "protocol_pass": protocol_pass,
        "passing_protocols": passing,
        "required_passing_fault_families": int(gate["min_passing_fault_families"]),
        "seeds": list(SEEDS),
        "bootstrap_repetitions": repetitions,
        "source_bank": ["CAM_BACK_LEFT", "CAM_BACK_RIGHT", "TEMPORAL_ANCHOR"],
        "top_k": 2,
        "classification_only": True,
        "regression_unchanged": True,
        "clean_identity_pass": bool(clean_identity),
        "probe_test_opened_only_after_training": True,
        "P2_status": "ELIGIBLE" if go else "LOCKED_P1_NO_GO",
        "recommended_next_step": (
            "P2_SYSTEM_ASSOCIATION_AND_CROSS_ARCHITECTURE"
            if go else "REVIEW_P1_FAILURE_WITHOUT_RETUNING_FROZEN_GATE"
        ),
    }
    atomic_json(REPORT / "decision.json", result)
    progress["status"] = decision
    progress["stages"]["analysis"] = "COMPLETE"
    progress["stages"]["P2"] = result["P2_status"]
    atomic_json(REPORT / "progress_manifest.json", progress)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
