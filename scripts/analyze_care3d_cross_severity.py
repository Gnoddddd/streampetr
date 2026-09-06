#!/usr/bin/env python3
"""Evaluate frozen CARE-3D P0 checkpoints on lighter Blur/Dark severity.

No fitting, calibration, threshold selection, or checkpoint selection is allowed
in this script.  It reuses all three already-frozen main-P0 seed checkpoints
and the exact main-P0 clean-anchor predictor inputs from probe_test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.care3d_cross_severity import (
    MAIN_PROTOCOL_INDEX,
    TRANSFER_PROTOCOLS,
    assert_exact_sample_alignment,
    transfer_gate_flags,
)
from scripts.analyze_care3d_p0 import (
    boundary_metrics,
    clustered_bootstrap,
    predict,
    regression_metrics,
    sigmoid,
)


ROOT = Path(__file__).resolve().parents[1]
MAIN_REPORT = ROOT / "reports/care3d/p0_counterfactual_vulnerability"
REPORT = ROOT / "reports/care3d/p0_cross_severity"
CONFIG = ROOT / "configs/care3d/p0_counterfactual_vulnerability.py"
SEEDS = (42, 2027, 2028)
SCHEMA = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstraps", type=int)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_sources() -> dict:
    source_path = REPORT / "source_validation.json"
    progress_path = REPORT / "progress_manifest.json"
    if not source_path.exists() or not progress_path.exists():
        raise RuntimeError("run scripts/export_care3d_cross_severity.py first")
    source = json.loads(source_path.read_text())
    if source.get("status") != "VALIDATED_BEFORE_CROSS_SEVERITY_FORWARD":
        raise RuntimeError("cross-severity source validation is not frozen")
    if source.get("retrain_predictor") is not False or source.get("recalibrate_predictor") is not False:
        raise RuntimeError("cross-severity transfer must not retrain/recalibrate CARE")
    if float(source.get("main_train_severity")) != 0.9 or float(source.get("transfer_severity")) != 0.3:
        raise RuntimeError("cross-severity severity identity changed")

    main_decision = json.loads((MAIN_REPORT / "decision.json").read_text())
    if main_decision.get("decision") != "GO_CARE3D_COUNTERFACTUAL_P0":
        raise RuntimeError("main CARE P0 GO is no longer present")
    if sha256(MAIN_REPORT / "decision.json") != source.get("main_p0_decision_sha256"):
        raise RuntimeError("main CARE P0 decision changed after transfer was frozen")

    for seed in SEEDS:
        checkpoint = MAIN_REPORT / "training" / f"seed_{seed}" / "best.pth"
        manifest = MAIN_REPORT / "training" / f"seed_{seed}" / "training_manifest.json"
        if not checkpoint.exists() or not manifest.exists():
            raise RuntimeError(f"missing frozen P0 checkpoint for seed {seed}")
        if sha256(checkpoint) != source["p0_checkpoint_sha256"][str(seed)]:
            raise RuntimeError(f"frozen P0 checkpoint changed for seed {seed}")
        meta = json.loads(manifest.read_text())
        if meta.get("status") != "TRAINING_COMPLETE_TEST_UNSEEN":
            raise RuntimeError(f"training provenance changed for seed {seed}")
        if meta.get("routing_enabled") is not False:
            raise RuntimeError(f"routing unexpectedly enabled in P0 seed {seed}")
    return source


def load_probe_test(source: dict):
    manifest = pd.read_csv(MAIN_REPORT / "frozen_scene_manifest.csv")
    scenes = manifest[manifest.split.astype(str) == "probe_test"].scene_token.astype(str).tolist()
    if len(scenes) != 132:
        raise RuntimeError(f"expected 132 frozen probe-test scenes, got {len(scenes)}")

    metadata = []
    main_arrays = {key: [] for key in (
        "object_features",
        "temporal_features",
        "decision_features",
        "camera_support",
        "camera_quality",
    )}
    labels = {key: [] for key in ("evidence_drop", "cross_topk", "tp_to_fn", "valid_mask")}

    for scene in scenes:
        transfer_prefix = REPORT / "incremental/probe_test" / scene
        marker_path = transfer_prefix.with_suffix(".complete.json")
        sample_path = transfer_prefix.with_suffix(".samples.csv")
        labels_path = transfer_prefix.with_suffix(".labels.npz")
        if not marker_path.exists() or not sample_path.exists() or not labels_path.exists():
            raise RuntimeError(f"missing cross-severity extraction: {scene}")
        marker = json.loads(marker_path.read_text())
        if not marker.get("complete"):
            raise RuntimeError(f"incomplete cross-severity scene: {scene}")
        if marker.get("schema_version") != SCHEMA:
            raise RuntimeError(f"cross-severity schema changed: {scene}")
        if marker.get("main_scene_manifest_sha256") != source["main_scene_manifest_sha256"]:
            raise RuntimeError(f"cross-severity scene-manifest mismatch: {scene}")
        if marker.get("protocol_sha256") != source["protocol_sha256"]:
            raise RuntimeError(f"cross-severity protocol mismatch: {scene}")
        if not marker.get("equivalence_pass") or not marker.get("predictor_input_identity_pass"):
            raise RuntimeError(f"cross-severity identity/equivalence failed: {scene}")

        transfer_frame = pd.read_csv(sample_path)
        main_prefix = MAIN_REPORT / "incremental/P0" / scene
        main_frame = pd.read_csv(main_prefix.with_suffix(".samples.csv"))
        assert_exact_sample_alignment(
            main_frame.sample_id.astype(str).tolist(),
            transfer_frame.sample_id.astype(str).tolist(),
        )
        with np.load(main_prefix.with_suffix(".features.npz")) as packed:
            for key in main_arrays:
                main_arrays[key].append(np.asarray(packed[key]).copy())
        with np.load(labels_path) as packed:
            n = len(transfer_frame)
            expected = {
                "evidence_drop": (n, 2),
                "cross_topk": (n, 2),
                "tp_to_fn": (n, 2),
                "valid_mask": (n, 2),
            }
            for key, shape in expected.items():
                value = np.asarray(packed[key])
                if value.shape != shape:
                    raise RuntimeError(f"{scene}: {key} shape {value.shape} != {shape}")
                labels[key].append(value.copy())
        metadata.append(
            transfer_frame[["sample_id", "scene_token", "instance_token"]].copy()
        )

    metadata_frame = pd.concat(metadata, ignore_index=True)
    inputs = {key: np.concatenate(value, axis=0) for key, value in main_arrays.items()}
    outcomes = {key: np.concatenate(value, axis=0) for key, value in labels.items()}
    n = len(metadata_frame)
    if n == 0 or len(inputs["object_features"]) != n or len(outcomes["evidence_drop"]) != n:
        raise RuntimeError("cross-severity probe-test arrays are empty/misaligned")
    return metadata_frame, inputs, outcomes


def main() -> None:
    args = parse_args()
    source = validate_sources()
    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    coverage = progress.get("stages", {}).get("probe_test_extraction")
    if not isinstance(coverage, dict):
        raise RuntimeError("cross-severity probe-test extraction has not started")
    if int(coverage.get("completed_scenes", -1)) != 132 or int(coverage.get("expected_scenes", -1)) != 132:
        raise RuntimeError("all 132 frozen probe-test scenes must be complete before analysis")

    cfg = runpy.run_path(str(CONFIG))
    gate = dict(cfg["gate"])
    repetitions = int(args.bootstraps or gate["bootstrap_repetitions"])
    if repetitions != 5000 and args.bootstraps is None:
        raise RuntimeError("frozen cross-severity analysis expects 5000 bootstraps")
    quantile = float(gate["vulnerability_quantile"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    metadata, inputs, outcomes = load_probe_test(source)
    metric_rows, bootstrap_rows, prediction_rows = [], [], []

    for seed_index, seed in enumerate(SEEDS):
        checkpoint = MAIN_REPORT / "training" / f"seed_{seed}" / "best.pth"
        predicted_drop, logits = predict(checkpoint, inputs, device, args.batch_size)
        probabilities = sigmoid(logits)

        for transfer_index, protocol in enumerate(TRANSFER_PROTOCOLS):
            model_index = int(MAIN_PROTOCOL_INDEX[protocol])
            mask = outcomes["valid_mask"][:, transfer_index].astype(bool)
            frame = metadata.loc[mask].reset_index(drop=True)
            actual = outcomes["evidence_drop"][mask, transfer_index]
            predicted = predicted_drop[mask, model_index]
            y = outcomes["cross_topk"][mask, transfer_index].astype(int)
            probability = probabilities[mask, model_index]

            r = regression_metrics(actual, predicted, quantile)
            b = boundary_metrics(y, probability)
            metric_rows.append({
                "seed": seed,
                "split": "probe_test",
                "protocol": protocol,
                "source_head": "blur_back" if model_index == 0 else "dark_back",
                "train_severity": 0.9,
                "transfer_severity": 0.3,
                "rows": len(frame),
                "positives": int(y.sum()),
                **r,
                **b,
            })

            for cluster_index, cluster in enumerate(("scene_token", "instance_token")):
                boot = clustered_bootstrap(
                    frame,
                    actual,
                    predicted,
                    y,
                    probability,
                    cluster,
                    repetitions,
                    seed=990000 + seed_index * 1000 + transfer_index * 100 + cluster_index,
                    quantile=quantile,
                )
                bootstrap_rows.append({
                    "seed": seed,
                    "protocol": protocol,
                    "train_severity": 0.9,
                    "transfer_severity": 0.3,
                    **boot,
                })

            prediction_rows.extend({
                "seed": seed,
                "protocol": protocol,
                "sample_id": frame.iloc[index].sample_id,
                "scene_token": frame.iloc[index].scene_token,
                "instance_token": frame.iloc[index].instance_token,
                "actual_drop": float(actual[index]),
                "predicted_drop": float(predicted[index]),
                "cross_topk": int(y[index]),
                "cross_probability": float(probability[index]),
            } for index in range(len(frame)))

    metric_frame = pd.DataFrame(metric_rows)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    metric_frame.to_csv(REPORT / "cross_severity_metrics.csv", index=False)
    bootstrap_frame.to_csv(REPORT / "cross_severity_cluster_ci.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(REPORT / "cross_severity_predictions.csv", index=False)

    gate_rows = []
    for protocol in TRANSFER_PROTOCOLS:
        for seed in SEEDS:
            point = metric_frame[
                (metric_frame.seed == seed) & (metric_frame.protocol == protocol)
            ].iloc[0].to_dict()
            scene = bootstrap_frame[
                (bootstrap_frame.seed == seed)
                & (bootstrap_frame.protocol == protocol)
                & (bootstrap_frame.cluster == "scene_token")
            ].iloc[0].to_dict()
            instance = bootstrap_frame[
                (bootstrap_frame.seed == seed)
                & (bootstrap_frame.protocol == protocol)
                & (bootstrap_frame.cluster == "instance_token")
            ].iloc[0].to_dict()
            flags = transfer_gate_flags(
                point,
                scene,
                instance,
                min_boundary_auroc=float(gate["min_boundary_auroc"]),
                min_boundary_auroc_ci_low=float(gate["min_boundary_auroc_ci_low"]),
                min_auprc_excess_ci_low=float(gate["min_auprc_excess_ci_low"]),
            )
            gate_rows.append({"protocol": protocol, "seed": seed, **flags})

    gate_frame = pd.DataFrame(gate_rows)
    gate_frame.to_csv(REPORT / "cross_severity_gate_summary.csv", index=False)
    protocol_pass = {
        protocol: bool(
            gate_frame[gate_frame.protocol == protocol].seed_protocol_pass.all()
        )
        for protocol in TRANSFER_PROTOCOLS
    }
    passed_protocols = [protocol for protocol, passed in protocol_pass.items() if passed]
    passed = all(protocol_pass.values())
    decision = (
        "PASS_CARE3D_P0_CROSS_SEVERITY"
        if passed else "FAIL_CARE3D_P0_CROSS_SEVERITY"
    )
    result = {
        "decision": decision,
        "main_p0_decision": "GO_CARE3D_COUNTERFACTUAL_P0",
        "main_p1_status": "ELIGIBLE_FROM_MAIN_P0",
        "retrained": False,
        "recalibrated": False,
        "train_severity": 0.9,
        "transfer_severity": 0.3,
        "protocol_pass": protocol_pass,
        "passing_protocols": passed_protocols,
        "seeds": list(SEEDS),
        "bootstrap_repetitions": repetitions,
        "probe_test_only": True,
        "predictor_input_identity_required": True,
        "recommended_next_step": (
            "CARE3D_P1_SPARSE_EVIDENCE_ROUTER"
            if passed else "REVIEW_CROSS_SEVERITY_FAILURE_BEFORE_P1"
        ),
    }
    atomic_json(REPORT / "decision.json", result)

    progress["status"] = decision
    progress["stages"]["analysis"] = "COMPLETE"
    progress["cross_severity_pass"] = passed
    atomic_json(REPORT / "progress_manifest.json", progress)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
