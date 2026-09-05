#!/usr/bin/env python3
"""Evaluate frozen CARE-3D P0 predictors and apply the preregistered gate."""

from __future__ import annotations

import argparse
import json
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from analysis.care3d_counterfactual import PROTOCOLS
from models.care3d import CARE3DCore


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/care3d/p0_counterfactual_vulnerability"
CONFIG = ROOT / "configs/care3d/p0_counterfactual_vulnerability.py"
SEEDS = (42, 2027, 2028)
SCHEMA = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstraps", type=int)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_split(split: str):
    if split not in {"probe_val", "probe_test"}:
        raise ValueError("analysis split must be probe_val/probe_test")
    manifest = pd.read_csv(REPORT / "frozen_scene_manifest.csv")
    scenes = manifest[manifest.split == split].scene_token.astype(str).tolist()
    metadata, arrays = [], {key: [] for key in (
        "object_features", "temporal_features", "decision_features", "camera_support",
        "camera_quality", "evidence_drop", "cross_topk", "valid_mask",
    )}
    for scene in scenes:
        prefix = REPORT / "incremental/P0" / scene
        marker = prefix.with_suffix(".complete.json")
        if not marker.exists() or not prefix.with_suffix(".features.npz").exists():
            raise RuntimeError(f"missing formal extraction for {split}/{scene}")
        meta = json.loads(marker.read_text())
        if not meta.get("complete") or meta.get("schema_version") != SCHEMA:
            raise RuntimeError(f"invalid completion marker: {scene}")
        frame = pd.read_csv(prefix.with_suffix(".samples.csv"))
        packed = np.load(prefix.with_suffix(".features.npz"))
        if len(frame) != len(packed["object_features"]):
            raise RuntimeError(f"metadata/feature misalignment: {scene}")
        metadata.append(frame[["sample_id", "scene_token", "instance_token"]])
        for key in arrays:
            arrays[key].append(np.asarray(packed[key]))
    metadata = pd.concat(metadata, ignore_index=True)
    return metadata, {key: np.concatenate(values, axis=0) for key, values in arrays.items()}


def sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def ece10(y, probability):
    y = np.asarray(y, dtype=float)
    p = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, 11)
    total = max(len(y), 1)
    value = 0.0
    for index in range(10):
        lower, upper = edges[index], edges[index + 1]
        mask = (p >= lower) & (p < upper if index < 9 else p <= upper)
        if not mask.any():
            continue
        value += float(mask.sum()) / total * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return value


def regression_metrics(actual, predicted, quantile=0.10):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) < 2:
        return {"spearman": np.nan, "pearson": np.nan, "mae": np.nan, "decile_drop_delta": np.nan}
    spearman = float(spearmanr(actual, predicted).statistic)
    pearson = float(pearsonr(actual, predicted).statistic) if np.std(actual) > 0 and np.std(predicted) > 0 else np.nan
    lower = np.quantile(predicted, quantile)
    upper = np.quantile(predicted, 1.0 - quantile)
    bottom, top = actual[predicted <= lower], actual[predicted >= upper]
    decile = float(top.mean() - bottom.mean()) if len(top) and len(bottom) else np.nan
    return {"spearman": spearman, "pearson": pearson,
            "mae": float(np.abs(actual - predicted).mean()), "decile_drop_delta": decile}


def boundary_metrics(y, probability):
    y = np.asarray(y, dtype=int)
    p = np.asarray(probability, dtype=float)
    base = float(y.mean()) if len(y) else np.nan
    if len(np.unique(y)) < 2:
        auroc, auprc = np.nan, np.nan
    else:
        auroc = float(roc_auc_score(y, p))
        auprc = float(average_precision_score(y, p))
    return {
        "positive_base_rate": base,
        "auroc": auroc,
        "auprc": auprc,
        "auprc_minus_base_rate": auprc - base if np.isfinite(auprc) else np.nan,
        "brier": float(brier_score_loss(y, p)) if len(y) else np.nan,
        "ece10": ece10(y, p),
    }


def metric_vector(actual_drop, predicted_drop, y, probability, quantile):
    r = regression_metrics(actual_drop, predicted_drop, quantile)
    b = boundary_metrics(y, probability)
    return np.asarray([r["spearman"], r["decile_drop_delta"], b["auroc"],
                       b["auprc_minus_base_rate"]], dtype=float)


def clustered_bootstrap(frame, actual_drop, predicted_drop, y, probability, cluster_col,
                        repetitions, seed, quantile):
    labels = frame[cluster_col].astype(str).to_numpy()
    clusters = np.unique(labels)
    by_cluster = {cluster: np.flatnonzero(labels == cluster) for cluster in clusters}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        indexes = np.concatenate([by_cluster[value] for value in sampled])
        vector = metric_vector(actual_drop[indexes], predicted_drop[indexes], y[indexes],
                               probability[indexes], quantile)
        values.append(vector)
    packed = np.asarray(values, dtype=float)
    names = ("spearman", "decile_drop_delta", "auroc", "auprc_minus_base_rate")
    output = {"cluster": cluster_col, "bootstrap_repetitions": repetitions}
    for index, name in enumerate(names):
        finite = packed[:, index][np.isfinite(packed[:, index])]
        output[f"{name}_finite_bootstraps"] = int(len(finite))
        output[f"{name}_ci_low"] = float(np.quantile(finite, 0.025)) if len(finite) else np.nan
        output[f"{name}_ci_high"] = float(np.quantile(finite, 0.975)) if len(finite) else np.nan
    return output


def predict(checkpoint, arrays, device, batch_size):
    saved = torch.load(checkpoint, map_location="cpu")
    if saved.get("routing_enabled"):
        raise RuntimeError(f"P0 checkpoint unexpectedly enabled routing: {checkpoint}")
    model = CARE3DCore(**saved["model_config"]).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    predicted_drop, crossing_logits = [], []
    n = len(arrays["object_features"])
    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            def tensor(key):
                return torch.from_numpy(np.asarray(arrays[key][start:stop])).float().to(device).unsqueeze(1)
            output = model(
                object_features=tensor("object_features"),
                camera_support=tensor("camera_support"),
                camera_quality=tensor("camera_quality"),
                temporal_features=tensor("temporal_features"),
                decision_features=tensor("decision_features"),
            )
            predicted_drop.append(output["vulnerability"][:, 0].cpu().numpy())
            crossing_logits.append(output["boundary_crossing_logits"][:, 0].cpu().numpy())
    return np.concatenate(predicted_drop), np.concatenate(crossing_logits)


def main() -> None:
    args = parse_args()
    cfg = runpy.run_path(str(CONFIG))
    gate = dict(cfg["gate"])
    repetitions = int(args.bootstraps or gate["bootstrap_repetitions"])
    quantile = float(gate["vulnerability_quantile"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    for seed in SEEDS:
        manifest_path = REPORT / "training" / f"seed_{seed}" / "training_manifest.json"
        checkpoint_path = REPORT / "training" / f"seed_{seed}" / "best.pth"
        if not manifest_path.exists() or not checkpoint_path.exists():
            raise RuntimeError(f"seed {seed} is not frozen/trained")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("probe_test_read") is not False or manifest.get("status") != "TRAINING_COMPLETE_TEST_UNSEEN":
            raise RuntimeError(f"seed {seed} training manifest violates test-blind rule")

    split_data = {split: load_split(split) for split in ("probe_val", "probe_test")}
    metric_rows, bootstrap_rows, prediction_rows = [], [], []

    for seed_index, seed in enumerate(SEEDS):
        checkpoint = REPORT / "training" / f"seed_{seed}" / "best.pth"
        for split_index, split in enumerate(("probe_val", "probe_test")):
            metadata, arrays = split_data[split]
            predicted_drop, logits = predict(checkpoint, arrays, device, args.batch_size)
            probabilities = sigmoid(logits)
            valid = arrays["valid_mask"].astype(bool)
            for protocol_index, protocol in enumerate(PROTOCOLS):
                mask = valid[:, protocol_index]
                frame = metadata.loc[mask].reset_index(drop=True)
                actual = arrays["evidence_drop"][mask, protocol_index]
                predicted = predicted_drop[mask, protocol_index]
                y = arrays["cross_topk"][mask, protocol_index].astype(int)
                probability = probabilities[mask, protocol_index]
                r = regression_metrics(actual, predicted, quantile)
                b = boundary_metrics(y, probability)
                metric_rows.append({
                    "seed": seed, "split": split, "protocol": protocol, "rows": len(frame),
                    "positives": int(y.sum()), **r, **b,
                })
                if split == "probe_test":
                    for cluster_index, cluster in enumerate(("scene_token", "instance_token")):
                        boot = clustered_bootstrap(
                            frame, actual, predicted, y, probability, cluster, repetitions,
                            seed=880000 + seed_index * 1000 + protocol_index * 100 + cluster_index,
                            quantile=quantile,
                        )
                        bootstrap_rows.append({"seed": seed, "protocol": protocol, **boot})
                prediction_rows.extend({
                    "seed": seed, "split": split, "protocol": protocol,
                    "sample_id": frame.iloc[index].sample_id,
                    "scene_token": frame.iloc[index].scene_token,
                    "instance_token": frame.iloc[index].instance_token,
                    "actual_drop": float(actual[index]), "predicted_drop": float(predicted[index]),
                    "cross_topk": int(y[index]), "cross_probability": float(probability[index]),
                } for index in range(len(frame)))

    metric_frame = pd.DataFrame(metric_rows)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    metric_frame.to_csv(REPORT / "p0_metrics.csv", index=False)
    bootstrap_frame.to_csv(REPORT / "p0_cluster_ci.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(REPORT / "p0_predictions.csv", index=False)

    gate_rows = []
    for protocol in PROTOCOLS:
        for seed in SEEDS:
            test = metric_frame[(metric_frame.seed == seed) & (metric_frame.split == "probe_test") &
                                (metric_frame.protocol == protocol)].iloc[0]
            val = metric_frame[(metric_frame.seed == seed) & (metric_frame.split == "probe_val") &
                               (metric_frame.protocol == protocol)].iloc[0]
            scene = bootstrap_frame[(bootstrap_frame.seed == seed) &
                                    (bootstrap_frame.protocol == protocol) &
                                    (bootstrap_frame.cluster == "scene_token")].iloc[0]
            instance = bootstrap_frame[(bootstrap_frame.seed == seed) &
                                       (bootstrap_frame.protocol == protocol) &
                                       (bootstrap_frame.cluster == "instance_token")].iloc[0]
            rank_pass = bool(test.spearman > 0 and scene.spearman_ci_low > 0 and instance.spearman_ci_low > 0)
            separation_pass = bool(test.decile_drop_delta > 0 and scene.decile_drop_delta_ci_low > 0 and
                                   instance.decile_drop_delta_ci_low > 0)
            boundary_pass = bool(test.auroc >= float(gate["min_boundary_auroc"]) and
                                 scene.auprc_minus_base_rate_ci_low > 0 and
                                 instance.auprc_minus_base_rate_ci_low > 0)
            val_direction_pass = bool(val.spearman > 0 and val.decile_drop_delta > 0 and
                                      val.auroc >= 0.5 and val.auprc_minus_base_rate > 0)
            passed = rank_pass and separation_pass and boundary_pass and val_direction_pass
            gate_rows.append({
                "protocol": protocol, "seed": seed, "rank_pass": rank_pass,
                "separation_pass": separation_pass, "boundary_pass": boundary_pass,
                "val_test_direction_pass": val_direction_pass, "seed_protocol_pass": passed,
            })

    gate_frame = pd.DataFrame(gate_rows)
    gate_frame.to_csv(REPORT / "p0_gate_summary.csv", index=False)
    protocol_pass = {protocol: bool(gate_frame[gate_frame.protocol == protocol].seed_protocol_pass.all())
                     for protocol in PROTOCOLS}
    passing_protocols = [protocol for protocol, passed in protocol_pass.items() if passed]
    go = len(passing_protocols) >= int(gate["min_passing_fault_families"])
    decision = "GO_CARE3D_COUNTERFACTUAL_P0" if go else "NO_GO_CARE3D_COUNTERFACTUAL_P0"
    result = {
        "decision": decision,
        "passing_protocols": passing_protocols,
        "protocol_pass": protocol_pass,
        "seeds": list(SEEDS),
        "bootstrap_repetitions": repetitions,
        "P0_complete": True,
        "P1_status": "ELIGIBLE" if go else "LOCKED_P0_FAILED",
        "cross_severity_status": "ELIGIBLE_PENDING_PROTOCOL_CREATION" if go else "LOCKED_P0_FAILED",
    }
    atomic_json(REPORT / "decision.json", result)
    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    progress["status"] = decision
    progress["stages"]["analysis"] = "COMPLETE"
    progress["stages"]["P1"] = "ELIGIBLE" if go else "LOCKED_P0_FAILED"
    atomic_json(REPORT / "progress_manifest.json", progress)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
