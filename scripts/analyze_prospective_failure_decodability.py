#!/usr/bin/env python3
"""Fit frozen linear probes only after complete prospective extraction."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from analysis.prospective_failure_decodability import (
    classification_metrics, clustered_metric_differences, train_standardize,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/full_nuscenes/prospective_failure_decodability"
PROTOCOLS = ("blur_back", "crash_back", "dark_back")
TAPS = ("temporal_alignment_query_state", "decoder_layer5_temporal_self_attn_output",
        "final_decoder_pre_cls_query")
PROBES = ("observable",) + TAPS
BOOTSTRAPS = 5000
SCHEMA = 2


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    default=lambda item: item.item() if isinstance(item, np.generic) else str(item)) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields: fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def coverage(validation):
    manifest = pd.read_csv(REPORT / "frozen_scene_manifest.csv")
    complete = {}
    rows = []
    for marker in (REPORT / "incremental/P0").glob("*.complete.json"):
        value = json.loads(marker.read_text())
        if value.get("complete") and value.get("schema_version") == SCHEMA \
                and value.get("scene_manifest_sha256") == validation["scene_manifest_sha256"]:
            complete[value["scene_token"]] = value
    for split, group in manifest.groupby("split"):
        expected = set(group.scene_token.astype(str)); observed = expected & set(complete)
        rows.append({"split": split, "completed_scenes": len(observed), "expected_scenes": len(expected),
                     "missing_scene_count": len(expected - observed),
                     "complete": observed == expected})
    atomic_csv(REPORT / "coverage.csv", rows)
    return manifest, complete, all(row["complete"] for row in rows)


def write_partial(manifest, complete):
    lines = ["# PARTIAL STATUS", "", "`PARTIAL_INSUFFICIENT_COVERAGE`", "",
             "Frozen prospective extraction is incomplete; no linear probe or Go/No-Go has been run.", "",
             "| split | scenes | expected |", "|---|---:|---:|"]
    for split in ("probe_train", "probe_val", "probe_test"):
        expected = set(manifest[manifest.split == split].scene_token.astype(str))
        lines.append(f"| {split} | {len(expected & set(complete))} | {len(expected)} |")
    samples = {protocol: sum(int(value["samples_by_protocol"][protocol]) for value in complete.values())
               for protocol in PROTOCOLS}
    positives = {protocol: sum(int(value["positives_by_protocol"][protocol]) for value in complete.values())
                 for protocol in PROTOCOLS}
    lines += ["", f"Current samples: `{samples}`", "", f"Current positives: `{positives}`", "",
              "Resume:", "", "```bash", "python scripts/run_prospective_failure_features.py",
              "python scripts/analyze_prospective_failure_decodability.py", "```", ""]
    temporary = REPORT / "PARTIAL_STATUS.md.tmp"; temporary.write_text("\n".join(lines)); os.replace(temporary, REPORT / "PARTIAL_STATUS.md")
    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    progress["status"] = "PARTIAL_INSUFFICIENT_COVERAGE"; atomic_json(REPORT / "progress_manifest.json", progress)


def load_complete(manifest):
    metadata, arrays = [], {probe: [] for probe in PROBES}
    for scene in manifest.scene_token.astype(str):
        prefix = REPORT / "incremental/P0" / scene
        frame = pd.read_csv(prefix.with_suffix(".samples.csv"))
        packed = np.load(prefix.with_suffix(".features.npz"))
        if len(frame) != len(packed["label"]) or not np.array_equal(
                frame.y_tp_to_fn.to_numpy(np.int8), packed["label"]):
            raise RuntimeError(f"metadata/feature label mismatch: {scene}")
        if len(frame):
            metadata.append(frame)
            for probe in PROBES: arrays[probe].append(np.asarray(packed[probe]))
    metadata = pd.concat(metadata, ignore_index=True)
    return metadata, {probe: np.concatenate(values) for probe, values in arrays.items()}


def main():
    validation = json.loads((REPORT / "source_validation.json").read_text())
    manifest, complete, ready = coverage(validation)
    if not ready:
        write_partial(manifest, complete)
        print(json.dumps({"status": "PARTIAL_INSUFFICIENT_COVERAGE", "complete_scenes": len(complete),
                          "expected_scenes": len(manifest)}, indent=2)); return
    metadata, arrays = load_complete(manifest)
    metadata.to_csv(REPORT / "sample_manifest.csv", index=False)
    probabilities = {protocol: {} for protocol in PROTOCOLS}
    model_rows = []
    for protocol in PROTOCOLS:
        protocol_mask = metadata.protocol.to_numpy() == protocol
        train_mask = protocol_mask & (metadata.split.to_numpy() == "probe_train")
        if np.unique(metadata.loc[train_mask, "y_tp_to_fn"]).size != 2:
            raise RuntimeError(f"training labels lack both classes: {protocol}")
        for probe in PROBES:
            x_train = arrays[probe][train_mask]
            x_all = arrays[probe][protocol_mask]
            train_z, all_z, mean, scale = train_standardize(x_train, x_all)
            model = LogisticRegression(penalty="l2", C=1., solver="lbfgs", fit_intercept=True,
                                       class_weight=None, max_iter=2000, tol=1e-6, random_state=2026)
            model.fit(train_z, metadata.loc[train_mask, "y_tp_to_fn"].to_numpy(int))
            probabilities[protocol][probe] = model.predict_proba(all_z)[:, 1]
            model_dir = REPORT / "probe_models"; model_dir.mkdir(exist_ok=True)
            np.savez_compressed(model_dir / f"{protocol}.{probe}.npz", coefficient=model.coef_,
                                intercept=model.intercept_, mean=mean, scale=scale,
                                n_iter=model.n_iter_)
            model_rows.append({"protocol": protocol, "probe": probe, "dimension": x_train.shape[1],
                               "train_rows": len(x_train), "train_positives": int(metadata.loc[train_mask, "y_tp_to_fn"].sum()),
                               "iterations": int(model.n_iter_[0]), "converged": int(model.n_iter_[0]) < 2000})
    atomic_csv(REPORT / "probe_fit_manifest.csv", model_rows)

    prediction_rows, metric_rows, ci_rows = [], [], []
    for protocol_index, protocol in enumerate(PROTOCOLS):
        indexes = np.flatnonzero(metadata.protocol.to_numpy() == protocol)
        frame = metadata.iloc[indexes].copy().reset_index(drop=True)
        for probe in PROBES:
            frame[f"{probe}_probability"] = probabilities[protocol][probe]
        prediction_rows.extend(frame.to_dict("records"))
        for split in ("probe_val", "probe_test"):
            selected = frame[frame.split == split]
            for probe in PROBES:
                result = classification_metrics(selected.y_tp_to_fn, selected[f"{probe}_probability"])
                metric_rows.append({"protocol": protocol, "split": split, "probe": probe,
                                    "rows": len(selected), "positives": int(selected.y_tp_to_fn.sum()),
                                    "positive_scenes": selected.loc[selected.y_tp_to_fn == 1, "scene_token"].nunique(),
                                    "negative_scenes": selected.loc[selected.y_tp_to_fn == 0, "scene_token"].nunique(), **result})
        test = frame[frame.split == "probe_test"].copy()
        for tap_index, tap in enumerate(TAPS):
            result = clustered_metric_differences(
                test, f"{tap}_probability", n_boot=BOOTSTRAPS,
                seed=737373 + protocol_index * 100 + tap_index)
            ci_rows.append({"protocol": protocol, "tap": tap, **result})
    atomic_csv(REPORT / "per_sample_predictions.csv", prediction_rows)
    atomic_csv(REPORT / "metrics.csv", metric_rows)
    atomic_csv(REPORT / "p0_cluster_ci.csv", ci_rows)

    metric_frame, ci_frame = pd.DataFrame(metric_rows), pd.DataFrame(ci_rows)
    gate_rows = []
    for protocol in PROTOCOLS:
        test_observable = metric_frame[(metric_frame.protocol == protocol) &
            (metric_frame.split == "probe_test") & (metric_frame.probe == "observable")].iloc[0]
        coverage_pass = (test_observable.positives >= 25 and test_observable.positive_scenes >= 8
                         and test_observable.rows - test_observable.positives >= 200
                         and test_observable.negative_scenes >= 20)
        for tap in TAPS:
            row = ci_frame[(ci_frame.protocol == protocol) & (ci_frame.tap == tap)].iloc[0]
            passes = bool(coverage_pass and row.delta_auprc_ci_low > 0 and row.delta_auroc_ci_low > 0
                          and row.auprc_minus_base_rate_ci_low > 0 and row.auprc_minus_base_rate >= .02)
            gate_rows.append({"protocol": protocol, "tap": tap, "coverage_pass": coverage_pass,
                              "delta_auprc_pass": row.delta_auprc_ci_low > 0,
                              "delta_auroc_pass": row.delta_auroc_ci_low > 0,
                              "above_random_pass": row.auprc_minus_base_rate_ci_low > 0 and row.auprc_minus_base_rate >= .02,
                              "P0_protocol_tap_pass": passes})
    atomic_csv(REPORT / "p0_gate_summary.csv", gate_rows)
    gate_frame = pd.DataFrame(gate_rows)
    passing = [tap for tap in TAPS if gate_frame[gate_frame.tap == tap].P0_protocol_tap_pass.all()]
    go = bool(passing)
    decision = "GO_PROSPECTIVE_SELF_AWARE_3D_P0" if go else "NO_GO_PROSPECTIVE_SELF_AWARE_3D"
    atomic_json(REPORT / "decision.json", {"decision": decision, "passing_taps": passing,
                "P0_complete": True, "P1_status": "ELIGIBLE" if go else "LOCKED_P0_FAILED",
                "P2_status": "LOCKED_PENDING_P1" if go else "LOCKED_P0_FAILED",
                "P3_status": "LOCKED_PENDING_P0_P2" if go else "LOCKED_P0_FAILED"})
    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    progress["status"] = "P0_" + decision; progress["stages"]["P0_probe"] = "COMPLETE"
    progress["stages"]["P1"] = "ELIGIBLE" if go else "LOCKED_P0_FAILED"
    progress["stages"]["P2"] = "LOCKED_PENDING_P1" if go else "LOCKED_P0_FAILED"
    progress["stages"]["P3"] = "LOCKED_PENDING_P0_P2" if go else "LOCKED_P0_FAILED"
    atomic_json(REPORT / "progress_manifest.json", progress)
    if not go: (REPORT / "PARTIAL_STATUS.md").unlink(missing_ok=True)
    print(json.dumps({"decision": decision, "passing_taps": passing}, indent=2))


if __name__ == "__main__":
    main()
