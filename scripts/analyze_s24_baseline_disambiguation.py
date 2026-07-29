#!/usr/bin/env python3
"""Summarize the frozen C0/C1 50-iteration disambiguation experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mmcv
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/stage2/s2_4_baseline_disambiguation"
REPORT = ROOT / "reports/stage2/s2_4_baseline_disambiguation"
S22 = ROOT / "outputs/stage2/s2_2_source_ledger_debug_50/eval"

MODES = {
    "c0": {
        "name": "canonical_no_discount",
        "config": "configs/evidence_conserving/mini_stage2_canonical_no_discount_50.py",
    },
    "c1": {
        "name": "legacy_fixed_discount",
        "config": "configs/evidence_conserving/mini_stage2_legacy_fixed_discount_50.py",
    },
}
PROTOCOLS = {
    "clean": "fixed_v3_stage2_clean",
    "crash5": "fixed_v3_stage2_camera_crash_back_5f",
    "crash10": "fixed_v3_stage2_camera_crash_back_10f",
    "compound": "fixed_v3_stage2_compound_fog_crash_10f",
}
HASH_FIELDS = (
    "classification_sha256",
    "box_sha256",
    "decoder_sha256",
    "propagated_query_sha256",
    "temporal_memory_embedding_sha256",
    "temporal_memory_reference_point_sha256",
    "temporal_memory_velocity_sha256",
)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_trace(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compare_values(left: Any, right: Any) -> Tuple[bool, float, int]:
    if hasattr(left, "tensor") and hasattr(right, "tensor"):
        return compare_values(left.tensor, right.tensor)
    if torch.is_tensor(left) or torch.is_tensor(right):
        a = torch.as_tensor(left).detach().cpu()
        b = torch.as_tensor(right).detach().cpu()
        if a.shape != b.shape:
            return False, math.inf, 0
        if a.numel() == 0:
            return torch.equal(a, b), 0.0, 0
        difference = (a.to(torch.float64) - b.to(torch.float64)).abs()
        return torch.equal(a, b), float(difference.max()), int(a.numel())
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return compare_values(torch.as_tensor(left), torch.as_tensor(right))
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False, math.inf, 0
        exact, maximum, count = True, 0.0, 0
        for key in sorted(left):
            same, difference, values = compare_values(left[key], right[key])
            exact &= same
            maximum = max(maximum, difference)
            count += values
        return exact, maximum, count
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False, math.inf, 0
        exact, maximum, count = True, 0.0, 0
        for a, b in zip(left, right):
            same, difference, values = compare_values(a, b)
            exact &= same
            maximum = max(maximum, difference)
            count += values
        return exact, maximum, count
    if isinstance(left, (int, float, bool)) and isinstance(
        right, (int, float, bool)
    ):
        return left == right, abs(float(left) - float(right)), 1
    return left == right, 0.0 if left == right else math.inf, 0


def metric_and_trace_rows() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    neff_rows: List[Dict[str, Any]] = []
    raw: Dict[str, Dict[str, Dict[str, float]]] = {}
    for mode, meta in MODES.items():
        raw[mode] = {}
        for protocol in PROTOCOLS:
            run_dir = OUTPUT / "eval" / mode / protocol
            metrics = json.load(
                (
                    run_dir
                    / "nuscenes_results/pts_bbox/metrics_summary.json"
                ).open()
            )
            trace = read_trace(run_dir / "trace.jsonl")
            neff = np.concatenate(
                [
                    np.asarray(
                        record["diagnostics"]["effective_count"],
                        dtype=np.float64,
                    ).reshape(-1)
                    for record in trace
                ]
            )
            actions = np.concatenate(
                [
                    np.asarray(
                        record["diagnostics"]["action"], dtype=np.int64
                    ).reshape(-1)
                    for record in trace
                ]
            )
            writes = np.concatenate(
                [
                    np.asarray(
                        record["diagnostics"]["write_mask"], dtype=np.bool_
                    ).reshape(-1)
                    for record in trace
                ]
            )
            residuals = np.concatenate(
                [
                    np.asarray(
                        record["diagnostics"]["conservation_residual"],
                        dtype=np.float64,
                    ).reshape(-1)
                    for record in trace
                ]
            )
            conservation_violations = sum(
                int(
                    np.asarray(
                        record["diagnostics"][
                            "conservation_violation_mask"
                        ],
                        dtype=np.bool_,
                    ).sum()
                )
                for record in trace
            )
            source_violations = sum(
                int(
                    np.asarray(
                        record["diagnostics"]["source_mass_violation"],
                        dtype=np.bool_,
                    ).sum()
                )
                for record in trace
            )
            unsupported = sum(
                int(
                    np.asarray(
                        record["diagnostics"]["unsupported_growth"],
                        dtype=np.bool_,
                    ).sum()
                )
                for record in trace
            )
            raw[mode][protocol] = {
                "map": float(metrics["mean_ap"]),
                "nds": float(metrics["nd_score"]),
            }
            rows.append(
                {
                    "candidate": mode,
                    "candidate_name": meta["name"],
                    "protocol": protocol,
                    "mAP": metrics["mean_ap"],
                    "NDS": metrics["nd_score"],
                    "records": len(trace),
                    "queries": int(actions.size),
                    "keep_count": int((actions == 0).sum()),
                    "recover_count": int((actions == 1).sum()),
                    "defer_count": int((actions == 2).sum()),
                    "write_count": int(writes.sum()),
                    "write_ratio": float(writes.mean()),
                    "conservation_residual_abs_max": float(
                        np.abs(residuals).max()
                    ),
                    "conservation_violation_count": conservation_violations,
                    "unsupported_growth_count": unsupported,
                    "source_mass_violation_count": source_violations,
                }
            )
            quantiles = np.quantile(
                neff, [0.05, 0.25, 0.5, 0.75, 0.95]
            )
            neff_rows.append(
                {
                    "candidate": mode,
                    "candidate_name": meta["name"],
                    "protocol": protocol,
                    "records": len(trace),
                    "queries": int(neff.size),
                    "min": float(neff.min()),
                    "p05": float(quantiles[0]),
                    "p25": float(quantiles[1]),
                    "mean": float(neff.mean()),
                    "median": float(quantiles[2]),
                    "p75": float(quantiles[3]),
                    "p95": float(quantiles[4]),
                    "max": float(neff.max()),
                    "zero_ratio": float((neff == 0).mean()),
                    "one_ratio": float((neff == 1).mean()),
                    "greater_than_one_ratio": float((neff > 1).mean()),
                }
            )

    for mode, meta in MODES.items():
        fault_maps = [raw[mode][p]["map"] for p in PROTOCOLS if p != "clean"]
        fault_nds = [raw[mode][p]["nds"] for p in PROTOCOLS if p != "clean"]
        rows.append(
            {
                "candidate": mode,
                "candidate_name": meta["name"],
                "protocol": "fault_average",
                "mAP": float(np.mean(fault_maps)),
                "NDS": float(np.mean(fault_nds)),
            }
        )

    c0 = {row["protocol"]: row for row in rows if row["candidate"] == "c0"}
    for row in rows:
        if row["candidate"] == "c0":
            row["delta_mAP_vs_c0"] = 0.0
            row["delta_NDS_vs_c0"] = 0.0
        else:
            baseline = c0[row["protocol"]]
            row["delta_mAP_vs_c0"] = float(row["mAP"]) - float(
                baseline["mAP"]
            )
            row["delta_NDS_vs_c0"] = float(row["NDS"]) - float(
                baseline["NDS"]
            )
    return rows, neff_rows


def prediction_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for protocol, historical_name in PROTOCOLS.items():
        c0 = mmcv.load(OUTPUT / "eval" / "c0" / protocol / "predictions.pkl")
        c1 = mmcv.load(OUTPUT / "eval" / "c1" / protocol / "predictions.pkl")
        historical = mmcv.load(
            S22 / historical_name / "predictions.pkl"
        )
        for comparison, left, right in (
            ("c0_50_vs_c1_50", c0, c1),
            ("s2_2_stable_vs_c1_50", historical, c1),
        ):
            exact, maximum, values = compare_values(left, right)
            rows.append(
                {
                    "comparison": comparison,
                    "protocol": protocol,
                    "component": "final_prediction",
                    "records": len(left),
                    "values": values,
                    "exact": exact,
                    "max_abs_diff": maximum,
                }
            )
        c0_trace = read_trace(OUTPUT / "eval" / "c0" / protocol / "trace.jsonl")
        c1_trace = read_trace(OUTPUT / "eval" / "c1" / protocol / "trace.jsonl")
        for field in HASH_FIELDS:
            mismatch = sum(
                a["diagnostics"].get(field) != b["diagnostics"].get(field)
                for a, b in zip(c0_trace, c1_trace)
            )
            rows.append(
                {
                    "comparison": "c0_50_vs_c1_50",
                    "protocol": protocol,
                    "component": field,
                    "records": len(c0_trace),
                    "values": len(c0_trace),
                    "exact": mismatch == 0,
                    "max_abs_diff": "" if mismatch == 0 else "hash_mismatch",
                    "mismatch_records": mismatch,
                }
            )
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_summary(mode: str) -> Dict[str, Any]:
    log_path = OUTPUT / f"{mode}_50" / "console.log"
    text = log_path.read_text(errors="ignore").replace("\r", "\n")
    losses = [float(value) for value in re.findall(r"\bloss: ([0-9.eE+-]+)", text)]
    grads = [
        float(value)
        for value in re.findall(r"\bgrad_norm: ([0-9.eE+-]+)", text)
    ]
    memories = [int(value) for value in re.findall(r"\bmemory: ([0-9]+)", text)]
    trace = read_trace(OUTPUT / f"{mode}_50" / "stage2_ledger_train_trace.jsonl")
    return {
        "iterations": len(trace),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "loss_max": max(losses),
        "grad_norm_first": grads[0],
        "grad_norm_last": grads[-1],
        "grad_norm_max": max(grads),
        "memory_max_mb": max(memories),
        "conservation_residual_abs_max": max(
            float(record["conservation_residual_abs_max"]) for record in trace
        ),
        "conservation_violation_count": sum(
            int(record["conservation_violation_count"]) for record in trace
        ),
        "unsupported_growth_count": sum(
            int(record["unsupported_growth_count"]) for record in trace
        ),
        "source_mass_violation_count": sum(
            int(record["source_mass_violation_count"]) for record in trace
        ),
        "keep_count": sum(int(record["keep_count"]) for record in trace),
        "recover_count": sum(int(record["recover_count"]) for record in trace),
        "defer_count": sum(int(record["defer_count"]) for record in trace),
        "nan_or_inf": bool(
            re.search(r"(?<![A-Za-z])(?:nan|inf)(?![A-Za-z])", text, re.I)
        ),
    }


def manifest_rows() -> List[Dict[str, Any]]:
    init = (
        ROOT
        / "outputs/final_snapshots/stage1_ternary_r50_200/checkpoint/iter_200.pth"
    )
    init_hash = sha256(init)
    rows: List[Dict[str, Any]] = []
    for mode, meta in MODES.items():
        checkpoint = OUTPUT / f"{mode}_50" / "iter_50.pth"
        train = training_summary(mode)
        rows.append(
            {
                "candidate": mode,
                "candidate_name": meta["name"],
                "phase": "train_50",
                "config": meta["config"],
                "initial_checkpoint": str(init),
                "initial_checkpoint_sha256": init_hash,
                "output_checkpoint": str(checkpoint),
                "output_checkpoint_sha256": sha256(checkpoint),
                "seed": 2026,
                "max_iters": 50,
                "exit_status": 0,
                **train,
            }
        )
        rows.append(
            {
                "candidate": mode,
                "candidate_name": meta["name"],
                "phase": "smoke_2",
                "config": meta["config"].replace(
                    "_50.py", "_smoke2.py"
                ),
                "initial_checkpoint": str(init),
                "initial_checkpoint_sha256": init_hash,
                "seed": 2026,
                "max_iters": 2,
                "exit_status": 0,
            }
        )
        rows.append(
            {
                "candidate": mode,
                "candidate_name": meta["name"],
                "phase": "zero_shot_clean",
                "config": meta["config"],
                "initial_checkpoint": str(init),
                "initial_checkpoint_sha256": init_hash,
                "seed": 2026,
                "max_iters": 0,
                "exit_status": 0,
            }
        )
        rows.append(
            {
                "candidate": mode,
                "candidate_name": meta["name"],
                "phase": "evaluation_4_protocols",
                "config": meta["config"],
                "initial_checkpoint": str(checkpoint),
                "initial_checkpoint_sha256": sha256(checkpoint),
                "seed": 2026,
                "max_iters": 0,
                "exit_status": 0,
            }
        )
    return rows


def main() -> None:
    metrics, neff = metric_and_trace_rows()
    predictions = prediction_rows()
    manifest = manifest_rows()
    write_csv(REPORT / "experiment_manifest.csv", manifest)
    write_csv(REPORT / "per_protocol_metrics.csv", metrics)
    write_csv(REPORT / "neff_summary.csv", neff)
    write_csv(REPORT / "prediction_invariance.csv", predictions)
    fault = [
        row
        for row in metrics
        if row["candidate"] == "c1" and row["protocol"] == "fault_average"
    ][0]
    exact_legacy = sum(
        row["exact"]
        for row in predictions
        if row["comparison"] == "s2_2_stable_vs_c1_50"
    )
    print(
        "wrote disambiguation reports; "
        f"C1-C0 fault mAP={fault['delta_mAP_vs_c0']:.9f}; "
        f"NDS={fault['delta_NDS_vs_c0']:.9f}; "
        f"legacy_exact_protocols={exact_legacy}/4"
    )


if __name__ == "__main__":
    main()
