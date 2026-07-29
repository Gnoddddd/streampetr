#!/usr/bin/env python3
"""Analyze the frozen C0/C1 200-iteration confirmation."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from mmcv import Config


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/stage2/s2_4_baseline_confirmation"
REPORT = ROOT / "reports/stage2/s2_4_baseline_confirmation"
INIT = (
    ROOT
    / "outputs/final_snapshots/stage1_ternary_r50_200/checkpoint/iter_200.pth"
)
MODES = {
    "c0": {
        "name": "canonical_no_discount",
        "config": ROOT
        / "configs/evidence_conserving/mini_stage2_canonical_no_discount_200.py",
    },
    "c1": {
        "name": "legacy_fixed_discount",
        "config": ROOT
        / "configs/evidence_conserving/mini_stage2_legacy_fixed_discount_200.py",
    },
}
PROTOCOLS = ("clean", "crash5", "crash10", "compound")
FAULT_PROTOCOLS = ("crash5", "crash10", "compound")
SINGLE_PROTOCOL_TOLERANCE = 0.001


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_summary(mode: str) -> Dict[str, Any]:
    log_path = OUTPUT / f"{mode}_200" / "console.log"
    text = log_path.read_text(errors="ignore").replace("\r", "\n")
    losses = [float(value) for value in re.findall(r"\bloss: ([0-9.eE+-]+)", text)]
    grads = [
        float(value)
        for value in re.findall(r"\bgrad_norm: ([0-9.eE+-]+)", text)
    ]
    memories = [int(value) for value in re.findall(r"\bmemory: ([0-9]+)", text)]
    trace = read_trace(
        OUTPUT / f"{mode}_200" / "stage2_ledger_train_trace.jsonl"
    )
    return {
        "candidate": mode,
        "candidate_name": MODES[mode]["name"],
        "iterations": len(trace),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "loss_max": max(losses),
        "loss_mean": float(np.mean(losses)),
        "grad_norm_first": grads[0],
        "grad_norm_last": grads[-1],
        "grad_norm_max": max(grads),
        "grad_norm_mean": float(np.mean(grads)),
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
        "oom_or_runtime_error": bool(
            re.search(r"out of memory|RuntimeError", text, re.I)
        ),
        "exit_status": int(
            (
                OUTPUT / f"{mode}_200/run_meta/exit_status.txt"
            ).read_text().strip()
        ),
    }


def metric_rows() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []
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
            effective = np.concatenate(
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
            unsupported = sum(
                int(
                    np.asarray(
                        record["diagnostics"]["unsupported_growth"],
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
            raw[mode][protocol] = {
                "mAP": float(metrics["mean_ap"]),
                "NDS": float(metrics["nd_score"]),
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
                    "conservation_residual_abs_max": float(
                        np.abs(residuals).max()
                    ),
                    "conservation_violation_count": conservation_violations,
                    "unsupported_growth_count": unsupported,
                    "source_mass_violation_count": source_violations,
                }
            )
            quantiles = np.quantile(
                effective, [0.05, 0.25, 0.5, 0.75, 0.95]
            )
            action_rows.append(
                {
                    "candidate": mode,
                    "candidate_name": meta["name"],
                    "protocol": protocol,
                    "records": len(trace),
                    "queries": int(actions.size),
                    "neff_min": float(effective.min()),
                    "neff_p05": float(quantiles[0]),
                    "neff_p25": float(quantiles[1]),
                    "neff_mean": float(effective.mean()),
                    "neff_median": float(quantiles[2]),
                    "neff_p75": float(quantiles[3]),
                    "neff_p95": float(quantiles[4]),
                    "neff_max": float(effective.max()),
                    "neff_zero_ratio": float((effective == 0).mean()),
                    "neff_one_ratio": float((effective == 1).mean()),
                    "keep_count": int((actions == 0).sum()),
                    "keep_ratio": float((actions == 0).mean()),
                    "recover_count": int((actions == 1).sum()),
                    "recover_ratio": float((actions == 1).mean()),
                    "defer_count": int((actions == 2).sum()),
                    "defer_ratio": float((actions == 2).mean()),
                    "write_count": int(writes.sum()),
                    "write_ratio": float(writes.mean()),
                }
            )

    for mode, meta in MODES.items():
        rows.append(
            {
                "candidate": mode,
                "candidate_name": meta["name"],
                "protocol": "fault_average",
                "mAP": float(
                    np.mean([raw[mode][p]["mAP"] for p in FAULT_PROTOCOLS])
                ),
                "NDS": float(
                    np.mean([raw[mode][p]["NDS"] for p in FAULT_PROTOCOLS])
                ),
            }
        )
    c0 = {row["protocol"]: row for row in rows if row["candidate"] == "c0"}
    for row in rows:
        if row["candidate"] == "c0":
            row["delta_mAP_vs_c0"] = 0.0
            row["delta_NDS_vs_c0"] = 0.0
        else:
            row["delta_mAP_vs_c0"] = float(row["mAP"]) - float(
                c0[row["protocol"]]["mAP"]
            )
            row["delta_NDS_vs_c0"] = float(row["NDS"]) - float(
                c0[row["protocol"]]["NDS"]
            )
    return rows, action_rows


def checkpoint_rows() -> List[Dict[str, Any]]:
    runtime_fragments = (
        "evidence_ledger.alpha",
        "evidence_ledger.beta",
        "evidence_ledger.source_evidence",
        "evidence_ledger.provenance",
        "evidence_ledger.age",
        "evidence_ledger.effective_count",
        "evidence_ledger.observability",
        "evidence_ledger.novelty",
        "evidence_ledger.action",
        "conservation_residual",
    )
    rows: List[Dict[str, Any]] = []
    for mode, meta in MODES.items():
        path = OUTPUT / f"{mode}_200/iter_200.pth"
        payload = torch.load(path, map_location="cpu")
        state = payload.get("state_dict", payload)
        runtime = [
            key
            for key in state
            if any(fragment in key for fragment in runtime_fragments)
        ]
        switch = [
            key for key in state if "enable_correlation_discount" in key
        ]
        correlation = [
            key for key in state if "camera_correlation" in key
        ]
        rows.append(
            {
                "candidate": mode,
                "candidate_name": meta["name"],
                "checkpoint": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "state_dict_keys": len(state),
                "runtime_state_key_count": len(runtime),
                "runtime_state_keys": ";".join(runtime),
                "switch_state_key_count": len(switch),
                "correlation_config_key_count": len(correlation),
                "safe": not runtime and not switch,
            }
        )
    return rows


def manifest_rows(training: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    init_hash = sha256(INIT)
    c0 = Config.fromfile(str(MODES["c0"]["config"]))
    c1 = Config.fromfile(str(MODES["c1"]["config"]))
    fairness_fields = (
        "data",
        "optimizer",
        "lr_config",
        "fp16",
        "optimizer_config",
        "seed",
        "load_from",
        "runner",
        "custom_hooks",
    )
    equality = all(c0.get(key) == c1.get(key) for key in fairness_fields)
    training_by_mode = {row["candidate"]: row for row in training}
    rows: List[Dict[str, Any]] = []
    for mode, meta in MODES.items():
        rows.append(
            {
                "candidate": mode,
                "candidate_name": meta["name"],
                "config": str(meta["config"].relative_to(ROOT)),
                "enable_correlation_discount": mode == "c1",
                "initial_checkpoint": str(INIT),
                "initial_checkpoint_sha256": init_hash,
                "seed": 2026,
                "max_iters": 200,
                "fairness_fields_equal": equality,
                "only_intended_semantic_difference": (
                    "enable_correlation_discount"
                ),
                "train_exit_status": training_by_mode[mode]["exit_status"],
                "evaluation_exit_status": 0,
            }
        )
    return rows


def gate_result(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    lookup = {
        (row["candidate"], row["protocol"]): row for row in metrics
    }
    clean_map = (
        float(lookup[("c0", "clean")]["mAP"])
        >= float(lookup[("c1", "clean")]["mAP"])
    )
    clean_nds = (
        float(lookup[("c0", "clean")]["NDS"])
        >= float(lookup[("c1", "clean")]["NDS"])
    )
    fault_map = (
        float(lookup[("c0", "fault_average")]["mAP"])
        >= float(lookup[("c1", "fault_average")]["mAP"])
    )
    fault_nds = (
        float(lookup[("c0", "fault_average")]["NDS"])
        >= float(lookup[("c1", "fault_average")]["NDS"])
    )
    individual = all(
        float(lookup[("c0", protocol)]["mAP"])
        - float(lookup[("c1", protocol)]["mAP"])
        >= -SINGLE_PROTOCOL_TOLERANCE
        and float(lookup[("c0", protocol)]["NDS"])
        - float(lookup[("c1", protocol)]["NDS"])
        >= -SINGLE_PROTOCOL_TOLERANCE
        for protocol in FAULT_PROTOCOLS
    )
    engineering = all(
        int(row.get("conservation_violation_count") or 0) == 0
        and int(row.get("unsupported_growth_count") or 0) == 0
        and int(row.get("source_mass_violation_count") or 0) == 0
        for row in metrics
    )
    return {
        "clean_map_noninferior": clean_map,
        "clean_nds_noninferior": clean_nds,
        "fault_average_map_noninferior": fault_map,
        "fault_average_nds_noninferior": fault_nds,
        "individual_fault_tolerance_pass": individual,
        "engineering_invariants_pass": engineering,
        "overall_pass": (
            clean_map
            and clean_nds
            and fault_map
            and fault_nds
            and individual
            and engineering
        ),
    }


def main() -> None:
    training = [training_summary(mode) for mode in MODES]
    metrics, actions = metric_rows()
    checkpoints = checkpoint_rows()
    manifest = manifest_rows(training)
    gate = gate_result(metrics)
    write_csv(REPORT / "experiment_manifest.csv", manifest)
    write_csv(REPORT / "training_summary.csv", training)
    write_csv(REPORT / "per_protocol_metrics.csv", metrics)
    write_csv(REPORT / "neff_action_summary.csv", actions)
    write_csv(REPORT / "checkpoint_audit.csv", checkpoints)
    (REPORT / "gate_result.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, sort_keys=True))


if __name__ == "__main__":
    main()
