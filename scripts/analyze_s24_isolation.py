#!/usr/bin/env python3
"""Compare S2.2, historical implicit N_eff, and the true disabled path."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import mmcv
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
INFERENCE = ROOT / "outputs/stage2/s2_4_isolation_audit/inference"
S22 = ROOT / "outputs/stage2/s2_2_source_ledger_debug_50"
REPORT = ROOT / "reports/stage2/s2_4_isolation_audit"

PROTOCOLS = {
    "clean": "fixed_v3_stage2_clean",
    "crash5": "fixed_v3_stage2_camera_crash_back_5f",
    "crash10": "fixed_v3_stage2_camera_crash_back_10f",
    "compound": "fixed_v3_stage2_compound_fog_crash_10f",
}

NUMERIC_DIAGNOSTICS = (
    "alpha",
    "beta",
    "actual_added_positive_evidence",
    "actual_added_negative_evidence",
    "source_evidence",
    "previous_action",
    "action",
    "write_mask",
    "conservation_residual",
    "source_mass_residual",
    "topk_indexes",
)

HASH_DIAGNOSTICS = (
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
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def compare_values(left: Any, right: Any) -> Tuple[bool, float, int]:
    if hasattr(left, "tensor") and hasattr(right, "tensor"):
        return compare_values(left.tensor, right.tensor)
    if torch.is_tensor(left) or torch.is_tensor(right):
        left_tensor = torch.as_tensor(left).detach().cpu()
        right_tensor = torch.as_tensor(right).detach().cpu()
        if left_tensor.shape != right_tensor.shape:
            return False, math.inf, 0
        equal = torch.equal(left_tensor, right_tensor)
        if left_tensor.numel() == 0:
            return equal, 0.0, 0
        if left_tensor.dtype == torch.bool:
            return equal, float((left_tensor != right_tensor).any()), int(
                left_tensor.numel()
            )
        difference = (
            left_tensor.to(torch.float64)
            - right_tensor.to(torch.float64)
        ).abs()
        return equal, float(difference.max()), int(left_tensor.numel())
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return compare_values(torch.as_tensor(left), torch.as_tensor(right))
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False, math.inf, 0
        exact = True
        maximum = 0.0
        count = 0
        for key in sorted(left):
            same, difference, values = compare_values(left[key], right[key])
            exact &= same
            maximum = max(maximum, difference)
            count += values
        return exact, maximum, count
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False, math.inf, 0
        exact = True
        maximum = 0.0
        count = 0
        for left_item, right_item in zip(left, right):
            same, difference, values = compare_values(left_item, right_item)
            exact &= same
            maximum = max(maximum, difference)
            count += values
        return exact, maximum, count
    if isinstance(left, (int, float, bool)) and isinstance(
        right, (int, float, bool)
    ):
        difference = abs(float(left) - float(right))
        return left == right, difference, 1
    return left == right, 0.0 if left == right else math.inf, 0


def read_trace(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compare_traces(
    left: List[Dict[str, Any]],
    right: List[Dict[str, Any]],
    protocol: str,
) -> List[Dict[str, Any]]:
    if len(left) != len(right):
        raise RuntimeError(
            f"trace length mismatch for {protocol}: "
            f"{len(left)} != {len(right)}"
        )
    rows: List[Dict[str, Any]] = []
    for key in NUMERIC_DIAGNOSTICS:
        exact = True
        maximum = 0.0
        values = 0
        mismatch_frames = 0
        for left_record, right_record in zip(left, right):
            left_value = left_record["diagnostics"].get(key)
            right_value = right_record["diagnostics"].get(key)
            if left_value is None or right_value is None:
                exact = False
                maximum = math.inf
                mismatch_frames += 1
                continue
            same, difference, count = compare_values(
                left_value,
                right_value,
            )
            exact &= same
            maximum = max(maximum, difference)
            values += count
            mismatch_frames += int(not same)
        rows.append(
            {
                "comparison": "legacy_vs_disabled",
                "protocol": protocol,
                "component": key,
                "records": len(left),
                "values": values,
                "exact": exact,
                "max_abs_diff": maximum,
                "mismatch_frames": mismatch_frames,
            }
        )
    for key in HASH_DIAGNOSTICS:
        mismatches = sum(
            left_record["diagnostics"].get(key)
            != right_record["diagnostics"].get(key)
            for left_record, right_record in zip(left, right)
        )
        rows.append(
            {
                "comparison": "legacy_vs_disabled",
                "protocol": protocol,
                "component": key,
                "records": len(left),
                "values": len(left),
                "exact": mismatches == 0,
                "max_abs_diff": "" if mismatches == 0 else "hash_mismatch",
                "mismatch_frames": mismatches,
            }
        )
    return rows


def evidence_summary(
    mode: str,
    protocol: str,
    trace: List[Dict[str, Any]],
) -> Dict[str, Any]:
    residuals: List[float] = []
    source_residuals: List[float] = []
    conservation_violations = 0
    source_violations = 0
    unsupported = 0
    for record in trace:
        diagnostics = record["diagnostics"]
        residual = np.asarray(
            diagnostics["conservation_residual"],
            dtype=np.float64,
        )
        source_residual = np.asarray(
            diagnostics["source_mass_residual"],
            dtype=np.float64,
        )
        residuals.extend(residual.reshape(-1).tolist())
        source_residuals.extend(source_residual.reshape(-1).tolist())
        conservation_violations += int(
            np.asarray(
                diagnostics["conservation_violation_mask"],
                dtype=np.bool_,
            ).sum()
        )
        source_violations += int(
            np.asarray(
                diagnostics["source_mass_violation"],
                dtype=np.bool_,
            ).sum()
        )
        unsupported += int(
            np.asarray(
                diagnostics["unsupported_growth"],
                dtype=np.bool_,
            ).sum()
        )
    return {
        "mode": mode,
        "protocol": protocol,
        "records": len(trace),
        "queries": len(residuals),
        "conservation_residual_abs_max": max(
            (abs(value) for value in residuals),
            default=0.0,
        ),
        "conservation_violation_count": conservation_violations,
        "unsupported_growth_count": unsupported,
        "source_mass_residual_abs_max": max(
            (abs(value) for value in source_residuals),
            default=0.0,
        ),
        "source_mass_violation_count": source_violations,
    }


def checkpoint_audit() -> List[Dict[str, Any]]:
    checkpoint_path = S22 / "iter_50.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    runtime_fragments = (
        "evidence_ledger.alpha",
        "evidence_ledger.beta",
        "evidence_ledger.source_evidence",
        "evidence_ledger.action",
        "conservation_residual",
    )
    runtime_keys = [
        key
        for key in state_dict
        if any(fragment in key for fragment in runtime_fragments)
    ]
    correlation_keys = [
        key for key in state_dict if "camera_correlation" in key
    ]
    switch_keys = [
        key for key in state_dict if "enable_correlation_discount" in key
    ]
    return [
        {
            "checkpoint": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "state_dict_keys": len(state_dict),
            "legacy_load_completed_inference_runs": 4,
            "disabled_load_completed_inference_runs": 4,
            "correlation_config_key_count": len(correlation_keys),
            "correlation_config_keys": ";".join(correlation_keys),
            "switch_state_key_count": len(switch_keys),
            "runtime_state_key_count": len(runtime_keys),
            "runtime_state_keys": ";".join(runtime_keys),
            "safe": not switch_keys and not runtime_keys,
        }
    ]


def main() -> None:
    invariance_rows: List[Dict[str, Any]] = []
    evidence_rows: List[Dict[str, Any]] = []
    for protocol, historical_name in PROTOCOLS.items():
        historical_predictions = mmcv.load(
            S22
            / "eval"
            / historical_name
            / "predictions.pkl"
        )
        legacy_predictions = mmcv.load(
            INFERENCE / "legacy" / protocol / "predictions.pkl"
        )
        disabled_predictions = mmcv.load(
            INFERENCE / "disabled" / protocol / "predictions.pkl"
        )
        for comparison, left, right in (
            (
                "s2_2_stable_vs_legacy",
                historical_predictions,
                legacy_predictions,
            ),
            (
                "s2_2_stable_vs_disabled",
                historical_predictions,
                disabled_predictions,
            ),
        ):
            exact, maximum, values = compare_values(left, right)
            invariance_rows.append(
                {
                    "comparison": comparison,
                    "protocol": protocol,
                    "component": "final_prediction",
                    "records": len(left),
                    "values": values,
                    "exact": exact,
                    "max_abs_diff": maximum,
                    "mismatch_frames": "",
                }
            )

        legacy_trace = read_trace(
            INFERENCE / "legacy" / protocol / "trace.jsonl"
        )
        disabled_trace = read_trace(
            INFERENCE / "disabled" / protocol / "trace.jsonl"
        )
        invariance_rows.extend(
            compare_traces(legacy_trace, disabled_trace, protocol)
        )
        evidence_rows.append(
            evidence_summary("legacy", protocol, legacy_trace)
        )
        evidence_rows.append(
            evidence_summary("disabled", protocol, disabled_trace)
        )

    write_csv(REPORT / "prediction_invariance.csv", invariance_rows)
    write_csv(REPORT / "checkpoint_audit.csv", checkpoint_audit())
    write_csv(REPORT / "evidence_invariants.csv", evidence_rows)
    exact_rows = [
        row for row in invariance_rows
        if str(row["exact"]).lower() == "true"
    ]
    numeric_differences = [
        float(row["max_abs_diff"])
        for row in invariance_rows
        if isinstance(row["max_abs_diff"], (int, float))
        and math.isfinite(float(row["max_abs_diff"]))
    ]
    print(
        f"wrote {len(invariance_rows)} comparisons; "
        f"{len(exact_rows)} exact; "
        f"max_abs_diff={max(numeric_differences, default=0.0)}"
    )


if __name__ == "__main__":
    main()
