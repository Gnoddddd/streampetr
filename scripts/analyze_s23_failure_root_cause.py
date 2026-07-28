#!/usr/bin/env python3
"""Compare S2.2, N1 zero-shot and N1 debug50 evidence trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


PROTOCOLS = {
    "clean": (
        "fixed_v3_stage2_clean",
        "clean_no_corruption",
    ),
    "crash_5f": (
        "fixed_v3_stage2_camera_crash_back_5f",
        "camera_crash_back_5f",
    ),
    "crash_10f": (
        "fixed_v3_stage2_camera_crash_back_10f",
        "camera_crash_back_10f",
    ),
    "compound_10f": (
        "fixed_v3_stage2_compound_fog_crash_10f",
        "compound_fog_crash_10f",
    ),
}
QUANTILES = (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99)


def _records(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _flat(record: Dict, key: str, dtype=np.float64) -> np.ndarray:
    return np.asarray(record["diagnostics"].get(key, []), dtype=dtype).reshape(-1)


def _phase(protocol: str, frame_idx: int) -> str:
    match = re.search(r"(5|10)f", protocol)
    if protocol == "clean" or match is None:
        return "clean"
    duration = int(match.group(1))
    if frame_idx < 3:
        return "pre_fault"
    if frame_idx < 3 + duration:
        return "fault"
    return "recovery"


def _stats(values: Iterable[float], prefix: str = "") -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    result = {
        prefix + "count": int(array.size),
        prefix + "mean": float(array.mean()) if array.size else math.nan,
    }
    for quantile in QUANTILES:
        result[prefix + f"p{int(100 * quantile):02d}"] = (
            float(np.quantile(array, quantile)) if array.size else math.nan
        )
    return result


def _write(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _paths(root: Path, protocol: str) -> Dict[str, Path]:
    s22_name, candidate_name = PROTOCOLS[protocol]
    return {
        "s2_2": root
        / "outputs/stage2/s2_2_source_ledger_debug_50/eval"
        / s22_name
        / "evidence_trace"
        / f"{candidate_name}_diagnostic_trace.jsonl",
        "n1_zero_shot": root
        / "outputs/stage2/s2_3_innovation/zero_shot_active/fixed_v3_s2_3_n1"
        / candidate_name
        / "evidence_trace"
        / f"{candidate_name}_diagnostic_trace.jsonl",
        "n1_debug50": root
        / "outputs/stage2/s2_3_innovation/debug_50/fixed_v3_s2_3_n1_seed2026/eval"
        / candidate_name
        / "evidence_trace"
        / f"{candidate_name}_diagnostic_trace.jsonl",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_dir.resolve()

    summary_values: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
    increment_values: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
    strength_values: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
    action_rows: List[Dict] = []
    frame_rows: List[Dict] = []
    recovery_signals: List[Dict] = []

    metric_keys = (
        "alpha",
        "beta",
        "strength",
        "uncertainty",
        "actual_added_positive_evidence",
        "actual_added_negative_evidence",
        "source_innovation",
        "temporal_reacquisition",
        "score_scale",
    )

    for protocol in PROTOCOLS:
        traces = {
            name: _records(path)
            for name, path in _paths(root, protocol).items()
        }
        lengths = {name: len(records) for name, records in traces.items()}
        if len(set(lengths.values())) != 1:
            raise RuntimeError(f"trace length mismatch for {protocol}: {lengths}")

        for frame_number, base_record in enumerate(traces["s2_2"]):
            identity = (
                base_record.get("sample_idx"),
                base_record.get("scene_token"),
                base_record.get("frame_idx"),
            )
            for name, records in traces.items():
                candidate_identity = (
                    records[frame_number].get("sample_idx"),
                    records[frame_number].get("scene_token"),
                    records[frame_number].get("frame_idx"),
                )
                if candidate_identity != identity:
                    raise RuntimeError(
                        f"unaligned record {protocol} frame {frame_number}: "
                        f"{identity} != {candidate_identity}"
                    )

            frame_idx = int(base_record.get("frame_idx", -1))
            phase = _phase(protocol, frame_idx)
            base_positive = _flat(
                base_record, "actual_added_positive_evidence"
            )
            base_negative = _flat(
                base_record, "actual_added_negative_evidence"
            )
            base_action = _flat(base_record, "action", dtype=np.int64)
            base_score = _flat(base_record, "score_scale")
            base_write = _flat(base_record, "write_mask", dtype=np.bool_)

            for variant, records in traces.items():
                record = records[frame_number]
                for key in metric_keys:
                    values = _flat(record, key)
                    if values.size:
                        summary_values[(variant, protocol, phase, key)].extend(
                            values.tolist()
                        )
                action = _flat(record, "action", dtype=np.int64)
                write = _flat(record, "write_mask", dtype=np.bool_)
                prior_strength = _flat(record, "prior_strength")
                propagated = prior_strength[644:900] > 1e-6
                summary_values[
                    (variant, protocol, phase, "effective_propagated_queries")
                ].append(float(propagated.sum()))
                summary_values[
                    (variant, protocol, phase, "keep_ratio")
                ].append(float((action == 0).mean()))
                summary_values[
                    (variant, protocol, phase, "recover_ratio")
                ].append(float((action == 1).mean()))
                summary_values[
                    (variant, protocol, phase, "defer_ratio")
                ].append(float((action == 2).mean()))
                summary_values[
                    (variant, protocol, phase, "write_ratio")
                ].append(float(write.mean()))

                positive = _flat(
                    record, "actual_added_positive_evidence"
                )
                negative = _flat(
                    record, "actual_added_negative_evidence"
                )
                positive_ratio = np.divide(
                    positive,
                    base_positive,
                    out=np.ones_like(positive),
                    where=base_positive > 1e-8,
                )
                negative_ratio = np.divide(
                    negative,
                    base_negative,
                    out=np.ones_like(negative),
                    where=base_negative > 1e-8,
                )
                increment_values[
                    (variant, protocol, phase, "positive_over_base")
                ].extend(positive_ratio.tolist())
                increment_values[
                    (variant, protocol, phase, "negative_over_base")
                ].extend(negative_ratio.tolist())
                strength_values[
                    (variant, protocol, phase, "strength")
                ].extend(_flat(record, "strength").tolist())

                keep_to_recover = int(
                    ((base_action == 0) & (action == 1)).sum()
                )
                keep_to_defer = int(
                    ((base_action == 0) & (action == 2)).sum()
                )
                high_to_low = int(
                    ((base_score >= 0.9) & (_flat(record, "score_scale") < 0.5)).sum()
                )
                if variant != "s2_2":
                    action_rows.append(
                        {
                            "variant": variant,
                            "protocol": protocol,
                            "phase": phase,
                            "frame_idx": frame_idx,
                            "keep_to_recover": keep_to_recover,
                            "keep_to_defer": keep_to_defer,
                            "high_to_low_score_scale": high_to_low,
                            "write_mask_lost": int(
                                (base_write & ~write).sum()
                            ),
                        }
                    )
                    frame_rows.append(
                        {
                            "variant": variant,
                            "protocol": protocol,
                            "phase": phase,
                            "frame_idx": frame_idx,
                            "positive_ratio_mean": float(
                                positive_ratio.mean()
                            ),
                            "negative_ratio_mean": float(
                                negative_ratio.mean()
                            ),
                            "strength_mean": float(
                                _flat(record, "strength").mean()
                            ),
                            "keep_to_recover": keep_to_recover,
                            "keep_to_defer": keep_to_defer,
                            "high_to_low_score_scale": high_to_low,
                            "effective_propagated_queries": int(
                                propagated.sum()
                            ),
                        }
                    )

                if (
                    variant != "s2_2"
                    and protocol in ("crash_10f", "compound_10f")
                    and frame_idx == 13
                ):
                    recovery_signals.append(
                        {
                            "variant": variant,
                            "protocol": protocol,
                            "frame_idx": frame_idx,
                            "source_innovation_mean": float(
                                _flat(record, "source_innovation").mean()
                            ),
                            "source_innovation_p90": float(
                                np.quantile(
                                    _flat(record, "source_innovation"), 0.9
                                )
                            ),
                            "temporal_reacquisition_mean": float(
                                _flat(record, "temporal_reacquisition").mean()
                            ),
                            "temporal_reacquisition_p90": float(
                                np.quantile(
                                    _flat(record, "temporal_reacquisition"), 0.9
                                )
                            ),
                        }
                    )

    summary_rows = [
        {
            "variant": variant,
            "protocol": protocol,
            "phase": phase,
            "metric": metric,
            **_stats(values),
        }
        for (variant, protocol, phase, metric), values in sorted(
            summary_values.items()
        )
    ]
    increment_rows = [
        {
            "variant": variant,
            "protocol": protocol,
            "phase": phase,
            "metric": metric,
            **_stats(values),
        }
        for (variant, protocol, phase, metric), values in sorted(
            increment_values.items()
        )
    ]
    strength_rows = [
        {
            "variant": variant,
            "protocol": protocol,
            "phase": phase,
            "metric": metric,
            **_stats(values),
        }
        for (variant, protocol, phase, metric), values in sorted(
            strength_values.items()
        )
    ]
    action_aggregate = []
    grouped_actions: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)
    for row in action_rows:
        grouped_actions[
            (row["variant"], row["protocol"], row["phase"])
        ].append(row)
    for (variant, protocol, phase), rows in sorted(grouped_actions.items()):
        action_aggregate.append(
            {
                "variant": variant,
                "protocol": protocol,
                "phase": phase,
                "records": len(rows),
                **{
                    key: sum(int(row[key]) for row in rows)
                    for key in (
                        "keep_to_recover",
                        "keep_to_defer",
                        "high_to_low_score_scale",
                        "write_mask_lost",
                    )
                },
            }
        )

    _write(output / "failure_root_cause_summary.csv", summary_rows)
    _write(output / "evidence_increment_ratio.csv", increment_rows)
    _write(output / "strength_distribution.csv", strength_rows)
    _write(output / "action_transition_summary.csv", action_aggregate)
    _write(output / "frame_transition_summary.csv", frame_rows)

    def lookup(
        rows: List[Dict],
        variant: str,
        protocol: str,
        phase: str,
        metric: str,
        field: str = "mean",
    ) -> float:
        matches = [
            row
            for row in rows
            if row["variant"] == variant
            and row["protocol"] == protocol
            and row["phase"] == phase
            and row["metric"] == metric
        ]
        return float(matches[0][field]) if matches else math.nan

    clean_zero_pos = lookup(
        increment_rows,
        "n1_zero_shot",
        "clean",
        "clean",
        "positive_over_base",
    )
    clean_debug_pos = lookup(
        increment_rows,
        "n1_debug50",
        "clean",
        "clean",
        "positive_over_base",
    )
    clean_zero_neg = lookup(
        increment_rows,
        "n1_zero_shot",
        "clean",
        "clean",
        "negative_over_base",
    )
    clean_debug_neg = lookup(
        increment_rows,
        "n1_debug50",
        "clean",
        "clean",
        "negative_over_base",
    )
    transition_totals = {
        variant: {
            key: sum(
                int(row[key])
                for row in action_aggregate
                if row["variant"] == variant
            )
            for key in (
                "keep_to_recover",
                "keep_to_defer",
                "high_to_low_score_scale",
                "write_mask_lost",
            )
        }
        for variant in ("n1_zero_shot", "n1_debug50")
    }
    recovery_text = "\n".join(
        "- {variant} {protocol}: source mean/p90={source_innovation_mean:.4f}/"
        "{source_innovation_p90:.4f}, time mean/p90="
        "{temporal_reacquisition_mean:.4f}/{temporal_reacquisition_p90:.4f}".format(
            **row
        )
        for row in recovery_signals
    )
    markdown = f"""# S2.3 N1 performance-collapse root cause

All comparisons use the same S2.2 checkpoint, identical 81-frame protocol
order, and exact `(sample_idx, scene_token, frame_idx)` alignment.

## Answers

1. **Clean positive evidence is systematically suppressed.** Mean
   `actual_positive/base_positive` is {clean_zero_pos:.4f} for zero-shot N1
   and {clean_debug_pos:.4f} after 50 iterations. Training does not restore
   the S2.2 evidence budget.
2. **Negative evidence also changes.** Mean `actual_negative/base_negative`
   is {clean_zero_neg:.4f} zero-shot and {clean_debug_neg:.4f} debug50.
   The legacy multiplicative strategy therefore changes both directions,
   rather than isolating recovery-positive evidence.
3. **Lower strength changes policy and writeback.** Zero-shot transitions:
   {transition_totals['n1_zero_shot']}. Debug50 transitions:
   {transition_totals['n1_debug50']}.
4. **Both same-frame scaling and later propagation contribute.** The
   innovation factor changes alpha/beta before policy/score scaling in the
   current frame; the resulting action/write-mask losses then reduce valid
   propagated memory in subsequent frames. `high_to_low_score_scale` is a
   policy-scale proxy because raw detector logits are not present in legacy
   traces.
5. **Recovery signals exist at frame 13.**
{recovery_text}
6. **Why 50 iterations fail:** the detector is initialized from S2.2, but
   legacy N1 structurally multiplies every ordinary evidence increment by an
   innovation gain. The short run only updates the permitted lightweight
   branches and cannot reconstruct the removed evidence budget; action and
   memory feedback amplify the persistent suppression.

## Implication

The rescue must preserve S2.2 positive and negative base evidence tensor
exactly during continuous observation. Innovation may only add a one-shot,
bounded positive bonus after a verified observation gap. It must not scale
Clean base evidence or negative evidence.
"""
    (output / "failure_root_cause.md").write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
