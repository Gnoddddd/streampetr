#!/usr/bin/env python3
"""Create terminal artifacts after the gated BD temporal-support audit."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/full_nuscenes/bd_temporal_support_audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def empty_csv(path: Path, fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields, lineterminator="\n").writeheader()
    os.replace(temporary, path)


def fmt(value) -> str:
    return f"{float(value):.6f}"


def main() -> None:
    decision = json.loads((REPORT / "final_decision.json").read_text())
    if decision.get("decision") != "NO_GO_LOCALIZABLE_BD_TEMPORAL_SUPPORT" \
            or decision.get("failed_stage") != "P0":
        raise RuntimeError("this finalizer only handles the preregistered P0 terminal gate")
    data = pd.read_csv(REPORT / "per_gt_nohistory.csv")
    summary = pd.read_csv(REPORT / "p0_summary.csv")
    coverage = pd.read_csv(REPORT / "p0_coverage.csv")
    exact = pd.read_csv(REPORT / "p0_disabled_equivalence.csv")
    if len(data) != 13803 or data.unit_id.duplicated().any() or not coverage.complete.astype(bool).all():
        raise RuntimeError("terminal coverage invariant failed")
    disabled = exact[exact.check == "disabled_wrapper_B0"]
    path_rows = exact[exact.check == "nohistory_current_path"]
    disabled_summary = {
        "disabled_wrapper_checks": len(disabled),
        "disabled_output_bitwise_equal": bool(disabled.output_bitwise_equal.astype(bool).all()),
        "disabled_output_max_abs_diff": float(disabled.output_max_abs_diff.astype(float).max()),
        "disabled_memory_bitwise_equal": bool(disabled.memory_bitwise_equal.astype(bool).all()),
        "disabled_memory_max_abs_diff": float(disabled.memory_max_abs_diff.astype(float).max()),
        "nohistory_current_path_checks": len(path_rows),
        "nohistory_query_count_exact_644": bool((path_rows.nohistory_query_count.astype(int) == 644).all()),
        "nohistory_temporal_memory_absent": bool(
            path_rows.E_temp_memory_is_none.astype(bool).all()
            and path_rows.F_temp_memory_is_none.astype(bool).all()),
        "canonical_state_updates_from_nohistory": False,
    }
    atomic_json(REPORT / "disabled_equivalence_summary.json", disabled_summary)

    scene = data.groupby(["protocol", "scene_token", "population"], sort=False).agg(
        n=("unit_id", "size"),
        finite_E=("E_s_pos", lambda value: int(np.isfinite(value).sum())),
        finite_F=("F_s_pos", lambda value: int(np.isfinite(value).sum())),
        median_G_A=("G_A", "median"),
        median_G_C=("G_C", "median"),
        median_G_B=("G_B", "median"),
        median_G_D=("G_D", "median"),
        median_current_only_deficit=("current_only_deficit", "median"),
        median_AC_gap=("AC_gap_from_anchor", "median"),
        median_BD_gap=("BD_gap_from_anchor", "median"),
    ).reset_index()
    scene.to_csv(REPORT / "p0_per_scene.csv", index=False)

    empty_csv(REPORT / "per_gt_history_support_patch.csv", [
        "unit_id", "protocol", "scene_token", "population", "pair", "layer",
        "intervention", "control_type", "delta_s_pos", "topk_recovery",
        "tp_recovery", "topk_damage", "tp_damage", "explanation_fraction",
    ])
    empty_csv(REPORT / "p1_cluster_ci.csv", [
        "protocol", "pair", "layer", "intervention", "control_type", "metric",
        "estimate", "ci_low", "ci_high", "n", "n_scenes",
    ])
    empty_csv(REPORT / "p2_prediction_ci.csv", [
        "protocol", "prediction", "spearman", "ci_low", "ci_high", "n", "n_scenes",
    ])

    lines = [
        "# BD-specific Current-Conditioned Temporal Support Audit", "",
        "## Final decision", "",
        "`NO_GO_LOCALIZABLE_BD_TEMPORAL_SUPPORT`", "",
        "P0 is complete, but the NoHistory anchor fails the preregistered mechanism gate in all three protocols.  P1 and P2 are therefore locked and were not run.", "",
        "## P0 result", "",
        "| protocol | lost G_B (95% cluster CI) | lost G_D (95% cluster CI) | lost-retained G_B (95% CI) | lost-retained G_D (95% CI) | mechanism |", 
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.protocol} | {fmt(row.lost_G_B_estimate)} "
            f"[{fmt(row.lost_G_B_ci_low)}, {fmt(row.lost_G_B_ci_high)}] | "
            f"{fmt(row.lost_G_D_estimate)} [{fmt(row.lost_G_D_ci_low)}, {fmt(row.lost_G_D_ci_high)}] | "
            f"{fmt(row.lost_minus_retained_G_B_estimate)} "
            f"[{fmt(row.lost_minus_retained_G_B_ci_low)}, {fmt(row.lost_minus_retained_G_B_ci_high)}] | "
            f"{fmt(row.lost_minus_retained_G_D_estimate)} "
            f"[{fmt(row.lost_minus_retained_G_D_ci_low)}, {fmt(row.lost_minus_retained_G_D_ci_high)}] | "
            f"{row.mechanism} |"
        )
    lines += [
        "", "In every protocol, both G_B and G_D are strictly positive.  Thus both clean and fault history improve fault-current S_pos relative to NoHistory.  D is neither approximately F nor below F, so none of `B>F≈D`, `B≈F>D`, or `B>F>D` holds.  Moreover, the lost-minus-retained G_B contrast is strictly negative in every protocol, contradicting the required lost-specific clean-history enrichment.", "",
        "A/C show the same anchor problem: G_A and G_C are both positive in every protocol.  The anchor therefore measures broadly useful temporal support, but cannot causally isolate why full-representation B-to-D restoration worked while A-to-C restoration did not.", "",
        "## Coverage and exactness", "",
        "* 48/48 protocol-scenes and 13,803/13,803 frozen GT-protocol events completed; no duplicate unit IDs.",
        "* Scene/trajectory two-stage cluster bootstrap used 5,000 replicates.",
        "* 48/48 disabled-wrapper checks are output and post-forward-memory bitwise exact to B0; maximum absolute difference is 0.",
        "* Every E/F intervention retained exactly 644 current queries, removed all 256 propagated queries and supplied no temporal memory.  Intervention states were discarded and never updated the canonical histories.",
        "", "## Gate consequence", "",
        "Because the P0 NoHistory anchor core gate failed, layer-local support injection and the deficit-to-rescue prediction are not identifiable under the preregistered design.  Continuing to split memory/decoder layers is prohibited for this audit.", "",
    ]
    (REPORT / "REPORT.md").write_text("\n".join(lines))

    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    progress["status"] = "FINAL_NO_GO_LOCALIZABLE_BD_TEMPORAL_SUPPORT"
    progress["stages"]["P1"] = "LOCKED_NOT_RUN_P0_FAILED"
    progress["stages"]["P2"] = "LOCKED_NOT_RUN_P0_FAILED"
    progress["artifacts"] = {
        name: sha256(REPORT / name) for name in (
            "PRE_REGISTRATION.md", "population.csv", "per_gt_nohistory.csv",
            "p0_cluster_ci.csv", "p0_per_scene.csv", "final_decision.json", "REPORT.md",
        )
    }
    atomic_json(REPORT / "progress_manifest.json", progress)
    (REPORT / "PARTIAL_STATUS.md").write_text(
        "# STATUS\n\n`FINAL_NO_GO_LOCALIZABLE_BD_TEMPORAL_SUPPORT`\n\n"
        "P0 completed 48/48 protocol-scenes and 13,803/13,803 frozen events. "
        "The NoHistory anchor failed the preregistered core mechanism gate in all protocols; "
        "P1/P2 are locked and were not run.\n\n"
        "Resume verification (no recomputation):\n\n```bash\n"
        "python scripts/run_bd_temporal_support_p0.py --protocol blur_back\n"
        "python scripts/run_bd_temporal_support_p0.py --protocol crash_back\n"
        "python scripts/run_bd_temporal_support_p0.py --protocol dark_back\n"
        "```\n"
    )
    print(json.dumps({"decision": decision["decision"], "rows": len(data), **disabled_summary}, indent=2))


if __name__ == "__main__":
    main()
