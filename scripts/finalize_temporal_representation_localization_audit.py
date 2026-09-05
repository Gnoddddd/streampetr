#!/usr/bin/env python3
"""Write the terminal localization decision and consolidated audit report."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/full_nuscenes/temporal_representation_localization_audit"
ROUTING = ROOT / "reports/full_nuscenes/ctep_gradient_routing_audit"
PROTOCOLS = ("blur_back", "crash_back", "dark_back")
PAIRS = ("AC", "BD")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: object, path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    p0 = json.loads((REPORT / "p0_decision.json").read_text())
    p1 = json.loads((REPORT / "p1_decision.json").read_text())
    if p0.get("verdict") != "P0_GO_P1_REQUIRED":
        raise RuntimeError(f"unexpected P0 terminal state: {p0.get('verdict')}")
    if p1.get("verdict") != "NO_GO_DIRECT_TEMPORAL_REPRESENTATION_PRESERVATION":
        raise RuntimeError(f"unexpected P1 terminal state: {p1.get('verdict')}")
    if p1.get("p1_passing_taps"):
        raise RuntimeError("P2 cannot be marked locked when a P1 tap passes")

    routing_decision_path = ROUTING / "routing_decision.json"
    routing_decision = json.loads(routing_decision_path.read_text())
    p0_equivalence = pd.read_csv(REPORT / "p0_disabled_equivalence.csv")
    p1_equivalence = pd.read_csv(REPORT / "p1_disabled_equivalence.csv")
    disabled = {
        "current_passive_capture_output_memory_bitwise_exact": bool(
            p0_equivalence.passive_capture_output_bitwise_equal.astype(bool).all()
            and (p0_equivalence.passive_capture_output_max_abs_diff == 0).all()
            and p0_equivalence.passive_capture_memory_bitwise_equal.astype(bool).all()
            and (p0_equivalence.passive_capture_memory_max_abs_diff == 0).all()
        ),
        "current_empty_patch_downstream_output_bitwise_exact": bool(
            p1_equivalence.no_patch_output_bitwise_equal.astype(bool).all()
            and (p1_equivalence.no_patch_output_max_abs_diff == 0).all()
        ),
        "source_B0_output_loss_gradient_exact": bool(
            routing_decision.get("disabled_output_loss_gradient_exact_B0")
        ),
        "source_routing_decision": str(routing_decision_path.relative_to(ROOT)),
        "source_routing_decision_sha256": sha256(routing_decision_path),
    }
    disabled["all_disabled_exact"] = all(
        value for key, value in disabled.items() if key.endswith("exact")
    )
    atomic_json(disabled, REPORT / "disabled_equivalence_summary.json")

    taps = pd.read_csv(REPORT / "tap_manifest.csv")
    p0_ci = pd.read_csv(REPORT / "p0_cluster_ci.csv")
    p1_ci = pd.read_csv(REPORT / "p1_cluster_ci.csv")
    p1_summary = pd.read_csv(REPORT / "p1_summary.csv")
    p0_coverage = pd.read_csv(REPORT / "p0_coverage.csv")
    p1_coverage = pd.read_csv(REPORT / "p1_coverage.csv")

    decision_rows = []
    for tap in taps.tap_id:
        decision_rows.append({
            "tap_id": tap,
            "readout_proximity_rank": int(
                taps.loc[taps.tap_id == tap, "readout_proximity_rank"].iloc[0]
            ),
            "p0_lost_specific_drift_pass": tap in p0["p0_passing_taps"],
            "p1_causal_rescue_pass": tap in p1["p1_passing_taps"],
            "p2_gradient_compatibility": "NOT_RUN_P1_GATE_FAILED",
            "joint_pass": False,
            "selected": False,
        })
    pd.DataFrame(decision_rows).to_csv(REPORT / "tap_decision_summary.csv", index=False)
    # Requested gradient artifact with an explicit empty schema: no tap was
    # eligible for P2, so fabricating gradient observations is prohibited.
    pd.DataFrame(columns=[
        "unit_id", "protocol", "scene_token", "gt_token", "tap_id", "pair",
        "gradient_level", "gradient_cosine", "gradient_conflict",
    ]).to_csv(REPORT / "per_gt_gradient.csv", index=False)
    p2_status = {
        "status": "LOCKED_NOT_RUN",
        "reason": "No tap passed every preregistered P1 causal-rescue gate.",
        "eligible_taps": [],
        "gradient_rows": 0,
    }
    atomic_json(p2_status, REPORT / "p2_status.json")

    final = {
        "verdict": "NO_GO_DIRECT_TEMPORAL_REPRESENTATION_PRESERVATION",
        "selected_tap": None,
        "complete_P0_48_protocol_scenes": bool(p0_coverage.complete.all()),
        "complete_P1_48_protocol_scenes": bool(p1_coverage.complete.all()),
        "frozen_population_gt_protocol_events": 13803,
        "p0_passing_taps": p0["p0_passing_taps"],
        "p1_passing_taps": p1["p1_passing_taps"],
        "P2": "LOCKED_NOT_RUN",
        "disabled_output_loss_gradient_exact_B0": disabled["all_disabled_exact"],
        "training": "NOT_RUN_PROHIBITED_IN_THIS_AUDIT",
        "next_route": (
            "Do not use direct temporal representation preservation at these taps; "
            "a different temporal objective family is required."
        ),
    }
    atomic_json(final, REPORT / "final_decision.json")

    p0_lines = []
    for protocol in PROTOCOLS:
        for tap in taps.tap_id:
            for pair in PAIRS:
                row = p0_ci[
                    (p0_ci.protocol == protocol) & (p0_ci.tap_id == tap)
                    & (p0_ci.pair == pair) & (p0_ci.population == "lost_minus_retained")
                    & (p0_ci.metric == "cosine_distance")
                ].iloc[0]
                p0_lines.append(
                    f"| {protocol} | {tap} | {pair} | {row.estimate:.5f} "
                    f"[{row.ci_low:.5f}, {row.ci_high:.5f}] |"
                )

    p1_lines = []
    mechanism_lines = []
    for protocol in PROTOCOLS:
        for tap in taps.tap_id:
            for pair in PAIRS:
                target = p1_ci[
                    (p1_ci.protocol == protocol) & (p1_ci.tap_id == tap)
                    & (p1_ci.pair == pair) & (p1_ci.comparison == "lost_target")
                    & (p1_ci.metric == "delta_s_pos")
                ].iloc[0]
                nongt = p1_ci[
                    (p1_ci.protocol == protocol) & (p1_ci.tap_id == tap)
                    & (p1_ci.pair == pair)
                    & (p1_ci.comparison == "lost_target_minus_lost_non_gt")
                    & (p1_ci.metric == "delta_s_pos")
                ].iloc[0]
                retained = p1_ci[
                    (p1_ci.protocol == protocol) & (p1_ci.tap_id == tap)
                    & (p1_ci.pair == pair)
                    & (p1_ci.comparison == "lost_target_minus_retained_target")
                    & (p1_ci.metric == "delta_s_pos")
                ].iloc[0]
                p1_lines.append(
                    f"| {protocol} | {tap} | {pair} | {target.estimate:.5f} "
                    f"[{target.ci_low:.5f}, {target.ci_high:.5f}] | "
                    f"{nongt.estimate:.5f} [{nongt.ci_low:.5f}, {nongt.ci_high:.5f}] | "
                    f"{retained.estimate:.5f} [{retained.ci_low:.5f}, {retained.ci_high:.5f}] |"
                )
                group = p1_summary[
                    (p1_summary.protocol == protocol) & (p1_summary.tap_id == tap)
                    & (p1_summary.pair == pair) & (p1_summary.group == "lost_target")
                ].iloc[0]
                mechanism_lines.append(
                    f"| {protocol} | {tap} | {pair} | {group.topk_recovery_rate:.4f} | "
                    f"{group.tp_recovery_rate:.4f} | {group.topk_damage_rate:.4f} | "
                    f"{group.tp_damage_rate:.4f} |"
                )

    report = [
        "# Temporal Target Representation Localization Audit", "", "## Decision", "",
        "**`NO_GO_DIRECT_TEMPORAL_REPRESENTATION_PRESERVATION`**", "",
        "All three preregistered taps have stable lost-specific representation drift, but none "
        "has causal target restoration that is positive and stronger than both controls across "
        "Blur/Crash/Dark and AC/BD. P2 is therefore locked by design; no directional surrogate, "
        "gradient audit, formal representation loss or training was run.", "",
        "## Coverage and invariants", "",
        "- Frozen population: 13,803 GT-protocol events; no resampling.",
        "- P0: 48/48 protocol-scenes, 82,818 per-GT tap/pair rows.",
        f"- P1 target patches: {int(p1_coverage.observed_gt_target_rows.sum()):,}; "
        f"same-count non-GT controls: {int(p1_coverage.observed_non_gt_rows.sum()):,}.",
        f"- Disabled output/loss/gradient exact B0: `{disabled['all_disabled_exact']}`.",
        "- Empty-patch replay is bitwise exact for every tap/scene.", "",
        "## P0 lost-specific cosine drift", "",
        "| protocol | tap | pair | lost-retained cosine distance [95% CI] |",
        "|---|---|---|---:|", *p0_lines, "",
        "## P1 causal target restoration", "",
        "| protocol | tap | pair | lost target delta S_pos [95% CI] | target-nonGT [95% CI] | target-retained [95% CI] |",
        "|---|---|---|---:|---:|---:|", *p1_lines, "",
        "The consistent failure is pair-specific: BD restoration is often positive in Crash/Dark, "
        "while AC restoration is non-positive or statistically unstable at every tap and protocol. "
        "Blur also fails the target-versus-non-GT control gate for downstream taps. Thus drift is "
        "not sufficient evidence that direct restoration is a valid temporal target.", "",
        "## Top-K and TP outcomes", "",
        "| protocol | tap | pair | Top-K recovery | TP recovery | Top-K damage | TP damage |",
        "|---|---|---|---:|---:|---:|---:|", *mechanism_lines, "",
        "## Stage lock", "",
        "P2 status is `LOCKED_NOT_RUN`; `per_gt_gradient.csv` contains only its declared empty "
        "schema. No tap can be selected, so the closest-to-readout tie-break is not invoked. "
        "Direct temporal representation preservation at these taps is closed without metric, "
        "threshold, lambda or layer rescue.", "",
        "## Artifacts", "",
        "- `PRE_REGISTRATION.md`, `tap_manifest.csv`, `population.csv`",
        "- `per_gt_drift.csv`, `p0_summary.csv`, `p0_cluster_ci.csv`, `p0_decision.json`",
        "- `per_gt_causal_patch.csv`, `p1_summary.csv`, `p1_per_scene.csv`, `p1_cluster_ci.csv`, `p1_decision.json`",
        "- `tap_decision_summary.csv`, `per_gt_gradient.csv`, `p2_status.json`",
        "- `disabled_equivalence_summary.json`, `final_decision.json`, `progress_manifest.json`", "",
    ]
    (REPORT / "REPORT.md").write_text("\n".join(report))

    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    progress["status"] = "FINAL_DIRECT_REPRESENTATION_NO_GO"
    progress["final_verdict"] = final["verdict"]
    progress["stages"]["P2"] = p2_status
    atomic_json(progress, REPORT / "progress_manifest.json")
    (REPORT / "PARTIAL_STATUS.md").write_text(
        "# STATUS\n\n`FINAL_DIRECT_REPRESENTATION_NO_GO`\n\n"
        "Decision: `NO_GO_DIRECT_TEMPORAL_REPRESENTATION_PRESERVATION`. "
        "See `REPORT.md` and `final_decision.json`.\n"
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()

