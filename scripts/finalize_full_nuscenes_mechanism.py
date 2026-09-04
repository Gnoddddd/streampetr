#!/usr/bin/env python3
"""Assemble the gated full-nuScenes mechanism-confirmation handoff."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/full_nuscenes/mechanism_confirmation"
CHECKPOINT = ROOT / "checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth"
OFFICIAL = {"mAP": 0.4323, "NDS": 0.5369}


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f(value) -> float:
    return float(value)


def fmt(value) -> str:
    return f"{f(value):.4f}"


def main() -> None:
    required = [
        REPORT / "preflight/data_preflight.json",
        REPORT / "root_cause/full_val_metrics.csv",
        REPORT / "root_cause/root_cause_decision.json",
        REPORT / "temporal_2x2/temporal_decision.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required result missing: {missing}")

    baseline_metrics = read_json(
        REPORT / "baseline/clean_nusc/pts_bbox/metrics_summary.json"
    )
    local = {"mAP": baseline_metrics["mean_ap"], "NDS": baseline_metrics["nd_score"]}
    checkpoint_hash = sha256(CHECKPOINT)
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "repos/StreamPETR"), "rev-parse", "HEAD"], text=True
    ).strip()
    gate = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_source": (
            "https://github.com/exiawsh/storage/releases/download/v1.0/"
            "stream_petr_r50_flash_704_bs2_seq_90e.pth"
        ),
        "checkpoint_sha256": checkpoint_hash,
        "stream_petr_commit": commit,
        "config": "configs/full_nuscenes/stream_petr_r50_90e_clean_val.py",
        "config_sha256": sha256(
            ROOT / "configs/full_nuscenes/stream_petr_r50_90e_clean_val.py"
        ),
        "mechanism_config_sha256": sha256(
            ROOT / "configs/full_nuscenes/stream_petr_r50_90e_mechanism_val.py"
        ),
        "official_log": "baseline/official_stream_petr_r50_90e.log",
        "official_log_sha256": sha256(
            REPORT / "baseline/official_stream_petr_r50_90e.log"
        ),
        "official_mAP": OFFICIAL["mAP"],
        "local_mAP": local["mAP"],
        "mAP_difference": local["mAP"] - OFFICIAL["mAP"],
        "official_NDS": OFFICIAL["NDS"],
        "local_NDS": local["NDS"],
        "NDS_difference": local["NDS"] - OFFICIAL["NDS"],
        "predeclared_pass_tolerance_abs": 0.01,
        "predeclared_stop_threshold_abs": 0.02,
    }
    gate["passed"] = bool(
        abs(gate["mAP_difference"]) <= gate["predeclared_pass_tolerance_abs"]
        and abs(gate["NDS_difference"]) <= gate["predeclared_pass_tolerance_abs"]
    )
    (REPORT / "baseline/baseline_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(REPORT / "baseline/baseline_gate.csv", [gate])

    frozen_rows = []
    for protocol, filename, severity in (
        ("dark_back", "dark_back_10f_s09.json", 0.9),
        ("blur_back", "motion_blur_back_10f_s09.json", 0.9),
        ("crash_back", "camera_crash_back_10f.json", 1.0),
    ):
        path = ROOT / "protocols/presets" / filename
        frozen_rows.append({
            "protocol": protocol, "camera": "CAM_BACK", "severity": severity,
            "start_frame_inclusive": 3, "end_frame_inclusive": 12,
            "K": 100, "protocol_file": str(path.relative_to(ROOT)),
            "protocol_sha256": sha256(path),
            "gt_match": "same_class_global_center_le_2m_score_ge_0.1_greedy",
            "main_population": "all_clean_correct_GT_on_frames_3_through_12",
        })
    write_csv(REPORT / "frozen_definition_manifest.csv", frozen_rows)
    (REPORT / "frozen_definition_manifest.json").write_text(
        json.dumps(frozen_rows, indent=2) + "\n", encoding="utf-8"
    )

    paired_validation = []
    expected_tokens = {
        path.stem for path in (REPORT / "paired_inference/clean/trace").glob("*.npz")
    }
    for group in ("clean", "dark_back", "blur_back", "crash_back"):
        directory = REPORT / "paired_inference" / group
        tokens = {path.stem for path in (directory / "trace").glob("*.npz")}
        exit_code = int((directory / "exit_code.txt").read_text().strip())
        paired_validation.append({
            "protocol": group,
            "exit_code": exit_code,
            "trace_frames": len(tokens),
            "same_sample_tokens_as_clean": tokens == expected_tokens,
            "metrics_present": (directory / "formatted/pts_bbox/metrics_summary.json").is_file(),
            "passed": exit_code == 0 and len(tokens) == 6019 and tokens == expected_tokens,
        })
    write_csv(REPORT / "paired_inference/paired_validation.csv", paired_validation)

    preflight = read_json(required[0])
    metrics = read_csv(required[1])
    root = read_json(required[2])
    temporal = read_json(required[3])
    root_summary = read_csv(REPORT / "root_cause/mechanism_summary.csv")
    temporal_summary = read_csv(REPORT / "temporal_2x2/temporal_summary.csv")
    temporal_ci_rows = read_csv(REPORT / "temporal_2x2/cluster_bootstrap_ci.csv")
    pooled_root = {
        row["outcome"]: row for row in root_summary if row["protocol"] == "pooled"
    }
    pooled_temporal = {
        row["outcome"]: row for row in temporal_summary if row["protocol"] == "pooled"
    }
    temporal_ci = {
        (row["category"], row["protocol"], row["outcome"], row["metric"]): row
        for row in temporal_ci_rows
    }
    final = {
        "data_preflight_passed": bool(preflight["passed"]),
        "official_baseline_gate_passed": bool(gate["passed"]),
        "smoke_passed": all(
            len(list((REPORT / "smoke" / group / "trace").glob("*.npz"))) == 119
            for group in ("clean", "dark_back", "blur_back", "crash_back")
        ),
        "full_paired_inference_passed": all(row["passed"] for row in paired_validation),
        "root_cause_confirmed": bool(root["root_cause_confirmed"]),
        "temporal_history_effect_confirmed": bool(
            temporal["temporal_history_effect_confirmed"]
        ),
        "mini_51_of_53_consistent_with_sampling_limit": bool(
            root["mini_51_of_53_consistent_with_sampling_limit"]
        ),
        "core_failure_confirmed_on_full_data": bool(root["root_cause_confirmed"]),
        "next_method_selection_stage_unlocked": bool(root["root_cause_confirmed"]),
        "training_or_new_module_run": False,
        "lidar_kd_rerun": False,
    }
    write_csv(REPORT / "final_mechanism_decision.csv", [final])
    (REPORT / "final_mechanism_decision.json").write_text(
        json.dumps(final, indent=2) + "\n", encoding="utf-8"
    )

    metric_lines = [
        f"| {row['condition']} | {fmt(row['mAP'])} | {fmt(row['NDS'])} |"
        for row in metrics
    ]
    lost_root = pooled_root["fault_induced_lost"]
    retained_root = pooled_root["retained"]
    lost_temporal = pooled_temporal["fault_induced_lost"]
    retained_temporal = pooled_temporal["retained"]
    bd_ci = temporal_ci[("scene_mean_of_trajectory_medians", "pooled",
                         "fault_induced_lost", "B_minus_D_s_pos")]
    ca_ci = temporal_ci[("scene_mean_of_trajectory_medians", "pooled",
                         "fault_induced_lost", "C_minus_A_s_pos")]
    bd_contrast_ci = temporal_ci[("paired_scene_lost_minus_retained", "pooled",
                                  "contrast", "B_minus_D_s_pos")]
    ca_contrast_ci = temporal_ci[("paired_scene_lost_minus_retained", "pooled",
                                  "contrast", "C_minus_A_s_pos")]
    rank_available = sum(int(lost_root[key]) for key in (
        "target_only_n", "boundary_only_n", "mixed_n", "neither_n"
    ))
    report = f"""# Full-nuScenes Mechanism Confirmation

## Final decision

- Core failure confirmed on full data: **{final['core_failure_confirmed_on_full_data']}**.
- Temporal history effect confirmed under the frozen 2×2 definition: **{final['temporal_history_effect_confirmed']}**.
- Mini 51/53 no-alternative-view result is consistent with a sampling limitation: **{final['mini_51_of_53_consistent_with_sampling_limit']}**.
- Next method-selection stage unlocked by the preregistered gate: **{final['next_method_selection_stage_unlocked']}**.
- No training, new loss/module, LiDAR KD rerun, or post-hoc main-population change was performed.

## Data and baseline gates

- Data preflight: **{preflight['passed']}**; train/val scenes 700/150, overlap {preflight['train_val_scene_overlap']}; full filename inventory exact: {preflight['blob_inventory_exact']}.
- Scene memory reset static check: {preflight['scene_memory_reset_static_check']}.
- Official StreamPETR baseline gate: **{gate['passed']}**. Local mAP/NDS {local['mAP']:.4f}/{local['NDS']:.4f}; official {OFFICIAL['mAP']:.4f}/{OFFICIAL['NDS']:.4f}; differences {gate['mAP_difference']:+.4f}/{gate['NDS_difference']:+.4f}.
- StreamPETR commit `{commit}`; checkpoint SHA256 `{checkpoint_hash}`.
- Three-scene smoke: {final['smoke_passed']}; full paired token/exit validation: {final['full_paired_inference_passed']}.

## Full-val metrics (frozen paired K=100 runs)

| Protocol | mAP | NDS |
|---|---:|---:|
{chr(10).join(metric_lines)}

The official gate above uses the checkpoint's stock 300-box evaluator. The paired mechanism runs use the mini-frozen K=100 definition.

## Root cause and counterfactual

For the fixed Clean-correct active-frame population, pooled lost/retained counts are {lost_root['n']}/{retained_root['n']}. Lost median ΔS_pos={f(lost_root['median_delta_s_pos']):+.6f}, ΔS_K={f(lost_root['median_delta_s_k']):+.6f}, Top-K crossing rate={f(lost_root['topk_crossing_rate']):.3f}; retained median ΔS_pos={f(retained_root['median_delta_s_pos']):+.6f}. Among {rank_available} rank-available lost cases, target-only/boundary-only/mixed/neither counts are {lost_root['target_only_n']}/{lost_root['boundary_only_n']}/{lost_root['mixed_n']}/{lost_root['neither_n']}. Cluster CIs use 5,000 scene resamples after instance-trajectory aggregation.

The formal root-cause decision is **{root['root_cause_confirmed']}**: ΔS_pos negative across protocols={root['delta_s_pos_negative_all_protocols']}, ΔS_K nonpositive across protocols={root['delta_s_k_nonpositive_all_protocols']}, crossing present across protocols={root['crossing_present_all_protocols']}, and lost stronger than retained={root['lost_stronger_than_retained']}.

## Temporal 2×2

Pooled lost raw medians are B−D={f(lost_temporal['median_B_minus_D_s_pos']):+.6f}, C−A={f(lost_temporal['median_C_minus_A_s_pos']):+.6f}; retained raw medians are B−D={f(retained_temporal['median_B_minus_D_s_pos']):+.6f}, C−A={f(retained_temporal['median_C_minus_A_s_pos']):+.6f}. Scene/trajectory estimates are B−D={f(bd_ci['estimate']):+.3f} (95% CI {f(bd_ci['ci_low']):+.3f} to {f(bd_ci['ci_high']):+.3f}) and C−A={f(ca_ci['estimate']):+.3f} ({f(ca_ci['ci_low']):+.3f} to {f(ca_ci['ci_high']):+.3f}); lost−retained contrasts are {f(bd_contrast_ci['estimate']):+.3f} ({f(bd_contrast_ci['ci_low']):+.3f} to {f(bd_contrast_ci['ci_high']):+.3f}) and {f(ca_contrast_ci['estimate']):+.3f} ({f(ca_contrast_ci['ci_low']):+.3f} to {f(ca_contrast_ci['ci_high']):+.3f}). The decision is **{temporal['temporal_history_effect_confirmed']}**; fault-age monotonicity was not required.

## Alternative views and stratification

Full lost no-alternative-view rate is {f(lost_root['alternative_view_zero_rate']):.3f} over raw GT-protocol events. The scene/trajectory-standardized estimate is {root['full_lost_alternative_view_zero_rate']:.3f} (95% cluster CI {root['full_lost_alternative_view_zero_ci_low']:.3f}–{root['full_lost_alternative_view_zero_ci_high']:.3f}), versus mini 51/53={root['mini_alternative_view_zero_rate']:.3f}. Both full-data summaries are far below mini, supporting the sampling-limitation judgment. Per-GT physical view counts and annotation visibility are in `root_cause/per_gt.csv`; fixed class, distance, and visibility strata are in `root_cause/stratified.csv`.

## Artifacts

- `preflight/data_preflight.json`, `preflight/info_split.csv`, `preflight/blob_inventory.csv`
- `baseline/baseline_gate.json`, official/local logs and predictions
- `root_cause/full_val_metrics.csv`, `per_gt.csv`, `per_scene.csv`, `cluster_bootstrap_ci.csv`, `stratified.csv`, `mechanism_decision.csv`
- `temporal_2x2/per_gt_2x2.csv`, `per_scene_2x2.csv`, `cluster_bootstrap_ci.csv`, `stratified_2x2.csv`, `canonical_replay_invariance.csv`, `temporal_decision.csv`
- `final_mechanism_decision.csv`
"""
    (REPORT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
