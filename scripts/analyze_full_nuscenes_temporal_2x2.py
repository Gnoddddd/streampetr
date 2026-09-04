#!/usr/bin/env python3
"""Full-nuScenes Clean/Fault history x current-observation 2x2 replay."""

from __future__ import annotations

import copy
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STREAM = ROOT / "repos/StreamPETR"
sys.path.insert(0, str(STREAM))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmcv.utils import import_modules_from_strings  # noqa: E402
from mmdet3d.datasets import build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from nuscenes.nuscenes import NuScenes  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402

from scripts.audit_cross_view_target_evidence import protocol_dataset  # noqa: E402
from scripts.audit_dark_target_recoverability import (  # noqa: E402
    features,
    local_gt,
    physical,
    run_head,
    snapshot,
    unpack,
)
from scripts.audit_temporal_state_counterfactual import (  # noqa: E402
    delta,
    output_metrics,
    state_max_difference,
)


CONFIG = ROOT / "configs/full_nuscenes/stream_petr_r50_90e_mechanism_val.py"
CHECKPOINT = ROOT / "checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth"
RUN = ROOT / "reports/full_nuscenes/mechanism_confirmation/paired_inference"
ROOT_ROWS = ROOT / "reports/full_nuscenes/mechanism_confirmation/root_cause/per_gt.csv"
REPORT = ROOT / "reports/full_nuscenes/mechanism_confirmation/temporal_2x2"
DATA = ROOT / "data/nuscenes"
PROTOCOLS = {
    "dark_back": ("CAM_BACK Dark", ROOT / "protocols/presets/dark_back_10f_s09.json"),
    "blur_back": ("CAM_BACK Motion Blur", ROOT / "protocols/presets/motion_blur_back_10f_s09.json"),
    "crash_back": ("CAM_BACK Crash", ROOT / "protocols/presets/camera_crash_back_10f.json"),
}
BOOTSTRAPS = 5000
SEED = 271828


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_rows() -> list[dict]:
    with ROOT_ROWS.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_median(rows: list[dict], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def rate(rows: list[dict], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows])) if rows else float("nan")


def scene_trajectory_estimates(rows: list[dict], key: str) -> np.ndarray:
    by_scene = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = float(row[key])
        if math.isfinite(value):
            by_scene[str(row["scene_token"])][str(row["instance_token"])].append(value)
    estimates = []
    for trajectories in by_scene.values():
        trajectory_medians = [float(np.median(values)) for values in trajectories.values()]
        if trajectory_medians:
            estimates.append(float(np.mean(trajectory_medians)))
    return np.asarray(estimates, dtype=float)


def cluster_ci(rows: list[dict], key: str, seed: int) -> dict:
    scene_values = scene_trajectory_estimates(rows, key)
    estimate = float(np.mean(scene_values)) if scene_values.size else float("nan")
    if not scene_values.size or not math.isfinite(estimate):
        return {"estimate": estimate, "ci_low": float("nan"),
                "ci_high": float("nan"), "iterations": BOOTSTRAPS,
                "scene_clusters": int(scene_values.size)}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, scene_values.size, size=(BOOTSTRAPS, scene_values.size))
    values = np.mean(scene_values[indices], axis=1)
    low, high = np.percentile(values, [2.5, 97.5])
    return {"estimate": estimate, "ci_low": float(low), "ci_high": float(high),
            "iterations": BOOTSTRAPS, "scene_clusters": int(scene_values.size)}


def contrast_ci(rows: list[dict], key: str, seed: int) -> dict:
    differences = []
    for scene in sorted({str(row["scene_token"]) for row in rows}):
        scene_rows = [row for row in rows if str(row["scene_token"]) == scene]
        lost = scene_trajectory_estimates(
            [row for row in scene_rows if row["outcome"] == "fault_induced_lost"], key
        )
        retained = scene_trajectory_estimates(
            [row for row in scene_rows if row["outcome"] == "retained"], key
        )
        if lost.size and retained.size:
            differences.append(float(lost[0] - retained[0]))
    scene_values = np.asarray(differences, dtype=float)
    estimate = float(np.mean(scene_values)) if scene_values.size else float("nan")
    if not scene_values.size:
        return {"estimate": estimate, "ci_low": float("nan"), "ci_high": float("nan"),
                "iterations": BOOTSTRAPS, "scene_clusters": 0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, scene_values.size, size=(BOOTSTRAPS, scene_values.size))
    values = np.mean(scene_values[indices], axis=1)
    low, high = np.percentile(values, [2.5, 97.5])
    return {"estimate": estimate, "ci_low": float(low), "ci_high": float(high),
            "iterations": BOOTSTRAPS, "scene_clusters": int(scene_values.size)}


def trace_difference(protocol: str, token: str, output, pc_range) -> tuple:
    with np.load(RUN / protocol / "trace" / f"{token}.npz") as trace:
        logits = output["all_cls_scores"][:, 0].detach().cpu().numpy().astype(np.float16)
        logits_exact = bool(np.array_equal(logits, trace["layer_logits"]))
        boxes = physical(output, pc_range)[:, 0].detach().float().cpu().numpy()
        box_difference = float(np.max(np.abs(boxes - trace["layer_boxes"])))
    return logits_exact, box_difference


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    REPORT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")

    population = load_rows()
    units = defaultdict(list)
    for row in population:
        units[(row["protocol"], row["sample_token"])].append(row)

    cfg = Config.fromfile(str(CONFIG))
    import_modules_from_strings(**cfg.custom_imports)
    cfg.model.train_cfg = None
    clean_config = copy.deepcopy(cfg.data.test)
    clean_config.test_mode = True
    clean_dataset = build_dataset(clean_config)
    datasets = {key: protocol_dataset(cfg, value[1]) for key, value in PROTOCOLS.items()}
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(CHECKPOINT), map_location="cpu")
    model = model.to(device).eval()
    head = model.pts_bbox_head
    head.reset_memory()
    initial = snapshot(head)
    states = {name: initial for name in ("clean", *PROTOCOLS)}
    pc_range = head.pc_range.detach()
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(DATA), verbose=False)

    per_gt = []
    replay = defaultdict(lambda: {"frames": 0, "logits_exact": True, "box_max": 0.0})
    previous_scene = None
    processed = 0
    with torch.no_grad():
        for index, info in enumerate(clean_dataset.data_infos):
            frame_idx = int(info["frame_idx"])
            if frame_idx > 12:
                continue
            clean_meta, clean_image, clean_data = unpack(clean_dataset[index], device)
            token = str(clean_meta["sample_idx"])
            scene = str(clean_meta["scene_token"])
            prev_exists = 0 if scene != previous_scene else 1
            previous_scene = scene
            _, _, clean_feats = features(model, clean_image)
            clean_pre = states["clean"]
            output_a, states["clean"], _ = run_head(
                model, clean_meta, clean_data, clean_feats, prev_exists, clean_pre
            )
            exact, box_diff = trace_difference("clean", token, output_a, pc_range)
            replay["clean"]["frames"] += 1
            replay["clean"]["logits_exact"] &= exact
            replay["clean"]["box_max"] = max(replay["clean"]["box_max"], box_diff)
            if not exact or box_diff > 1e-5:
                raise RuntimeError(f"clean replay divergence {token}: {exact}/{box_diff}")

            targets = local_gt(nusc, token)
            for target in targets:
                target["global_center"] = np.asarray(
                    nusc.get("sample_annotation", target["token"])["translation"],
                    dtype=float,
                )
            by_token = {target["token"]: target for target in targets}
            match_context = {
                "lidar2ego_rotation": Quaternion(info["lidar2ego_rotation"]).rotation_matrix,
                "lidar2ego_translation": np.asarray(info["lidar2ego_translation"], float),
                "ego2global_rotation": Quaternion(info["ego2global_rotation"]).rotation_matrix,
                "ego2global_translation": np.asarray(info["ego2global_translation"], float),
                "class_range": clean_dataset.eval_detection_configs.class_range,
            }
            for protocol, dataset in datasets.items():
                fault_meta, fault_image, fault_data = unpack(dataset[index], device)
                if str(fault_meta["sample_idx"]) != token:
                    raise RuntimeError(f"paired token mismatch {protocol}/{index}")
                _, _, fault_feats = features(model, fault_image)
                fault_pre = states[protocol]
                state_difference = state_max_difference(clean_pre, fault_pre)
                output_d, states[protocol], _ = run_head(
                    model, fault_meta, fault_data, fault_feats, prev_exists, fault_pre
                )
                exact, box_diff = trace_difference(protocol, token, output_d, pc_range)
                replay[protocol]["frames"] += 1
                replay[protocol]["logits_exact"] &= exact
                replay[protocol]["box_max"] = max(replay[protocol]["box_max"], box_diff)
                if not exact or box_diff > 1e-5:
                    raise RuntimeError(
                        f"{protocol} replay divergence {token}: {exact}/{box_diff}"
                    )
                frame_units = units.get((protocol, token), [])
                if not frame_units:
                    continue
                output_b = run_head(
                    model, fault_meta, fault_data, fault_feats, prev_exists, clean_pre
                )[0]
                output_c = run_head(
                    model, clean_meta, clean_data, clean_feats, prev_exists, fault_pre
                )[0]
                for root_row in frame_units:
                    target = by_token[root_row["gt_token"]]
                    values = {
                        "A": output_metrics(output_a, target, targets, pc_range, match_context),
                        "B": output_metrics(output_b, target, targets, pc_range, match_context),
                        "C": output_metrics(output_c, target, targets, pc_range, match_context),
                        "D": output_metrics(output_d, target, targets, pc_range, match_context),
                    }
                    expected_d = root_row["outcome"] == "retained"
                    if not values["A"]["tp"] or values["D"]["tp"] != expected_d:
                        raise RuntimeError(
                            f"canonical outcome mismatch {protocol}/{token}/{target['token']}"
                        )
                    row = {
                        "protocol": protocol, "condition": PROTOCOLS[protocol][0],
                        "outcome": root_row["outcome"], "sample_token": token,
                        "scene_token": scene, "frame_idx": frame_idx,
                        "gt_token": target["token"],
                        "instance_token": root_row["instance_token"],
                        "gt_class": target["name"],
                        "distance_m": root_row["distance_m"],
                        "distance_bin": root_row["distance_bin"],
                        "visibility_token": root_row["visibility_token"],
                        "alternative_view_count": root_row["alternative_view_count"],
                        "state_max_abs_diff": state_difference,
                    }
                    for condition, value in values.items():
                        for key in ("candidate", "qplus", "s_pos", "rank", "margin", "topk", "tp"):
                            row[f"{condition}_{key}"] = value[key]
                    row.update({
                        "B_minus_D_s_pos": delta(values["B"]["s_pos"], values["D"]["s_pos"]),
                        "C_minus_A_s_pos": delta(values["C"]["s_pos"], values["A"]["s_pos"]),
                        "B_minus_D_margin": delta(values["B"]["margin"], values["D"]["margin"]),
                        "C_minus_A_margin": delta(values["C"]["margin"], values["A"]["margin"]),
                        "BD_topk_recovered": not values["D"]["topk"] and values["B"]["topk"],
                        "BD_tp_recovered": not values["D"]["tp"] and values["B"]["tp"],
                        "CA_topk_lost": values["A"]["topk"] and not values["C"]["topk"],
                        "CA_tp_lost": values["A"]["tp"] and not values["C"]["tp"],
                    })
                    per_gt.append(row)
            processed += 1
            if processed % 100 == 0:
                print(f"replay frames {processed}/1950 rows={len(per_gt)}", flush=True)

    write_csv(REPORT / "per_gt_2x2.csv", per_gt)
    replay_rows = [{"protocol": key, **value,
        "exact": bool(value["logits_exact"] and value["box_max"] <= 1e-5)}
        for key, value in replay.items()]
    write_csv(REPORT / "canonical_replay_invariance.csv", replay_rows)

    summary = []
    scene_rows = []
    for protocol in (*PROTOCOLS, "pooled"):
        protocol_rows = per_gt if protocol == "pooled" else [
            row for row in per_gt if row["protocol"] == protocol
        ]
        for outcome in ("fault_induced_lost", "retained"):
            selected = [row for row in protocol_rows if row["outcome"] == outcome]
            summary.append({"protocol": protocol,
                "condition": "Pooled" if protocol == "pooled" else PROTOCOLS[protocol][0],
                "outcome": outcome, "n": len(selected),
                "median_A_s_pos": finite_median(selected, "A_s_pos"),
                "median_B_s_pos": finite_median(selected, "B_s_pos"),
                "median_C_s_pos": finite_median(selected, "C_s_pos"),
                "median_D_s_pos": finite_median(selected, "D_s_pos"),
                "median_B_minus_D_s_pos": finite_median(selected, "B_minus_D_s_pos"),
                "median_C_minus_A_s_pos": finite_median(selected, "C_minus_A_s_pos"),
                "BD_topk_recovery_rate": rate(selected, "BD_topk_recovered"),
                "BD_tp_recovery_rate": rate(selected, "BD_tp_recovered"),
                "CA_topk_loss_rate": rate(selected, "CA_topk_lost"),
                "CA_tp_loss_rate": rate(selected, "CA_tp_lost")})
        for scene in sorted({row["scene_token"] for row in protocol_rows}):
            for outcome in ("fault_induced_lost", "retained"):
                selected = [row for row in protocol_rows
                            if row["scene_token"] == scene and row["outcome"] == outcome]
                if selected:
                    scene_rows.append({"protocol": protocol, "scene_token": scene,
                        "outcome": outcome, "n": len(selected),
                        "median_B_minus_D_s_pos": finite_median(selected, "B_minus_D_s_pos"),
                        "median_C_minus_A_s_pos": finite_median(selected, "C_minus_A_s_pos")})
    write_csv(REPORT / "temporal_summary.csv", summary)
    write_csv(REPORT / "per_scene_2x2.csv", scene_rows)

    ci_rows = []
    ci_number = 0
    for protocol in (*PROTOCOLS, "pooled"):
        protocol_rows = per_gt if protocol == "pooled" else [
            row for row in per_gt if row["protocol"] == protocol
        ]
        for outcome in ("fault_induced_lost", "retained"):
            selected = [row for row in protocol_rows if row["outcome"] == outcome]
            for metric in ("B_minus_D_s_pos", "C_minus_A_s_pos"):
                result = cluster_ci(selected, metric, SEED + ci_number)
                ci_number += 1
                ci_rows.append({"category": "scene_mean_of_trajectory_medians", "protocol": protocol,
                    "outcome": outcome, "metric": metric,
                    "cluster": "scene_bootstrap_on_trajectory_aggregates", **result})
        for metric in ("B_minus_D_s_pos", "C_minus_A_s_pos"):
            result = contrast_ci(protocol_rows, metric, SEED + ci_number)
            ci_number += 1
            ci_rows.append({"category": "paired_scene_lost_minus_retained",
                "protocol": protocol, "outcome": "contrast", "metric": metric,
                "cluster": "scene_bootstrap_on_trajectory_aggregates", **result})
    write_csv(REPORT / "cluster_bootstrap_ci.csv", ci_rows)

    strata_rows = []
    dimensions = {
        "class": "gt_class",
        "distance": "distance_bin",
        "visibility": "visibility_token",
    }
    for protocol in (*PROTOCOLS, "pooled"):
        protocol_rows = per_gt if protocol == "pooled" else [
            row for row in per_gt if row["protocol"] == protocol
        ]
        for dimension, field in dimensions.items():
            for value in sorted({str(row[field]) for row in protocol_rows}):
                subset = [row for row in protocol_rows if str(row[field]) == value]
                for outcome in ("fault_induced_lost", "retained"):
                    selected = [row for row in subset if row["outcome"] == outcome]
                    for metric in ("B_minus_D_s_pos", "C_minus_A_s_pos"):
                        result = cluster_ci(selected, metric, SEED + ci_number)
                        ci_number += 1
                        strata_rows.append({
                            "protocol": protocol, "dimension": dimension,
                            "stratum": value, "outcome": outcome, "n": len(selected),
                            "metric": metric,
                            "raw_median": finite_median(selected, metric),
                            "cluster": "scene_bootstrap_on_trajectory_aggregates",
                            **result,
                        })
    write_csv(REPORT / "stratified_2x2.csv", strata_rows)

    index = {(row["category"], row["protocol"], row["outcome"], row["metric"]): row
             for row in ci_rows}
    summaries = {(row["protocol"], row["outcome"]): row for row in summary}
    bd = index[("scene_mean_of_trajectory_medians", "pooled",
                "fault_induced_lost", "B_minus_D_s_pos")]
    ca = index[("scene_mean_of_trajectory_medians", "pooled",
                "fault_induced_lost", "C_minus_A_s_pos")]
    bd_contrast = index[("paired_scene_lost_minus_retained", "pooled", "contrast",
                         "B_minus_D_s_pos")]
    ca_contrast = index[("paired_scene_lost_minus_retained", "pooled", "contrast",
                         "C_minus_A_s_pos")]
    bd_cross_protocol = all(
        summaries[(protocol, "fault_induced_lost")]["median_B_minus_D_s_pos"] > 0
        for protocol in PROTOCOLS
    )
    ca_cross_protocol = all(
        summaries[(protocol, "fault_induced_lost")]["median_C_minus_A_s_pos"] < 0
        for protocol in PROTOCOLS
    )
    bd_ci_cross_protocol = all(
        index[("scene_mean_of_trajectory_medians", protocol,
               "fault_induced_lost", "B_minus_D_s_pos")]["ci_low"] > 0
        for protocol in PROTOCOLS
    )
    ca_ci_cross_protocol = all(
        index[("scene_mean_of_trajectory_medians", protocol,
               "fault_induced_lost", "C_minus_A_s_pos")]["ci_high"] < 0
        for protocol in PROTOCOLS
    )
    bd_contrast_cross_protocol = all(
        index[("paired_scene_lost_minus_retained", protocol, "contrast",
               "B_minus_D_s_pos")]["ci_low"] > 0
        for protocol in PROTOCOLS
    )
    ca_contrast_cross_protocol = all(
        index[("paired_scene_lost_minus_retained", protocol, "contrast",
               "C_minus_A_s_pos")]["ci_high"] < 0
        for protocol in PROTOCOLS
    )
    decision = {
        "temporal_history_effect_confirmed": bool(
            bd_cross_protocol and ca_cross_protocol
            and bd_ci_cross_protocol and ca_ci_cross_protocol
            and bd_contrast_cross_protocol and ca_contrast_cross_protocol
            and bd["ci_low"] > 0
            and ca["ci_high"] < 0 and bd_contrast["ci_low"] > 0
            and ca_contrast["ci_high"] < 0
        ),
        "B_minus_D_positive_all_protocols": bd_cross_protocol,
        "C_minus_A_negative_all_protocols": ca_cross_protocol,
        "B_minus_D_ci_above_zero_all_protocols": bd_ci_cross_protocol,
        "C_minus_A_ci_below_zero_all_protocols": ca_ci_cross_protocol,
        "pooled_B_minus_D_ci_above_zero": bd["ci_low"] > 0,
        "pooled_C_minus_A_ci_below_zero": ca["ci_high"] < 0,
        "B_minus_D_stronger_than_retained": bd_contrast["ci_low"] > 0,
        "C_minus_A_stronger_than_retained": ca_contrast["ci_high"] < 0,
        "B_minus_D_stronger_than_retained_all_protocols": bd_contrast_cross_protocol,
        "C_minus_A_stronger_than_retained_all_protocols": ca_contrast_cross_protocol,
        "fault_age_monotonicity_required": False,
    }
    write_csv(REPORT / "temporal_decision.csv", [decision])
    (REPORT / "temporal_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
