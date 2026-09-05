#!/usr/bin/env python3
"""Incremental, state-isolated frame-2 probes for stress-reserve P0."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STREAM = ROOT / "repos/StreamPETR"
sys.dont_write_bytecode = True
sys.path.insert(0, str(STREAM))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmcv.utils import import_modules_from_strings  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from nuscenes.nuscenes import NuScenes  # noqa: E402

from analysis.fault_boundary_root_cause import (  # noqa: E402
    candidate_pool_statistics, projected_box_visibility,
)
from scripts.audit_dark_target_recoverability import (  # noqa: E402
    features, local_gt, physical, run_head, snapshot, unpack,
)
from scripts.audit_temporal_state_counterfactual import global_deployed_matches  # noqa: E402
from scripts.run_bd_temporal_support_p0 import (  # noqa: E402
    CHECKPOINT, CONFIG, DATA, atomic_csv, atomic_json, compare_outputs,
    compare_states, frame_context, protocol_dataset,
)


REPORT = ROOT / "reports/full_nuscenes/fault_stress_reserve_audit"
SOURCE = ROOT / "reports/full_nuscenes/temporal_utility_audit"
SCENES = ROOT / "reports/full_nuscenes/ctep_method_activation/scene_list.csv"
PROBES = {
    "blur_back": REPORT / "probes/cam_back_blur_09_frame2.json",
    "crash_back": REPORT / "probes/cam_back_crash_frame2.json",
    "dark_back": REPORT / "probes/cam_back_dark_09_frame2.json",
}
DISABLED = REPORT / "probes/disabled_empty_frame2.json"
SCHEMA = 1
STOP_REQUESTED = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--scene-token")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def output_arrays(output, pc_range):
    return (
        output["all_cls_scores"][-1, 0].detach().float().cpu().numpy(),
        physical(output, pc_range)[-1, 0].detach().float().cpu().numpy(),
    )


def metrics(logits, boxes, target, matches):
    pool = candidate_pool_statistics(logits, boxes, target["center"], target["label"], 100, 2.0)
    raw = float(pool["s_pos"])
    effective = raw if pool["candidate_available"] else 0.0
    return {
        "candidate": bool(pool["candidate_available"]), "candidate_count": int(pool["count"]),
        "qplus": int(pool["best_query"]), "s_pos_raw": raw, "s_pos": effective,
        "s_k": float(pool["s_k"]), "rank": int(pool["rank"]),
        "margin": float(pool["margin"]), "topk": bool(0 < pool["rank"] <= 100),
        "tp": bool(target["token"] in matches),
    }


def physical_fields(target, annotation, matrices, image_hw):
    view = projected_box_visibility(target["corners"], matrices, image_hw)
    areas, visible = view["area_fraction"], view["visible"]
    total = float(areas.sum())
    size = np.asarray(annotation["size"], dtype=float)
    return {
        "distance_m": float(np.linalg.norm(target["center"][:2])),
        "visibility_token": str(annotation["visibility_token"]),
        "projected_gt_area_cam_back": float(areas[3]),
        "fault_camera_visible": bool(visible[3]),
        "fault_camera_visible_fraction": float(areas[3] / total) if total > 0 else 0.0,
        "physical_visible_camera_count": int(visible.sum()),
        "alternative_visible_camera_count": int(np.delete(visible, 3).sum()),
        "object_width_m": float(size[0]), "object_length_m": float(size[1]),
        "object_height_m": float(size[2]), "object_volume_m3": float(np.prod(size)),
    }


def update_status(validation):
    cohort = pd.read_csv(REPORT / "frozen_cohort.csv")
    expected = set(cohort.scene_token.astype(str))
    metas = []
    for path in (REPORT / "incremental/P0").glob("*.complete.json"):
        value = json.loads(path.read_text())
        if value.get("complete") and value.get("schema_version") == SCHEMA \
                and value.get("frozen_cohort_sha256") == validation["frozen_cohort_sha256"]:
            metas.append(value)
    completed = sorted({value["scene_token"] for value in metas})
    rows = sum(int(value["rows"]) for value in metas)
    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    if not str(progress.get("status", "")).startswith("FINAL_"):
        progress["stages"]["P0_forward"] = {
            "completed_scenes": completed, "expected_scenes": len(expected), "rows": rows,
        }
        progress["status"] = ("P0_FORWARD_COMPLETE_ANALYSIS_PENDING"
                              if set(completed) == expected else "P0_FORWARD_RUNNING")
        atomic_json(REPORT / "progress_manifest.json", progress)
    lines = ["# PARTIAL STATUS", "", f"`{progress['status']}`", "",
             "Partial P0 coverage cannot produce a Go/No-Go decision.", "",
             f"Completed scenes: {len(completed)}/{len(expected)}; per-GT protocol rows: {rows}.", "",
             "Resume:", "", "```bash", "python scripts/run_fault_stress_reserve_p0.py",
             "python scripts/analyze_fault_stress_reserve_p0.py", "```", ""]
    temporary = REPORT / "PARTIAL_STATUS.md.tmp"
    temporary.write_text("\n".join(lines))
    os.replace(temporary, REPORT / "PARTIAL_STATUS.md")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    validation = json.loads((REPORT / "source_validation.json").read_text())
    if validation.get("status") != "VALIDATED_BEFORE_FORWARD":
        raise RuntimeError("pre-forward validation is missing")
    cohort = pd.read_csv(REPORT / "frozen_cohort.csv")
    if len(cohort) != 1323:
        raise RuntimeError("frozen cohort row count changed")
    scene_source = pd.read_csv(SCENES)
    scene_tokens = {row.scene_token: json.loads(row.sample_tokens_0_12)
                    for row in scene_source.itertuples(index=False)}
    expected_scenes = cohort.scene_token.astype(str).unique().tolist()
    complete_dir = REPORT / "incremental/P0"
    pending = []
    for scene in expected_scenes:
        marker = complete_dir / f"{scene}.complete.json"
        if marker.exists():
            value = json.loads(marker.read_text())
            if value.get("complete") and value.get("schema_version") == SCHEMA \
                    and value.get("frozen_cohort_sha256") == validation["frozen_cohort_sha256"]:
                continue
        pending.append(scene)
    if args.scene_token:
        pending = [scene for scene in pending if scene == args.scene_token]
    if args.max_scenes is not None:
        pending = pending[:args.max_scenes]
    if not pending:
        print("no pending P0 scenes")
        update_status(validation)
        return

    def request_stop(_signum, _frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    cfg = Config.fromfile(str(CONFIG))
    import_modules_from_strings(**cfg.custom_imports)
    cfg.model.pretrained = None
    clean_dataset = protocol_dataset(cfg, None)
    disabled_dataset = protocol_dataset(cfg, DISABLED)
    probe_datasets = {key: protocol_dataset(cfg, path) for key, path in PROBES.items()}
    token_index = {str(info["token"]): index for index, info in enumerate(clean_dataset.data_infos)}
    for name, dataset in {"disabled": disabled_dataset, **probe_datasets}.items():
        if token_index != {str(info["token"]): index for index, info in enumerate(dataset.data_infos)}:
            raise RuntimeError(f"paired dataset mismatch: {name}")
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(CHECKPOINT), map_location="cpu")
    model = model.to(device).eval()
    head = model.pts_bbox_head
    head.reset_memory()
    initial = snapshot(head)
    pc_range = head.pc_range.detach()
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(DATA), verbose=False)
    source_anchor = pd.read_csv(SOURCE / "per_gt_frame_cohort.csv")
    source_anchor = source_anchor[source_anchor.frame_idx == 2].set_index("trajectory_id")

    for scene in pending:
        selected_all = cohort[cohort.scene_token.astype(str) == scene]
        protocol_instances = {
            protocol: set(group.instance_token.astype(str))
            for protocol, group in selected_all.groupby("protocol")
        }
        if len(protocol_instances) != 3 or len({frozenset(v) for v in protocol_instances.values()}) != 1:
            raise RuntimeError(f"protocol anchor identities differ: {scene}")
        # The three protocol cohorts are identical before fault exposure.  Replay
        # each physical GT once, then emit one row per fixed probe protocol.
        selected = selected_all[selected_all.protocol == "blur_back"].copy()
        canonical_state = initial
        clean_pack = None
        for frame_idx, token in enumerate(scene_tokens[scene][:3]):
            index = token_index[token]
            meta, image, data = unpack(clean_dataset[index], device)
            with torch.no_grad():
                _, _, feats = features(model, image)
                feats = feats.detach()
                pre_state = canonical_state
                clean_output, canonical_state, _ = run_head(
                    model, meta, data, feats, frame_idx > 0, pre_state)
            if frame_idx == 2:
                clean_pack = (token, index, meta, image, data, feats, pre_state, clean_output,
                              copy.deepcopy(canonical_state))
            else:
                del clean_output, feats, image, data
        if clean_pack is None:
            raise RuntimeError("frame-2 clean anchor missing")
        token, index, clean_meta, clean_image, clean_data, clean_feats, pre_state, clean_output, saved_post = clean_pack

        disabled_meta, disabled_image, disabled_data = unpack(disabled_dataset[index], device)
        with torch.no_grad():
            _, _, disabled_feats = features(model, disabled_image)
            disabled_output, disabled_post, _ = run_head(
                model, disabled_meta, disabled_data, disabled_feats.detach(), True, pre_state)
        output_exact, output_diff = compare_outputs(clean_output, disabled_output)
        state_exact, state_diff = compare_states(canonical_state, disabled_post)
        if not output_exact or not state_exact:
            raise RuntimeError(f"disabled exactness failed for {scene}")
        equivalence = [{
            "scene_token": scene, "sample_token": token, "frame_idx": 2,
            "check": "disabled_empty_probe_B0", "output_bitwise_equal": output_exact,
            "output_max_abs_diff": output_diff, "memory_bitwise_equal": state_exact,
            "memory_max_abs_diff": state_diff,
        }]
        clean_logits, clean_boxes = output_arrays(clean_output, pc_range)
        targets = local_gt(nusc, token)
        annotations = {target["token"]: nusc.get("sample_annotation", target["token"])
                       for target in targets}
        for target in targets:
            target["instance_token"] = str(annotations[target["token"]]["instance_token"])
            target["global_center"] = np.asarray(
                annotations[target["token"]]["translation"], dtype=float)
        target_map = {target["token"]: target for target in targets}
        context = frame_context(clean_dataset.data_infos[index], clean_dataset)
        clean_matches = global_deployed_matches(clean_logits, clean_boxes, targets, context)
        probe_arrays = {}
        for protocol, dataset in probe_datasets.items():
            probe_meta, probe_image, probe_data = unpack(dataset[index], device)
            with torch.no_grad():
                _, _, probe_feats = features(model, probe_image)
                probe_output, _, _ = run_head(
                    model, probe_meta, probe_data, probe_feats.detach(), True, pre_state)
            probe_arrays[protocol] = output_arrays(probe_output, pc_range)
            state_unchanged, state_change = compare_states(saved_post, canonical_state)
            if not state_unchanged:
                raise RuntimeError(f"probe mutated stored canonical state: {protocol}/{scene}")
            equivalence.append({
                "scene_token": scene, "sample_token": token, "frame_idx": 2,
                "check": f"{protocol}_canonical_state_object_isolated",
                "output_bitwise_equal": True, "output_max_abs_diff": 0.0,
                "memory_bitwise_equal": state_unchanged, "memory_max_abs_diff": state_change,
            })
            del probe_output, probe_feats, probe_image, probe_data
        matrices = clean_data["lidar2img"][0].detach().float().cpu().numpy()
        image_hw = tuple(int(value) for value in clean_image.shape[-2:])
        matches = {key: global_deployed_matches(logits, boxes, targets, context)
                   for key, (logits, boxes) in probe_arrays.items()}
        rows = []
        for frozen in selected.to_dict("records"):
            gt_token = str(frozen["gt_token"])
            if gt_token not in target_map:
                raise RuntimeError(f"frozen GT absent at anchor: {frozen['trajectory_id']}")
            target = target_map[gt_token]
            annotation = annotations[gt_token]
            if target["instance_token"] != str(frozen["instance_token"]):
                raise RuntimeError(f"instance identity mismatch: {frozen['trajectory_id']}")
            clean = metrics(clean_logits, clean_boxes, target, clean_matches)
            if not clean["candidate"] or not clean["tp"]:
                raise RuntimeError(f"prospective Clean-TP anchor did not replay: {frozen['trajectory_id']}")
            cached = source_anchor.loc[frozen["trajectory_id"]]
            if abs(clean["s_pos"] - float(cached.A_s_pos)) > 2e-6:
                raise RuntimeError(f"anchor S_pos replay mismatch: {frozen['trajectory_id']}")
            values = {key: metrics(logits, boxes, target, matches[key])
                      for key, (logits, boxes) in probe_arrays.items()}
            r_k = clean["s_pos"] - clean["s_k"]
            base = {**frozen, "sample_token": token, "frame_idx": 2,
                    **physical_fields(target, annotation, matrices, image_hw)}
            for key, value in clean.items():
                base[f"clean_{key}"] = value
            base["R_K"] = r_k
            crash_s = values["crash_back"]["s_pos"]
            base["D_cam"] = clean["s_pos"] - crash_s
            for protocol in PROBES:
                value = values[protocol]
                for key, item in value.items():
                    base[f"probe_{key}"] = item
                base["J_p"] = clean["s_pos"] - value["s_pos"]
                base["M_p"] = base["J_p"] - r_k
                row = dict(base)
                row["protocol"] = protocol
                row["trajectory_id"] = (f"{protocol}:{scene}:{frozen['instance_token']}")
                rows.append(row)
        if len(rows) != len(selected_all):
            raise RuntimeError(f"scene row mismatch: {len(rows)} != {len(selected_all)}")
        atomic_csv(complete_dir / f"{scene}.csv", rows)
        atomic_csv(complete_dir / f"{scene}.equivalence.csv", equivalence)
        atomic_json(complete_dir / f"{scene}.complete.json", {
            "schema_version": SCHEMA, "frozen_cohort_sha256": validation["frozen_cohort_sha256"],
            "scene_token": scene, "sample_token": token, "rows": len(rows),
            "equivalence_rows": len(equivalence), "complete": True,
        })
        update_status(validation)
        print(f"completed P0 {scene}: {len(rows)} protocol-GT rows", flush=True)
        del clean_output, disabled_output, clean_feats, disabled_feats, clean_image, disabled_image
        torch.cuda.empty_cache()
        if STOP_REQUESTED:
            print("stop requested; current scene checkpoint saved", flush=True)
            break


if __name__ == "__main__":
    main()
