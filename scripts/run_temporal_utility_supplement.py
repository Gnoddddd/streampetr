#!/usr/bin/env python3
"""Supplement only pre-fault anchors and missing NoHistory F rows."""

from __future__ import annotations

import argparse
import csv
import json
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

from analysis.fault_boundary_root_cause import projected_box_visibility  # noqa: E402
from scripts.audit_dark_target_recoverability import (  # noqa: E402
    features, local_gt, physical, run_head, snapshot, unpack,
)
from scripts.audit_temporal_state_counterfactual import global_deployed_matches  # noqa: E402
from scripts.run_bd_temporal_support_p0 import (  # noqa: E402
    CHECKPOINT, CONFIG, DATA, PROTOCOLS, atomic_csv, atomic_json,
    compare_outputs, compare_states, frame_context, gt_metrics, nohistory_head,
    protocol_dataset,
)


SOURCE = ROOT / "reports/full_nuscenes/ctep_method_activation"
ABCD = SOURCE / "per_gt_p0.csv"
F_CACHE = ROOT / "reports/full_nuscenes/bd_temporal_support_audit/per_gt_nohistory.csv"
REPORT = ROOT / "reports/full_nuscenes/temporal_utility_audit"
SCHEMA = 1
STOP_REQUESTED = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=tuple(PROTOCOLS), required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--scene-token")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def physical_fields(target: dict, matrices, image_hw) -> dict:
    visibility = projected_box_visibility(target["corners"], matrices, image_hw)
    visible = visibility["visible"]
    return {
        "distance_m": float(np.linalg.norm(target["center"][:2])),
        "visibility_token": str(target["visibility_token"]),
        "cam_back_visible": bool(visible[3]),
        "physical_visible_view_count": int(np.count_nonzero(visible)),
        "alternative_view_count": int(np.count_nonzero(np.delete(visible, 3))),
        "max_projected_area_fraction": float(np.max(visibility["area_fraction"])),
    }


def output_arrays(output, pc_range):
    return (
        output["all_cls_scores"][-1, 0].detach().float().cpu().numpy(),
        physical(output, pc_range)[-1, 0].detach().float().cpu().numpy(),
    )


def status_update() -> None:
    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    if str(progress.get("status", "")).startswith("FINAL_"):
        return
    split = pd.read_csv(REPORT / "scene_split.csv")
    lines = [
        "# PARTIAL STATUS", "", "`SUPPLEMENT_RUNNING`", "",
        "Supplement coverage is incomplete; no mechanism decision is permitted.", "",
        "| protocol | scenes | rows |", "|---|---:|---:|",
    ]
    complete = True
    stage = {}
    for protocol in PROTOCOLS:
        directory = REPORT / "incremental/supplement" / protocol
        metas = [json.loads(path.read_text()) for path in directory.glob("*.complete.json")]
        tokens = sorted(item["scene_token"] for item in metas if item.get("complete"))
        rows = sum(int(item["rows"]) for item in metas if item.get("complete"))
        stage[protocol] = {"completed_scenes": tokens, "expected_scenes": 16, "rows": rows}
        lines.append(f"| {protocol} | {len(tokens)}/16 | {rows} |")
        complete &= set(tokens) == set(split.scene_token.astype(str))
    progress["stages"]["supplement"] = stage
    progress["status"] = "SUPPLEMENT_COMPLETE_ANALYSIS_PENDING" if complete else "SUPPLEMENT_RUNNING"
    atomic_json(REPORT / "progress_manifest.json", progress)
    lines += ["", "Resume:", "", "```bash"]
    lines += [f"python scripts/run_temporal_utility_supplement.py --protocol {p}" for p in PROTOCOLS]
    lines += ["python scripts/analyze_temporal_utility.py", "```", ""]
    temporary = REPORT / "PARTIAL_STATUS.md.tmp"
    temporary.write_text("\n".join(lines))
    os.replace(temporary, REPORT / "PARTIAL_STATUS.md")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    validation = json.loads((REPORT / "source_validation.json").read_text())
    if validation.get("status") != "VALIDATED_BEFORE_SUPPLEMENT_FORWARD":
        raise RuntimeError("pre-forward validation missing")
    split = pd.read_csv(REPORT / "scene_split.csv")
    scenes = split.scene_token.astype(str).tolist()
    output_dir = REPORT / "incremental/supplement" / args.protocol
    pending = []
    for scene in scenes:
        path = output_dir / f"{scene}.complete.json"
        if path.exists():
            value = json.loads(path.read_text())
            if value.get("complete") and value.get("schema_version") == SCHEMA \
                    and value.get("scene_split_sha256") == validation["scene_split_sha256"]:
                continue
        pending.append(scene)
    if args.scene_token:
        pending = [scene for scene in pending if scene == args.scene_token]
    if args.max_scenes is not None:
        pending = pending[:args.max_scenes]
    if not pending:
        print(f"no pending supplement scenes for {args.protocol}")
        status_update()
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
    fault_dataset = protocol_dataset(cfg, PROTOCOLS[args.protocol])
    token_index = {str(info["token"]): index for index, info in enumerate(clean_dataset.data_infos)}
    if token_index != {str(info["token"]): index for index, info in enumerate(fault_dataset.data_infos)}:
        raise RuntimeError("paired dataset mismatch")
    scene_source = pd.read_csv(SOURCE / "scene_list.csv")
    scene_tokens = {
        row.scene_token: json.loads(row.sample_tokens_0_12)
        for row in scene_source.itertuples(index=False)
    }
    abcd = pd.read_csv(ABCD)
    abcd = abcd[abcd.protocol == args.protocol]
    abcd_ids = set(abcd.unit_id)
    f_cache = pd.read_csv(F_CACHE, usecols=[
        "unit_id", "F_candidate", "F_qplus", "F_s_pos", "F_rank", "F_margin", "F_topk", "F_tp",
    ])
    cached_ids = set(f_cache.unit_id)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(CHECKPOINT), map_location="cpu")
    model = model.to(device).eval()
    head = model.pts_bbox_head
    head.reset_memory()
    initial = snapshot(head)
    pc_range = head.pc_range.detach()
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(DATA), verbose=False)

    for scene in pending:
        clean_state = initial
        rows, equivalence = [], []
        expected_active = abcd[abcd.scene_token == scene]
        expected_active_ids = set(expected_active.unit_id)
        missing_active_ids = expected_active_ids - cached_ids
        for frame_idx, token in enumerate(scene_tokens[scene]):
            index = token_index[token]
            clean_meta, clean_image, clean_data = unpack(clean_dataset[index], device)
            targets = local_gt(nusc, token)
            annotations = {}
            for target in targets:
                annotation = nusc.get("sample_annotation", target["token"])
                target["instance_token"] = str(annotation["instance_token"])
                target["visibility_token"] = str(annotation["visibility_token"])
                target["global_center"] = np.asarray(annotation["translation"], float)
                annotations[target["token"]] = annotation
            matrices = clean_data["lidar2img"][0].detach().float().cpu().numpy()
            image_hw = tuple(int(value) for value in clean_image.shape[-2:])
            physical_by_token = {
                target["token"]: physical_fields(target, matrices, image_hw) for target in targets
            }
            if frame_idx <= 2:
                with torch.no_grad():
                    _, _, clean_feats = features(model, clean_image)
                    clean_feats = clean_feats.detach()
                    clean_pre = clean_state
                    if frame_idx == 0:
                        canonical, canonical_state, _ = run_head(
                            model, clean_meta, clean_data, clean_feats, False, clean_pre)
                        output_a, clean_state, capture_disabled = nohistory_head(
                            model, clean_meta, clean_data, clean_feats, False, clean_pre,
                            enabled=False)
                        output_equal, output_diff = compare_outputs(canonical, output_a)
                        state_equal, state_diff = compare_states(canonical_state, clean_state)
                        if not output_equal or not state_equal:
                            raise RuntimeError("supplement disabled wrapper changed B0")
                        equivalence.append({
                            "protocol": args.protocol, "scene_token": scene,
                            "sample_token": token, "frame_idx": frame_idx,
                            "output_bitwise_equal": output_equal,
                            "output_max_abs_diff": output_diff,
                            "memory_bitwise_equal": state_equal,
                            "memory_max_abs_diff": state_diff,
                            "canonical_query_count": capture_disabled["transformer_query_count"],
                        })
                    else:
                        output_a, clean_state, _ = run_head(
                            model, clean_meta, clean_data, clean_feats, True, clean_pre)
                    output_f, _, _ = nohistory_head(
                        model, clean_meta, clean_data, clean_feats, frame_idx > 0, clean_pre,
                        enabled=True)
                logits_a, boxes_a = output_arrays(output_a, pc_range)
                logits_f, boxes_f = output_arrays(output_f, pc_range)
                context = frame_context(clean_dataset.data_infos[index], clean_dataset)
                matches_a = global_deployed_matches(logits_a, boxes_a, targets, context)
                matches_f = global_deployed_matches(logits_f, boxes_f, targets, context)
                for target in targets:
                    metrics_a = gt_metrics(logits_a, boxes_a, target, matches_a)
                    metrics_f = gt_metrics(logits_f, boxes_f, target, matches_f)
                    row = {
                        "unit_id": f"{args.protocol}:{token}:{target['token']}",
                        "protocol": args.protocol, "scene_token": scene,
                        "sample_token": token, "frame_idx": frame_idx,
                        "gt_token": target["token"],
                        "instance_token": target["instance_token"],
                        "gt_class": target["name"],
                        **physical_by_token[target["token"]],
                        "supplement_role": "pre_fault_anchor",
                    }
                    for condition, metrics in (("A", metrics_a), ("F", metrics_f)):
                        for key, value in metrics.items():
                            row[f"{condition}_{key}"] = value
                    rows.append(row)
                del output_a, output_f, clean_feats
            else:
                fault_meta, fault_image, fault_data = unpack(fault_dataset[index], device)
                frame_source = expected_active[expected_active.sample_token == token]
                frame_missing = [
                    row for row in frame_source.itertuples(index=False) if row.unit_id in missing_active_ids
                ]
                output_f = None
                if frame_missing:
                    with torch.no_grad():
                        _, _, fault_feats = features(model, fault_image)
                        fault_feats = fault_feats.detach()
                        output_f, _, _ = nohistory_head(
                            model, fault_meta, fault_data, fault_feats, True, initial, enabled=True)
                    logits_f, boxes_f = output_arrays(output_f, pc_range)
                    context = frame_context(fault_dataset.data_infos[index], fault_dataset)
                    matches_f = global_deployed_matches(logits_f, boxes_f, targets, context)
                for target in targets:
                    unit_id = f"{args.protocol}:{token}:{target['token']}"
                    if unit_id not in abcd_ids:
                        continue
                    row = {
                        "unit_id": unit_id, "protocol": args.protocol,
                        "scene_token": scene, "sample_token": token,
                        "frame_idx": frame_idx, "gt_token": target["token"],
                        "instance_token": target["instance_token"],
                        "gt_class": target["name"],
                        **physical_by_token[target["token"]],
                    }
                    if unit_id in cached_ids:
                        row["supplement_role"] = "reuse_existing_F"
                    else:
                        if output_f is None:
                            raise RuntimeError("missing F output for uncached row")
                        metrics_f = gt_metrics(logits_f, boxes_f, target, matches_f)
                        row["supplement_role"] = "missing_F_computed"
                        for key, value in metrics_f.items():
                            row[f"F_{key}"] = value
                    rows.append(row)
                if output_f is not None:
                    del output_f, fault_feats
            torch.cuda.empty_cache()
        prelude_count = sum(row["supplement_role"] == "pre_fault_anchor" for row in rows)
        active_count = sum(row["frame_idx"] >= 3 for row in rows)
        computed_missing = sum(row["supplement_role"] == "missing_F_computed" for row in rows)
        if active_count != len(expected_active) or computed_missing != len(missing_active_ids):
            raise RuntimeError(
                f"supplement coverage mismatch active={active_count}/{len(expected_active)} "
                f"missing={computed_missing}/{len(missing_active_ids)}")
        atomic_csv(output_dir / f"{scene}.csv", rows)
        atomic_csv(output_dir / f"{scene}.equivalence.csv", equivalence)
        atomic_json(output_dir / f"{scene}.complete.json", {
            "schema_version": SCHEMA,
            "scene_split_sha256": validation["scene_split_sha256"],
            "protocol": args.protocol, "scene_token": scene,
            "rows": len(rows), "pre_fault_rows": prelude_count,
            "active_rows": active_count, "existing_F_reused": active_count - computed_missing,
            "missing_F_computed": computed_missing,
            "complete": True,
        })
        status_update()
        print(
            f"completed supplement {args.protocol}/{scene} rows={len(rows)} "
            f"reused_F={active_count - computed_missing} computed_F={computed_missing}",
            flush=True,
        )
        if STOP_REQUESTED:
            print("stop requested; current supplement scene saved", flush=True)
            break


if __name__ == "__main__":
    main()
