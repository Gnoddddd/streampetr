#!/usr/bin/env python3
"""Incremental P0 NoHistory anchor audit on the frozen train population."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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
from mmdet3d.datasets import build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from nuscenes.nuscenes import NuScenes  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402

from analysis.fault_boundary_root_cause import candidate_pool_statistics  # noqa: E402
from scripts.audit_dark_target_recoverability import (  # noqa: E402
    features, local_gt, physical, run_head, snapshot, unpack,
)
from scripts.audit_temporal_state_counterfactual import global_deployed_matches  # noqa: E402
from scripts.run_temporal_representation_p0 import (  # noqa: E402
    compare_outputs, compare_states, validate_source_replay,
)


SOURCE = ROOT / "reports/full_nuscenes/ctep_method_activation"
REPORT = ROOT / "reports/full_nuscenes/bd_temporal_support_audit"
CONFIG = ROOT / "configs/full_nuscenes/stream_petr_r50_90e_ctep_train_audit.py"
CHECKPOINT = ROOT / "checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth"
DATA = ROOT / "data/nuscenes"
PROTOCOLS = {
    "blur_back": ROOT / "protocols/presets/motion_blur_back_10f_s09.json",
    "crash_back": ROOT / "protocols/presets/camera_crash_back_10f.json",
    "dark_back": ROOT / "protocols/presets/dark_back_10f_s09.json",
}
SCHEMA = 1
STOP_REQUESTED = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=tuple(PROTOCOLS), required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--scene-token")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_dataset(config, schedule: Path | None):
    value = copy.deepcopy(config.data.test)
    nodes = [node for node in value.pipeline if node.get("type") == "ApplyPartialObservation"]
    if len(nodes) != 1:
        raise RuntimeError(f"expected one ApplyPartialObservation, got {len(nodes)}")
    nodes[0]["schedule_file"] = None if schedule is None else str(schedule)
    value.test_mode = True
    return build_dataset(value)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"empty checkpoint: {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def frame_context(info: dict, dataset) -> dict:
    return {
        "lidar2ego_rotation": Quaternion(info["lidar2ego_rotation"]).rotation_matrix,
        "lidar2ego_translation": np.asarray(info["lidar2ego_translation"], float),
        "ego2global_rotation": Quaternion(info["ego2global_rotation"]).rotation_matrix,
        "ego2global_translation": np.asarray(info["ego2global_translation"], float),
        "class_range": dataset.eval_detection_configs.class_range,
    }


def gt_metrics(logits, boxes, target, deployed_matches) -> dict:
    pool = candidate_pool_statistics(
        logits, boxes, target["center"], int(target["label"]), 100, 2.0)
    return {
        "candidate": pool["candidate_available"],
        "qplus": pool["best_query"],
        "s_pos": pool["s_pos"],
        "rank": pool["rank"],
        "margin": pool["margin"],
        "topk": 0 < pool["rank"] <= 100,
        "tp": target["token"] in deployed_matches,
    }


def nohistory_head(model, meta, data, feats, prev_exists, state, *, enabled: bool):
    """Use the real temporal path, optionally removing all effective history."""

    head = model.pts_bbox_head
    if "temporal_alignment" in head.__dict__:
        raise RuntimeError("unexpected temporal_alignment instance override")
    original = head.temporal_alignment
    capture = {}

    def wrapper(query_pos, tgt, reference_points):
        result = original(query_pos, tgt, reference_points)
        capture["original_shapes"] = [tuple(value.shape) for value in result[:5]]
        capture["original_current_tgt"] = result[0][:, :head.num_query].detach().clone()
        capture["original_current_query_pos"] = result[1][:, :head.num_query].detach().clone()
        capture["original_current_reference"] = result[2][:, :head.num_query].detach().clone()
        if not enabled:
            return result
        rec_ego_pose = result[5][:, :head.num_query]
        reduced = (
            result[0][:, :head.num_query],
            result[1][:, :head.num_query],
            result[2][:, :head.num_query],
            None,
            None,
            rec_ego_pose,
        )
        capture["returned_current_tgt"] = reduced[0].detach().clone()
        capture["returned_current_query_pos"] = reduced[1].detach().clone()
        capture["returned_current_reference"] = reduced[2].detach().clone()
        return reduced

    def transformer_pre(_module, arguments):
        capture["transformer_query_count"] = int(arguments[1].shape[1])
        capture["transformer_temp_memory_is_none"] = arguments[5] is None
        capture["transformer_temp_pos_is_none"] = arguments[6] is None

    head.temporal_alignment = wrapper
    handle = head.transformer.register_forward_pre_hook(transformer_pre)
    try:
        output, next_state, _ = run_head(model, meta, data, feats, prev_exists, state)
    finally:
        handle.remove()
        del head.temporal_alignment
    if enabled:
        for name in ("tgt", "query_pos", "reference"):
            if not torch.equal(capture[f"original_current_{name}"], capture[f"returned_current_{name}"]):
                raise RuntimeError(f"NoHistory changed current {name}")
        if capture["transformer_query_count"] != int(head.num_query):
            raise RuntimeError("NoHistory query count mismatch")
        if not capture["transformer_temp_memory_is_none"] or not capture["transformer_temp_pos_is_none"]:
            raise RuntimeError("NoHistory temporal memory remained active")
    return output, next_state, capture


def update_status() -> None:
    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    if str(progress.get("status", "")).startswith("FINAL_"):
        return
    population = pd.read_csv(REPORT / "population.csv", usecols=["protocol", "scene_token"])
    lines = [
        "# PARTIAL STATUS", "", "`P0_RUNNING`", "",
        "Partial P0 coverage cannot produce a mechanism or final decision.", "",
        "| protocol | scenes | per-GT rows |", "|---|---:|---:|",
    ]
    complete = True
    stage = {}
    for protocol in PROTOCOLS:
        expected = int(population[population.protocol == protocol].scene_token.nunique())
        directory = REPORT / "incremental/P0" / protocol
        metas = [json.loads(path.read_text()) for path in directory.glob("*.complete.json")]
        tokens = sorted(meta["scene_token"] for meta in metas if meta.get("complete"))
        rows = sum(int(meta["rows"]) for meta in metas if meta.get("complete"))
        stage[protocol] = {"completed_scenes": tokens, "expected_scenes": expected, "rows": rows}
        lines.append(f"| {protocol} | {len(tokens)}/{expected} | {rows} |")
        complete &= len(tokens) == expected
    progress["stages"]["P0"] = stage
    progress["status"] = "P0_COMPLETE_ANALYSIS_PENDING" if complete else "P0_RUNNING"
    atomic_json(REPORT / "progress_manifest.json", progress)
    lines += ["", "Resume:", "", "```bash"]
    lines += [f"python scripts/run_bd_temporal_support_p0.py --protocol {p}" for p in PROTOCOLS]
    lines += ["python scripts/analyze_bd_temporal_support_p0.py", "```", ""]
    temporary = REPORT / "PARTIAL_STATUS.md.tmp"
    temporary.write_text("\n".join(lines))
    os.replace(temporary, REPORT / "PARTIAL_STATUS.md")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    validation = json.loads((REPORT / "source_validation.json").read_text())
    if validation.get("status") != "VALIDATED_BEFORE_FORWARD":
        raise RuntimeError("pre-forward validation missing")
    if sha256(REPORT / "population.csv") != validation["population_sha256"]:
        raise RuntimeError("population hash mismatch")
    population = pd.read_csv(REPORT / "population.csv")
    population = population[population.protocol == args.protocol]
    scene_order = pd.read_csv(SOURCE / "scene_list.csv").scene_token.tolist()
    output_dir = REPORT / "incremental/P0" / args.protocol
    pending = []
    for scene in scene_order:
        path = output_dir / f"{scene}.complete.json"
        if path.exists():
            meta = json.loads(path.read_text())
            if meta.get("complete") and meta.get("schema_version") == SCHEMA \
                    and meta.get("population_sha256") == validation["population_sha256"]:
                continue
        pending.append(scene)
    if args.scene_token:
        pending = [scene for scene in pending if scene == args.scene_token]
    if args.max_scenes is not None:
        pending = pending[:args.max_scenes]
    if not pending:
        print(f"no pending P0 scenes for {args.protocol}")
        update_status()
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
    clean_index = {str(info["token"]): i for i, info in enumerate(clean_dataset.data_infos)}
    if clean_index != {str(info["token"]): i for i, info in enumerate(fault_dataset.data_infos)}:
        raise RuntimeError("paired dataset mismatch")
    scene_rows = pd.read_csv(SOURCE / "scene_list.csv")
    scene_tokens = {
        row.scene_token: json.loads(row.sample_tokens_0_12)
        for row in scene_rows.itertuples(index=False)
    }
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(CHECKPOINT), map_location="cpu")
    model = model.to(device).eval()
    head = model.pts_bbox_head
    if int(head.num_query) != 644 or int(head.num_propagated) != 256 or int(head.memory_len) != 1024:
        raise RuntimeError("NoHistory preregistered dimensions changed")
    head.reset_memory()
    initial = snapshot(head)
    pc_range = head.pc_range.detach()
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(DATA), verbose=False)

    for scene in pending:
        selected = population[population.scene_token == scene]
        by_sample = {token: group.to_dict("records") for token, group in selected.groupby("sample_token")}
        clean_state, fault_state = initial, initial
        rows, equivalence = [], []
        disabled_checked = False
        for frame_idx, token in enumerate(scene_tokens[scene]):
            index = clean_index[token]
            info = clean_dataset.data_infos[index]
            clean_meta, clean_image, clean_data = unpack(clean_dataset[index], device)
            fault_meta, fault_image, fault_data = unpack(fault_dataset[index], device)
            with torch.no_grad():
                _, _, clean_feats = features(model, clean_image)
                _, _, fault_feats = features(model, fault_image)
                clean_feats, fault_feats = clean_feats.detach(), fault_feats.detach()
                clean_pre, fault_pre = clean_state, fault_state
                if by_sample.get(token) and not disabled_checked:
                    canonical_a, canonical_state, _ = run_head(
                        model, clean_meta, clean_data, clean_feats, frame_idx > 0, clean_pre)
                    output_a, clean_state, disabled_capture = nohistory_head(
                        model, clean_meta, clean_data, clean_feats, frame_idx > 0, clean_pre,
                        enabled=False)
                    out_exact, out_diff = compare_outputs(canonical_a, output_a)
                    state_exact, state_diff = compare_states(canonical_state, clean_state)
                    if not out_exact or not state_exact:
                        raise RuntimeError("disabled NoHistory wrapper changed B0")
                    equivalence.append({
                        "protocol": args.protocol, "scene_token": scene,
                        "sample_token": token, "frame_idx": frame_idx,
                        "check": "disabled_wrapper_B0",
                        "output_bitwise_equal": out_exact,
                        "output_max_abs_diff": out_diff,
                        "memory_bitwise_equal": state_exact,
                        "memory_max_abs_diff": state_diff,
                        "canonical_query_count": disabled_capture["transformer_query_count"],
                        "canonical_temp_memory_is_none": disabled_capture["transformer_temp_memory_is_none"],
                    })
                    disabled_checked = True
                else:
                    output_a, clean_state, _ = run_head(
                        model, clean_meta, clean_data, clean_feats, frame_idx > 0, clean_pre)
                output_d, fault_state, _ = run_head(
                    model, fault_meta, fault_data, fault_feats, frame_idx > 0, fault_pre)
            frame_population = by_sample.get(token, [])
            if not frame_population:
                continue
            with torch.no_grad():
                output_e, _, capture_e = nohistory_head(
                    model, clean_meta, clean_data, clean_feats, frame_idx > 0, clean_pre,
                    enabled=True)
                output_f, _, capture_f = nohistory_head(
                    model, fault_meta, fault_data, fault_feats, frame_idx > 0, fault_pre,
                    enabled=True)
            if output_e["all_cls_scores"].shape[2] != 644 or output_f["all_cls_scores"].shape[2] != 644:
                raise RuntimeError("NoHistory output contains propagated queries")
            equivalence.append({
                "protocol": args.protocol, "scene_token": scene,
                "sample_token": token, "frame_idx": frame_idx,
                "check": "nohistory_current_path",
                "output_bitwise_equal": True,
                "output_max_abs_diff": 0.0,
                "memory_bitwise_equal": True,
                "memory_max_abs_diff": 0.0,
                "canonical_query_count": capture_e["original_shapes"][0][1],
                "nohistory_query_count": capture_e["transformer_query_count"],
                "E_temp_memory_is_none": capture_e["transformer_temp_memory_is_none"],
                "F_temp_memory_is_none": capture_f["transformer_temp_memory_is_none"],
            })
            targets_list = local_gt(nusc, token)
            for target in targets_list:
                annotation = nusc.get("sample_annotation", target["token"])
                target["global_center"] = np.asarray(annotation["translation"], float)
            target_map = {target["token"]: target for target in targets_list}
            context = frame_context(info, clean_dataset)
            logits_a = output_a["all_cls_scores"][-1, 0].detach().float().cpu().numpy()
            logits_d = output_d["all_cls_scores"][-1, 0].detach().float().cpu().numpy()
            boxes_a = physical(output_a, pc_range)[-1, 0].detach().float().cpu().numpy()
            boxes_d = physical(output_d, pc_range)[-1, 0].detach().float().cpu().numpy()
            logits_e = output_e["all_cls_scores"][-1, 0].detach().float().cpu().numpy()
            logits_f = output_f["all_cls_scores"][-1, 0].detach().float().cpu().numpy()
            boxes_e = physical(output_e, pc_range)[-1, 0].detach().float().cpu().numpy()
            boxes_f = physical(output_f, pc_range)[-1, 0].detach().float().cpu().numpy()
            matches_e = global_deployed_matches(logits_e, boxes_e, targets_list, context)
            matches_f = global_deployed_matches(logits_f, boxes_f, targets_list, context)
            for frozen in frame_population:
                target = target_map[frozen["gt_token"]]
                # Replayed canonical endpoints guard the frozen population without
                # recomputing the already complete B/C branches.
                enriched = dict(frozen)
                enriched["gt_center"], enriched["gt_label"] = target["center"], target["label"]
                validate_source_replay(enriched, "A", logits_a, boxes_a)
                validate_source_replay(enriched, "D", logits_d, boxes_d)
                metrics_e = gt_metrics(logits_e, boxes_e, target, matches_e)
                metrics_f = gt_metrics(logits_f, boxes_f, target, matches_f)
                a, b = float(frozen["A_s_pos"]), float(frozen["B_s_pos"])
                c, d = float(frozen["C_s_pos"]), float(frozen["D_s_pos"])
                e = float(metrics_e["s_pos"]) if metrics_e["candidate"] else math.nan
                f = float(metrics_f["s_pos"]) if metrics_f["candidate"] else math.nan
                row = dict(frozen)
                row["population"] = "lost" if bool(frozen["lost"]) else "retained"
                for condition, metrics in (("E", metrics_e), ("F", metrics_f)):
                    for key, value in metrics.items():
                        row[f"{condition}_{key}"] = value
                row.update({
                    "G_A": a - e if math.isfinite(e) else math.nan,
                    "G_C": c - e if math.isfinite(e) else math.nan,
                    "G_B": b - f if math.isfinite(f) else math.nan,
                    "G_D": d - f if math.isfinite(f) else math.nan,
                    "current_only_deficit": e - f if math.isfinite(e) and math.isfinite(f) else math.nan,
                })
                rows.append(row)
            del output_a, output_d, output_e, output_f
            torch.cuda.empty_cache()
        if len(rows) != len(selected):
            raise RuntimeError(f"scene row mismatch: {len(rows)} != {len(selected)}")
        if not equivalence:
            raise RuntimeError("missing exactness evidence")
        atomic_csv(output_dir / f"{scene}.csv", rows)
        atomic_csv(output_dir / f"{scene}.equivalence.csv", equivalence)
        atomic_json(output_dir / f"{scene}.complete.json", {
            "schema_version": SCHEMA,
            "population_sha256": validation["population_sha256"],
            "protocol": args.protocol,
            "scene_token": scene,
            "rows": len(rows),
            "equivalence_rows": len(equivalence),
            "complete": True,
        })
        update_status()
        print(f"completed P0 {args.protocol}/{scene} rows={len(rows)}", flush=True)
        if STOP_REQUESTED:
            print("stop requested; current scene saved", flush=True)
            break


if __name__ == "__main__":
    main()
