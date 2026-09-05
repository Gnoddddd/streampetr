#!/usr/bin/env python3
"""Incremental train-split A/B/C/D replay for the CTEP P0 audit."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import signal
import sys
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

from scripts.audit_dark_target_recoverability import (  # noqa: E402
    features,
    local_gt,
    run_head,
    snapshot,
    unpack,
)
from scripts.audit_temporal_state_counterfactual import (  # noqa: E402
    output_metrics,
    state_max_difference,
)


REPORT = ROOT / "reports/full_nuscenes/ctep_method_activation"
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
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def protocol_dataset(config, schedule: Path | None):
    data_config = copy.deepcopy(config.data.test)
    nodes = [node for node in data_config.pipeline
             if node.get("type") == "ApplyPartialObservation"]
    if len(nodes) != 1:
        raise RuntimeError(f"expected one ApplyPartialObservation, got {len(nodes)}")
    nodes[0]["schedule_file"] = None if schedule is None else str(schedule)
    data_config.test_mode = True
    return build_dataset(data_config)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty scene checkpoint: {path}")
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


def scene_rows() -> list[dict]:
    with (REPORT / "scene_list.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def completed(protocol: str, scene: str, scene_hash: str) -> bool:
    meta_path = REPORT / "incremental/p0" / protocol / f"{scene}.complete.json"
    csv_path = REPORT / "incremental/p0" / protocol / f"{scene}.csv"
    if not meta_path.is_file() or not csv_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    return bool(
        meta.get("complete") is True
        and meta.get("schema_version") == SCHEMA
        and meta.get("scene_list_sha256") == scene_hash
        and meta.get("protocol") == protocol
        and meta.get("scene_token") == scene
        and int(meta.get("rows", -1)) > 0
    )


def match_context(info: dict, dataset) -> dict:
    return {
        "lidar2ego_rotation": Quaternion(info["lidar2ego_rotation"]).rotation_matrix,
        "lidar2ego_translation": np.asarray(info["lidar2ego_translation"], float),
        "ego2global_rotation": Quaternion(info["ego2global_rotation"]).rotation_matrix,
        "ego2global_translation": np.asarray(info["ego2global_translation"], float),
        "class_range": dataset.eval_detection_configs.class_range,
    }


def finite_delta(left, right) -> float:
    left, right = float(left), float(right)
    return left - right if math.isfinite(left) and math.isfinite(right) else float("nan")


def term(left: dict, right: dict) -> tuple[bool, bool, float]:
    eligible = bool(left["candidate"] and right["candidate"])
    if not eligible:
        return False, False, 0.0
    loss = max(0.0, float(left["s_pos"]) - float(right["s_pos"]))
    return True, loss > 0.0, loss


def update_status() -> None:
    scenes = scene_rows()
    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    lines = ["# PARTIAL STATUS", "", "`P0_RUNNING`", "", "| protocol | scenes | rows |",
             "|---|---:|---:|"]
    all_complete = True
    for protocol in PROTOCOLS:
        directory = REPORT / "incremental/p0" / protocol
        metas = []
        for path in directory.glob("*.complete.json") if directory.exists() else []:
            value = json.loads(path.read_text())
            if value.get("complete") is True:
                metas.append(value)
        tokens = sorted(str(value["scene_token"]) for value in metas)
        rows = sum(int(value["rows"]) for value in metas)
        progress["p0"][protocol] = {"completed_scenes": tokens, "rows": rows}
        lines.append(f"| {protocol} | {len(tokens)}/{len(scenes)} | {rows} |")
        all_complete &= len(tokens) == len(scenes)
    progress["status"] = "P0_COMPLETE_ANALYSIS_PENDING" if all_complete else "P0_RUNNING"
    atomic_json(REPORT / "progress_manifest.json", progress)
    lines += ["", "Resume:", "", "```bash"]
    lines += [f"python scripts/run_ctep_p0.py --protocol {key}" for key in PROTOCOLS]
    lines += ["```", ""]
    temporary = REPORT / "PARTIAL_STATUS.md.tmp"
    temporary.write_text("\n".join(lines))
    os.replace(temporary, REPORT / "PARTIAL_STATUS.md")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    manifest = json.loads((REPORT / "audit_manifest.json").read_text())
    selected_scenes = scene_rows()
    pending = [row for row in selected_scenes if not completed(
        args.protocol, row["scene_token"], manifest["scene_list_sha256"]
    )]
    if args.max_scenes is not None:
        pending = pending[:args.max_scenes]
    if not pending:
        print(f"no pending scenes for {args.protocol}")
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
    cfg.model.train_cfg = None
    clean_dataset = protocol_dataset(cfg, None)
    fault_dataset = protocol_dataset(cfg, PROTOCOLS[args.protocol])
    clean_token_to_index = {
        str(info["token"]): index for index, info in enumerate(clean_dataset.data_infos)
    }
    fault_token_to_index = {
        str(info["token"]): index for index, info in enumerate(fault_dataset.data_infos)
    }
    if clean_token_to_index != fault_token_to_index:
        raise RuntimeError("Clean/Fault train dataset token indexes differ")
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(CHECKPOINT), map_location="cpu")
    model = model.to(device).eval()
    head = model.pts_bbox_head
    head.reset_memory()
    initial = snapshot(head)
    pc_range = head.pc_range.detach()
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(DATA), verbose=False)

    total = len(selected_scenes)
    done_before = total - len([row for row in selected_scenes if not completed(
        args.protocol, row["scene_token"], manifest["scene_list_sha256"]
    )])
    for offset, scene_row in enumerate(pending, 1):
        scene = scene_row["scene_token"]
        sample_tokens = json.loads(scene_row["sample_tokens_0_12"])
        indexes = [clean_token_to_index[token] for token in sample_tokens]
        clean_state = initial
        fault_state = initial
        rows = []
        with torch.no_grad():
            for frame_idx, index in enumerate(indexes):
                info = clean_dataset.data_infos[index]
                if str(info["scene_token"]) != scene or int(info["frame_idx"]) != frame_idx:
                    raise RuntimeError(f"scene/index mismatch {scene}/{index}/{frame_idx}")
                clean_meta, clean_image, clean_data = unpack(clean_dataset[index], device)
                fault_meta, fault_image, fault_data = unpack(fault_dataset[index], device)
                token = str(clean_meta["sample_idx"])
                if str(fault_meta["sample_idx"]) != token:
                    raise RuntimeError(f"paired token mismatch {args.protocol}/{token}")
                _, _, clean_feats = features(model, clean_image)
                _, _, fault_feats = features(model, fault_image)
                clean_pre, fault_pre = clean_state, fault_state
                state_difference = state_max_difference(clean_pre, fault_pre)
                output_a, clean_state, _ = run_head(
                    model, clean_meta, clean_data, clean_feats, frame_idx > 0, clean_pre
                )
                output_d, fault_state, _ = run_head(
                    model, fault_meta, fault_data, fault_feats, frame_idx > 0, fault_pre
                )
                if frame_idx < 3:
                    continue
                output_b = run_head(
                    model, fault_meta, fault_data, fault_feats, True, clean_pre
                )[0]
                output_c = run_head(
                    model, clean_meta, clean_data, clean_feats, True, fault_pre
                )[0]
                targets = local_gt(nusc, token)
                for target in targets:
                    annotation = nusc.get("sample_annotation", target["token"])
                    target["global_center"] = np.asarray(annotation["translation"], float)
                    target["instance_token"] = str(annotation["instance_token"])
                    target["visibility_token"] = str(annotation["visibility_token"])
                context = match_context(info, clean_dataset)
                for target in targets:
                    values = {
                        "A": output_metrics(output_a, target, targets, pc_range, context),
                        "B": output_metrics(output_b, target, targets, pc_range, context),
                        "C": output_metrics(output_c, target, targets, pc_range, context),
                        "D": output_metrics(output_d, target, targets, pc_range, context),
                    }
                    eligible_ac, active_ac, loss_ac = term(values["A"], values["C"])
                    eligible_bd, active_bd, loss_bd = term(values["B"], values["D"])
                    clean_correct = bool(values["A"]["tp"])
                    lost = bool(clean_correct and not values["D"]["tp"])
                    retained = bool(clean_correct and values["D"]["tp"])
                    history_sensitive = bool(
                        lost and ((values["B"]["tp"] and not values["D"]["tp"])
                                  or (values["A"]["tp"] and not values["C"]["tp"]))
                    )
                    easy = bool(all(values[key]["tp"] for key in "ABCD"))
                    row = {
                        "unit_id": f"{args.protocol}:{token}:{target['token']}",
                        "protocol": args.protocol,
                        "scene_token": scene,
                        "sample_token": token,
                        "frame_idx": frame_idx,
                        "gt_token": target["token"],
                        "instance_token": target["instance_token"],
                        "gt_class": target["name"],
                        "distance_m": float(np.linalg.norm(target["center"][:2])),
                        "visibility_token": target["visibility_token"],
                        "state_max_abs_diff": state_difference,
                        "clean_correct": clean_correct,
                        "lost": lost,
                        "retained": retained,
                        "history_sensitive_lost": history_sensitive,
                        "easy": easy,
                    }
                    for condition, value in values.items():
                        for key in ("candidate", "qplus", "s_pos", "rank", "margin", "topk", "tp"):
                            row[f"{condition}_{key}"] = value[key]
                    row.update({
                        "A_minus_C_s_pos": finite_delta(values["A"]["s_pos"], values["C"]["s_pos"]),
                        "B_minus_D_s_pos": finite_delta(values["B"]["s_pos"], values["D"]["s_pos"]),
                        "eligible_AC": eligible_ac,
                        "eligible_BD": eligible_bd,
                        "ctep_eligible": eligible_ac or eligible_bd,
                        "active_AC": active_ac,
                        "active_BD": active_bd,
                        "ctep_active": active_ac or active_bd,
                        "L_AC": loss_ac,
                        "L_BD": loss_bd,
                        "L_CTEP": loss_ac + loss_bd,
                    })
                    rows.append(row)
                del output_b, output_c, output_a, output_d, clean_feats, fault_feats

        output_dir = REPORT / "incremental/p0" / args.protocol
        atomic_csv(output_dir / f"{scene}.csv", rows)
        atomic_json(output_dir / f"{scene}.complete.json", {
            "schema_version": SCHEMA,
            "scene_list_sha256": manifest["scene_list_sha256"],
            "protocol": args.protocol,
            "scene_token": scene,
            "rows": len(rows),
            "clean_correct_rows": sum(bool(row["clean_correct"]) for row in rows),
            "lost_rows": sum(bool(row["lost"]) for row in rows),
            "retained_rows": sum(bool(row["retained"]) for row in rows),
            "history_sensitive_lost_rows": sum(
                bool(row["history_sensitive_lost"]) for row in rows
            ),
            "replay_frames": len(indexes),
            "complete": True,
        })
        update_status()
        print(
            f"completed {args.protocol} scene {done_before + offset}/{total} "
            f"token={scene} rows={len(rows)}",
            flush=True,
        )
        if STOP_REQUESTED:
            print("stop requested; current scene saved", flush=True)
            break


if __name__ == "__main__":
    main()
