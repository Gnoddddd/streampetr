#!/usr/bin/env python3
"""Real StreamPETR detection-vs-CTEP gradient audit on frozen P1 units."""

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
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmcv.utils import import_modules_from_strings  # noqa: E402
from mmdet3d.datasets import build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from nuscenes.nuscenes import NuScenes  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402

from analysis.ctep_objective import ctep_term, disabled_detection_loss  # noqa: E402
from scripts.audit_dark_target_recoverability import (  # noqa: E402
    features,
    local_gt,
    run_head,
    snapshot,
    unpack,
)
from scripts.audit_temporal_state_counterfactual import output_metrics  # noqa: E402


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
    value = copy.deepcopy(config.data.test)
    nodes = [node for node in value.pipeline if node.get("type") == "ApplyPartialObservation"]
    if len(nodes) != 1:
        raise RuntimeError(f"expected one ApplyPartialObservation, got {len(nodes)}")
    nodes[0]["schedule_file"] = None if schedule is None else str(schedule)
    value.test_mode = True
    return build_dataset(value)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"empty checkpoint {path}")
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


def context(info: dict, dataset) -> dict:
    return {
        "lidar2ego_rotation": Quaternion(info["lidar2ego_rotation"]).rotation_matrix,
        "lidar2ego_translation": np.asarray(info["lidar2ego_translation"], float),
        "ego2global_rotation": Quaternion(info["ego2global_rotation"]).rotation_matrix,
        "ego2global_translation": np.asarray(info["ego2global_translation"], float),
        "class_range": dataset.eval_detection_configs.class_range,
    }


def output_with_query(model, meta, data, feats, prev_exists, state):
    captured = []

    def hook(_module, arguments):
        captured.append(arguments[0])

    handle = model.pts_bbox_head.cls_branches[-1].register_forward_pre_hook(hook)
    try:
        output, next_state, _ = run_head(
            model, meta, data, feats, prev_exists, state
        )
    finally:
        handle.remove()
    # This checkpoint shares one classification-branch module across all six
    # decoder levels, so the module hook fires once per level.  The final call
    # is the final-decoder query representation used by all_cls_scores[-1].
    if not captured:
        raise RuntimeError("final query representation was not captured")
    return output, next_state, captured[-1]


def detection_loss(head, dataset, index: int, output, device) -> torch.Tensor:
    annotation = dataset.get_ann_info(index)
    boxes = annotation["gt_bboxes_3d"]
    labels = torch.as_tensor(annotation["gt_labels_3d"], dtype=torch.long, device=device)
    losses = head.loss([boxes], [labels], output)
    selected = [value for key, value in losses.items() if "dn_loss" not in key]
    if not selected:
        raise RuntimeError("no original detection loss tensors")
    return torch.stack([value.float() for value in selected]).sum()


def flatten(values: list[torch.Tensor | None], like: list[torch.Tensor]) -> torch.Tensor:
    pieces = []
    for value, reference in zip(values, like):
        if value is None:
            pieces.append(torch.zeros(reference.numel(), device=reference.device))
        else:
            pieces.append(value.reshape(-1).float())
    return torch.cat(pieces) if pieces else torch.empty(0)


def relation(left: torch.Tensor, right: torch.Tensor) -> dict:
    left = left.detach().float().reshape(-1)
    right = right.detach().float().reshape(-1)
    left_norm = float(torch.linalg.vector_norm(left).item())
    right_norm = float(torch.linalg.vector_norm(right).item())
    dot = float(torch.dot(left, right).item())
    cosine = dot / (left_norm * right_norm) if left_norm > 0 and right_norm > 0 else math.nan
    return {
        "ctep_grad_norm": left_norm,
        "detection_grad_norm": right_norm,
        "gradient_dot": dot,
        "gradient_cosine": cosine,
        "gradient_conflict": dot < 0,
        "nonzero_cosine": left_norm > 0 and right_norm > 0,
    }


def max_gradient_difference(left, right) -> float:
    differences = []
    for a, b in zip(left, right):
        if a is None and b is None:
            differences.append(0.0)
        elif a is None or b is None:
            return math.inf
        else:
            differences.append(float((a.detach() - b.detach()).abs().max().item()))
    return max(differences, default=0.0)


def gradient_rows_for_condition(
    condition: str,
    output,
    query_repr: torch.Tensor,
    references: dict[str, dict],
    targets: dict[str, dict],
    units: list[dict],
    det_loss: torch.Tensor,
    head,
) -> tuple[list[dict], dict]:
    cls_scores = output["all_cls_scores"]
    parameters = list(head.cls_branches[-1].parameters())
    differentiation_targets = [cls_scores, query_repr, *parameters]
    det_gradients = torch.autograd.grad(
        det_loss, differentiation_targets, retain_graph=True, allow_unused=True
    )
    disabled = disabled_detection_loss(det_loss)
    disabled_gradients = torch.autograd.grad(
        disabled, differentiation_targets, retain_graph=True, allow_unused=True
    )
    equivalence = {
        "condition": condition,
        "detection_loss": float(det_loss.detach().item()),
        "disabled_loss": float(disabled.detach().item()),
        "loss_tensor_identity": disabled is det_loss,
        "output_tensor_leaves": 2,
        "output_max_abs_diff": 0.0,
        "gradient_max_abs_diff": max_gradient_difference(det_gradients, disabled_gradients),
    }
    rows = []
    for unit in units:
        target = targets[unit["gt_token"]]
        term_name = "AC" if condition == "C" else "BD"
        if term_name not in json.loads(unit["active_terms"]):
            continue
        reference_condition = "A" if condition == "C" else "B"
        reference = references[unit["gt_token"]][reference_condition]
        target_value = references[unit["gt_token"]][condition]
        query = int(target_value["qplus"])
        label = int(target["label"])
        target_logit = cls_scores[-1, 0, query, label]
        reference_score = target_logit.new_tensor(float(reference["s_pos"]))
        loss = ctep_term(reference_score, target_logit)
        if not float(loss.detach().item()) > 0:
            raise RuntimeError(f"frozen active term became inactive: {unit['unit_id']}/{term_name}")
        ctep_gradients = torch.autograd.grad(
            loss, differentiation_targets, retain_graph=True, allow_unused=True
        )
        selected_logit_ctep = ctep_gradients[0][-1, 0, query, label].reshape(1)
        selected_logit_det = det_gradients[0][-1, 0, query, label].reshape(1)
        selected_query_ctep = ctep_gradients[1][0, query]
        selected_query_det = det_gradients[1][0, query]
        parameter_ctep = flatten(list(ctep_gradients[2:]), parameters)
        parameter_det = flatten(list(det_gradients[2:]), parameters)
        base = {
            "unit_id": unit["unit_id"],
            "protocol": unit["protocol"],
            "scene_token": unit["scene_token"],
            "sample_token": unit["sample_token"],
            "frame_idx": int(unit["frame_idx"]),
            "gt_token": unit["gt_token"],
            "instance_token": unit["instance_token"],
            "gt_class": unit["gt_class"],
            "gradient_stratum": unit["gradient_stratum"],
            "term": term_name,
            "reference_condition": reference_condition,
            "target_condition": condition,
            "reference_s_pos": float(reference["s_pos"]),
            "target_s_pos": float(target_value["s_pos"]),
            "target_query": query,
            "ctep_loss": float(loss.detach().item()),
            "ctep_descent_increases_target_s_pos": float(selected_logit_ctep.item()) < 0,
        }
        for level, left, right in (
            ("selected_gt_class_logit", selected_logit_ctep, selected_logit_det),
            ("selected_query_representation", selected_query_ctep, selected_query_det),
            ("final_cls_head_parameters", parameter_ctep, parameter_det),
        ):
            rows.append({**base, "gradient_level": level, **relation(left, right)})
    return rows, equivalence


def update_status() -> None:
    units = pd.read_csv(REPORT / "gradient_units.csv")
    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    lines = ["# PARTIAL STATUS", "", "`P1_GRADIENT_RUNNING`", "",
             "P0 is complete and passed. Partial P1 data must not be used for Activation Go/No-Go.",
             "", "| protocol | scenes | gradient rows |", "|---|---:|---:|"]
    all_complete = True
    for protocol in PROTOCOLS:
        expected = units.loc[units.protocol == protocol, "scene_token"].nunique()
        directory = REPORT / "incremental/p1_gradient" / protocol
        metas = [json.loads(path.read_text()) for path in directory.glob("*.complete.json")]
        rows = sum(int(meta["gradient_rows"]) for meta in metas)
        tokens = sorted(str(meta["scene_token"]) for meta in metas)
        progress["p1"][protocol] = {"completed_scenes": tokens, "rows": rows}
        lines.append(f"| {protocol} | {len(tokens)}/{expected} | {rows} |")
        all_complete &= len(tokens) == expected
    progress["status"] = "P1_COMPLETE_ANALYSIS_PENDING" if all_complete else "P1_GRADIENT_RUNNING"
    atomic_json(REPORT / "progress_manifest.json", progress)
    lines += ["", "Resume:", "", "```bash"]
    lines += [f"python scripts/run_ctep_p1_gradients.py --protocol {key}" for key in PROTOCOLS]
    lines += ["```", ""]
    temporary = REPORT / "PARTIAL_STATUS.md.tmp"
    temporary.write_text("\n".join(lines))
    os.replace(temporary, REPORT / "PARTIAL_STATUS.md")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    decision = json.loads((REPORT / "p0_decision.json").read_text())
    if decision["verdict"] != "P0_GO_P1_REQUIRED":
        raise RuntimeError(f"P1 locked by {decision['verdict']}")
    manifest = json.loads((REPORT / "audit_manifest.json").read_text())
    units_frame = pd.read_csv(REPORT / "gradient_units.csv")
    units_frame = units_frame[units_frame.protocol == args.protocol]
    selected_scenes = sorted(units_frame.scene_token.unique())
    output_dir = REPORT / "incremental/p1_gradient" / args.protocol
    pending = []
    for scene in selected_scenes:
        meta_path = output_dir / f"{scene}.complete.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if (meta.get("complete") is True and meta.get("schema_version") == SCHEMA
                    and meta.get("scene_list_sha256") == manifest["scene_list_sha256"]):
                continue
        pending.append(scene)
    if args.max_scenes is not None:
        pending = pending[:args.max_scenes]
    if not pending:
        print(f"no pending gradient scenes for {args.protocol}")
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
    token_to_index = {
        str(info["token"]): index for index, info in enumerate(clean_dataset.data_infos)
    }
    if token_to_index != {
        str(info["token"]): index for index, info in enumerate(fault_dataset.data_infos)
    }:
        raise RuntimeError("paired dataset index mismatch")
    scene_list = pd.read_csv(REPORT / "scene_list.csv")
    scene_tokens = {
        row.scene_token: json.loads(row.sample_tokens_0_12)
        for row in scene_list.itertuples(index=False)
    }
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(CHECKPOINT), map_location="cpu")
    model = model.to(device).eval()
    head = model.pts_bbox_head
    head.reset_memory()
    initial = snapshot(head)
    pc_range = head.pc_range.detach()
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(DATA), verbose=False)

    for scene in pending:
        scene_units_frame = units_frame[units_frame.scene_token == scene]
        by_sample = {
            token: group.to_dict("records")
            for token, group in scene_units_frame.groupby("sample_token")
        }
        clean_state, fault_state = initial, initial
        gradient_rows = []
        equivalence_rows = []
        for frame_idx, token in enumerate(scene_tokens[scene]):
            index = token_to_index[token]
            info = clean_dataset.data_infos[index]
            clean_meta, clean_image, clean_data = unpack(clean_dataset[index], device)
            fault_meta, fault_image, fault_data = unpack(fault_dataset[index], device)
            with torch.no_grad():
                _, _, clean_feats = features(model, clean_image)
                _, _, fault_feats = features(model, fault_image)
                clean_feats = clean_feats.detach()
                fault_feats = fault_feats.detach()
                clean_pre, fault_pre = clean_state, fault_state
                output_a, clean_state, _ = run_head(
                    model, clean_meta, clean_data, clean_feats, frame_idx > 0, clean_pre
                )
            frame_units = by_sample.get(token, [])
            if not frame_units:
                with torch.no_grad():
                    _, fault_state, _ = run_head(
                        model, fault_meta, fault_data, fault_feats, frame_idx > 0, fault_pre
                    )
                continue
            targets_list = local_gt(nusc, token)
            for target in targets_list:
                annotation = nusc.get("sample_annotation", target["token"])
                target["global_center"] = np.asarray(annotation["translation"], float)
            targets = {target["token"]: target for target in targets_list}
            frame_context = context(info, clean_dataset)
            references: dict[str, dict] = {}
            with torch.no_grad():
                output_b = run_head(
                    model, fault_meta, fault_data, fault_feats, frame_idx > 0, clean_pre
                )[0]
                for unit in frame_units:
                    target = targets[unit["gt_token"]]
                    references[unit["gt_token"]] = {
                        "A": output_metrics(output_a, target, targets_list, pc_range, frame_context),
                        "B": output_metrics(output_b, target, targets_list, pc_range, frame_context),
                    }
            output_c, _, query_c = output_with_query(
                model, clean_meta, clean_data, clean_feats, frame_idx > 0, fault_pre
            )
            for unit in frame_units:
                target = targets[unit["gt_token"]]
                references[unit["gt_token"]]["C"] = output_metrics(
                    output_c, target, targets_list, pc_range, frame_context
                )
                if int(references[unit["gt_token"]]["C"]["qplus"]) != int(unit["C_qplus"]):
                    raise RuntimeError(f"C q+ replay mismatch {unit['unit_id']}")
            det_c = detection_loss(head, clean_dataset, index, output_c, device)
            rows, equivalence = gradient_rows_for_condition(
                "C", output_c, query_c, references, targets, frame_units, det_c, head
            )
            gradient_rows.extend(rows)
            equivalence_rows.append({
                "protocol": args.protocol, "scene_token": scene, "sample_token": token,
                "frame_idx": frame_idx, **equivalence,
            })
            del output_c, query_c, det_c

            output_d, fault_state, query_d = output_with_query(
                model, fault_meta, fault_data, fault_feats, frame_idx > 0, fault_pre
            )
            for unit in frame_units:
                target = targets[unit["gt_token"]]
                references[unit["gt_token"]]["D"] = output_metrics(
                    output_d, target, targets_list, pc_range, frame_context
                )
                if int(references[unit["gt_token"]]["D"]["qplus"]) != int(unit["D_qplus"]):
                    raise RuntimeError(f"D q+ replay mismatch {unit['unit_id']}")
            det_d = detection_loss(head, clean_dataset, index, output_d, device)
            rows, equivalence = gradient_rows_for_condition(
                "D", output_d, query_d, references, targets, frame_units, det_d, head
            )
            gradient_rows.extend(rows)
            equivalence_rows.append({
                "protocol": args.protocol, "scene_token": scene, "sample_token": token,
                "frame_idx": frame_idx, **equivalence,
            })
            del output_d, query_d, det_d, output_b, output_a
            torch.cuda.empty_cache()

        atomic_csv(output_dir / f"{scene}.csv", gradient_rows)
        atomic_csv(output_dir / f"{scene}.equivalence.csv", equivalence_rows)
        atomic_json(output_dir / f"{scene}.complete.json", {
            "schema_version": SCHEMA,
            "scene_list_sha256": manifest["scene_list_sha256"],
            "protocol": args.protocol,
            "scene_token": scene,
            "events": len(scene_units_frame),
            "active_terms": sum(len(json.loads(value)) for value in scene_units_frame.active_terms),
            "gradient_rows": len(gradient_rows),
            "equivalence_rows": len(equivalence_rows),
            "complete": True,
        })
        update_status()
        print(
            f"completed P1 {args.protocol}/{scene} terms="
            f"{len(gradient_rows) // 3} gradient_rows={len(gradient_rows)}",
            flush=True,
        )
        if STOP_REQUESTED:
            print("stop requested; current P1 scene saved", flush=True)
            break


if __name__ == "__main__":
    main()
