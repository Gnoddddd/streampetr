#!/usr/bin/env python3
"""Incremental real-graph representation-side CTEP gradient routing audit."""

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
from collections import OrderedDict
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

from analysis.ctep_gradient_routing import (  # noqa: E402
    frozen_sequential_classifier,
    stream_petr_parameter_groups,
    unique_parameters,
)
from analysis.ctep_objective import ctep_term, disabled_detection_loss  # noqa: E402
from scripts.audit_dark_target_recoverability import (  # noqa: E402
    features,
    local_gt,
    run_head,
    snapshot,
    unpack,
)
from scripts.audit_temporal_state_counterfactual import output_metrics  # noqa: E402


SOURCE = ROOT / "reports/full_nuscenes/ctep_method_activation"
REPORT = ROOT / "reports/full_nuscenes/ctep_gradient_routing_audit"
CONFIG = ROOT / "configs/full_nuscenes/stream_petr_r50_90e_ctep_train_audit.py"
CHECKPOINT = ROOT / "checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth"
DATA = ROOT / "data/nuscenes"
PROTOCOLS = {
    "blur_back": ROOT / "protocols/presets/motion_blur_back_10f_s09.json",
    "crash_back": ROOT / "protocols/presets/camera_crash_back_10f.json",
    "dark_back": ROOT / "protocols/presets/dark_back_10f_s09.json",
}
FROZEN_UNITS_SHA256 = "eb733e626930032bcf952a6a609e56ae113ae28f180f126233580c129cb5737f"
GRADIENT_LEVELS = (
    "selected_query_representation",
    "final_decoder_layer_5",
    "final_decoder_temporal_self_attention",
    "all_decoder_temporal_self_attention",
    "temporal_alignment_modules",
)
SCHEMA = 1
STOP_REQUESTED = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=tuple(PROTOCOLS), required=True)
    parser.add_argument("--max-scenes", type=int)
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
        output, next_state, _ = run_head(model, meta, data, feats, prev_exists, state)
    finally:
        handle.remove()
    if len(captured) != len(model.pts_bbox_head.cls_branches):
        raise RuntimeError(f"expected six shared classifier calls, observed {len(captured)}")
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


def flatten_from_map(parameters: OrderedDict, gradients: dict[int, torch.Tensor | None]) -> torch.Tensor:
    pieces = []
    for parameter in parameters.values():
        gradient = gradients.get(id(parameter))
        if gradient is None:
            pieces.append(torch.zeros(parameter.numel(), device=parameter.device))
        else:
            pieces.append(gradient.reshape(-1).float())
    return torch.cat(pieces) if pieces else torch.empty(0)


def relation(ctep: torch.Tensor, detection: torch.Tensor) -> dict:
    ctep = ctep.detach().float().reshape(-1)
    detection = detection.detach().float().reshape(-1)
    ctep_norm = float(torch.linalg.vector_norm(ctep).item())
    detection_norm = float(torch.linalg.vector_norm(detection).item())
    dot = float(torch.dot(ctep, detection).item())
    cosine = dot / (ctep_norm * detection_norm) if ctep_norm > 0 and detection_norm > 0 else math.nan
    return {
        "ctep_grad_norm": ctep_norm,
        "detection_grad_norm": detection_norm,
        "gradient_dot": dot,
        "gradient_cosine": cosine,
        "gradient_conflict": dot < 0,
        "nonzero_cosine": ctep_norm > 0 and detection_norm > 0,
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
    classifier: OrderedDict,
    groups: OrderedDict,
    head,
) -> tuple[list[dict], dict, list[dict]]:
    canonical_logits = output["all_cls_scores"][-1]
    frozen_logits = frozen_sequential_classifier(head.cls_branches[-1], query_repr)
    mapping_equal = torch.equal(canonical_logits, frozen_logits)
    mapping_diff = float((canonical_logits.detach() - frozen_logits.detach()).abs().max().item())
    if not mapping_equal:
        raise RuntimeError(f"detached classifier mapping is not exact: {mapping_diff}")

    unique = unique_parameters(classifier, *groups.values())
    differentiation_targets = [query_repr, *unique]
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
        "output_tensor_identity": True,
        "output_max_abs_diff": 0.0,
        "gradient_max_abs_diff": max_gradient_difference(det_gradients, disabled_gradients),
        "frozen_classifier_logits_bitwise_equal": mapping_equal,
        "frozen_classifier_logits_max_abs_diff": mapping_diff,
    }
    det_parameter_map = {
        id(parameter): gradient for parameter, gradient in zip(unique, det_gradients[1:])
    }
    rows = []
    head_rows = []
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
        target_logit = frozen_logits[0, query, label]
        reference_score = target_logit.new_tensor(float(reference["s_pos"]))
        loss = ctep_term(reference_score, target_logit)
        if not float(loss.detach().item()) > 0:
            raise RuntimeError(f"frozen active term became inactive: {unit['unit_id']}/{term_name}")
        ctep_gradients = torch.autograd.grad(
            loss, differentiation_targets, retain_graph=True, allow_unused=True
        )
        ctep_parameter_map = {
            id(parameter): gradient for parameter, gradient in zip(unique, ctep_gradients[1:])
        }
        classifier_aux = [ctep_parameter_map[id(parameter)] for parameter in classifier.values()]
        head_all_none = all(gradient is None for gradient in classifier_aux)
        head_max_abs = max(
            (float(gradient.detach().abs().max().item()) for gradient in classifier_aux if gradient is not None),
            default=0.0,
        )
        head_norm = math.sqrt(sum(
            float(gradient.detach().float().square().sum().item())
            for gradient in classifier_aux if gradient is not None
        ))
        target_logit_gradient = torch.autograd.grad(loss, target_logit, retain_graph=True)[0]
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
            "target_logit_aux_gradient": float(target_logit_gradient.detach().item()),
            "ctep_descent_increases_target_s_pos": float(target_logit_gradient.item()) < 0,
            "frozen_classifier_logits_bitwise_equal": mapping_equal,
            "classification_head_aux_all_none": head_all_none,
            "classification_head_aux_max_abs": head_max_abs,
            "classification_head_aux_grad_norm": head_norm,
        }
        selected_query_ctep = ctep_gradients[0][0, query]
        selected_query_det = det_gradients[0][0, query]
        rows.append({
            **base,
            "gradient_level": "selected_query_representation",
            **relation(selected_query_ctep, selected_query_det),
        })
        for level, parameters in groups.items():
            rows.append({
                **base,
                "gradient_level": level,
                **relation(
                    flatten_from_map(parameters, ctep_parameter_map),
                    flatten_from_map(parameters, det_parameter_map),
                ),
            })
        head_rows.append({
            key: base[key]
            for key in (
                "unit_id", "protocol", "scene_token", "sample_token", "frame_idx",
                "gt_token", "instance_token", "term", "target_condition",
                "classification_head_aux_all_none", "classification_head_aux_max_abs",
                "classification_head_aux_grad_norm", "frozen_classifier_logits_bitwise_equal",
                "ctep_descent_increases_target_s_pos",
            )
        })
    return rows, equivalence, head_rows


def write_parameter_manifest(classifier: OrderedDict, groups: OrderedDict) -> None:
    path = REPORT / "parameter_group_manifest.csv"
    rows = []
    for name, parameter in classifier.items():
        rows.append({
            "gradient_group": "classification_head_stopped_for_aux",
            "parameter_name": name,
            "shape": json.dumps(list(parameter.shape)),
            "numel": parameter.numel(),
            "source_role": "shared cls_branches[0] fixed classification mapping",
        })
    roles = {
        "final_decoder_layer_5": "entire actual final decoder layer",
        "final_decoder_temporal_self_attention": "final layer attention that concatenates query/temp_memory",
        "all_decoder_temporal_self_attention": "same temporal self-attention in decoder layers 0-5",
        "temporal_alignment_modules": "query/time/ego-pose modules called by temporal_alignment",
    }
    for group, parameters in groups.items():
        for name, parameter in parameters.items():
            rows.append({
                "gradient_group": group,
                "parameter_name": name,
                "shape": json.dumps(list(parameter.shape)),
                "numel": parameter.numel(),
                "source_role": roles[group],
            })
    if path.exists():
        existing = pd.read_csv(path).to_dict("records")
        normalized = [{key: str(value) for key, value in row.items()} for row in rows]
        existing_normalized = [{key: str(value) for key, value in row.items()} for row in existing]
        if existing_normalized != normalized:
            raise RuntimeError("parameter group manifest changed on resume")
    else:
        atomic_csv(path, rows)


def update_status() -> None:
    units = pd.read_csv(REPORT / "gradient_units.csv")
    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    # A resume probe after final analysis must remain a no-op; do not demote a
    # terminal audited decision back to analysis-pending.
    if str(progress.get("status", "")).startswith("FINAL_"):
        return
    lines = [
        "# PARTIAL STATUS", "", "`GRADIENT_ROUTING_RUNNING`", "",
        "Partial protocol coverage must not be used for a final Go/No-Go.", "",
        "| protocol | scenes | gradient rows | head-zero rows |", "|---|---:|---:|---:|",
    ]
    all_complete = True
    for protocol in PROTOCOLS:
        expected = int(units.loc[units.protocol == protocol, "scene_token"].nunique())
        directory = REPORT / "incremental/gradient_routing" / protocol
        metas = [json.loads(path.read_text()) for path in directory.glob("*.complete.json")]
        tokens = sorted(str(meta["scene_token"]) for meta in metas if meta.get("complete"))
        rows = sum(int(meta["gradient_rows"]) for meta in metas if meta.get("complete"))
        head_rows = sum(int(meta["head_rows"]) for meta in metas if meta.get("complete"))
        progress["protocols"][protocol].update({
            "completed_scenes": tokens, "gradient_rows": rows, "head_zero_rows": head_rows
        })
        lines.append(f"| {protocol} | {len(tokens)}/{expected} | {rows} | {head_rows} |")
        all_complete &= len(tokens) == expected
    progress["status"] = "ROUTING_COMPLETE_ANALYSIS_PENDING" if all_complete else "GRADIENT_ROUTING_RUNNING"
    atomic_json(REPORT / "progress_manifest.json", progress)
    lines += ["", "Resume:", "", "```bash"]
    lines += [f"python scripts/run_ctep_gradient_routing_audit.py --protocol {key}" for key in PROTOCOLS]
    lines += ["python scripts/analyze_ctep_gradient_routing_audit.py", "```", ""]
    temporary = REPORT / "PARTIAL_STATUS.md.tmp"
    temporary.write_text("\n".join(lines))
    os.replace(temporary, REPORT / "PARTIAL_STATUS.md")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    validation = json.loads((REPORT / "source_validation.json").read_text())
    if validation.get("status") != "VALIDATED_BEFORE_FORWARD":
        raise RuntimeError("source validation missing")
    if sha256(REPORT / "gradient_units.csv") != FROZEN_UNITS_SHA256:
        raise RuntimeError("frozen gradient-unit hash mismatch")
    units_frame = pd.read_csv(REPORT / "gradient_units.csv")
    units_frame = units_frame[units_frame.protocol == args.protocol]
    selected_scenes = sorted(units_frame.scene_token.unique())
    output_dir = REPORT / "incremental/gradient_routing" / args.protocol
    pending = []
    for scene in selected_scenes:
        meta_path = output_dir / f"{scene}.complete.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if (
                meta.get("complete") is True
                and meta.get("schema_version") == SCHEMA
                and meta.get("frozen_gradient_units_sha256") == FROZEN_UNITS_SHA256
            ):
                continue
        pending.append(scene)
    if args.max_scenes is not None:
        pending = pending[:args.max_scenes]
    if not pending:
        print(f"no pending routing scenes for {args.protocol}")
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
    token_to_index = {str(info["token"]): index for index, info in enumerate(clean_dataset.data_infos)}
    if token_to_index != {
        str(info["token"]): index for index, info in enumerate(fault_dataset.data_infos)
    }:
        raise RuntimeError("paired dataset index mismatch")
    scene_list = pd.read_csv(SOURCE / "scene_list.csv")
    scene_tokens = {
        row.scene_token: json.loads(row.sample_tokens_0_12)
        for row in scene_list.itertuples(index=False)
    }
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(CHECKPOINT), map_location="cpu")
    model = model.to(device).eval()
    head = model.pts_bbox_head
    classifier, groups = stream_petr_parameter_groups(head)
    if tuple(groups) != GRADIENT_LEVELS[1:]:
        raise RuntimeError(f"undeclared group order: {tuple(groups)}")
    write_parameter_manifest(classifier, groups)
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
        head_rows = []
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
                observed = references[unit["gt_token"]]["C"]
                if int(observed["qplus"]) != int(unit["C_qplus"]):
                    raise RuntimeError(f"C q+ replay mismatch {unit['unit_id']}")
            det_c = detection_loss(head, clean_dataset, index, output_c, device)
            rows, equivalence, stopped = gradient_rows_for_condition(
                "C", output_c, query_c, references, targets, frame_units,
                det_c, classifier, groups, head,
            )
            gradient_rows.extend(rows)
            head_rows.extend(stopped)
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
                observed = references[unit["gt_token"]]["D"]
                if int(observed["qplus"]) != int(unit["D_qplus"]):
                    raise RuntimeError(f"D q+ replay mismatch {unit['unit_id']}")
            det_d = detection_loss(head, clean_dataset, index, output_d, device)
            rows, equivalence, stopped = gradient_rows_for_condition(
                "D", output_d, query_d, references, targets, frame_units,
                det_d, classifier, groups, head,
            )
            gradient_rows.extend(rows)
            head_rows.extend(stopped)
            equivalence_rows.append({
                "protocol": args.protocol, "scene_token": scene, "sample_token": token,
                "frame_idx": frame_idx, **equivalence,
            })
            del output_d, query_d, det_d, output_b, output_a
            torch.cuda.empty_cache()

        expected_terms = sum(len(json.loads(value)) for value in scene_units_frame.active_terms)
        if len(head_rows) != expected_terms or len(gradient_rows) != expected_terms * len(GRADIENT_LEVELS):
            raise RuntimeError(
                f"scene row mismatch: expected {expected_terms}, got {len(head_rows)}/{len(gradient_rows)}"
            )
        atomic_csv(output_dir / f"{scene}.csv", gradient_rows)
        atomic_csv(output_dir / f"{scene}.head_zero.csv", head_rows)
        atomic_csv(output_dir / f"{scene}.equivalence.csv", equivalence_rows)
        atomic_json(output_dir / f"{scene}.complete.json", {
            "schema_version": SCHEMA,
            "frozen_gradient_units_sha256": FROZEN_UNITS_SHA256,
            "protocol": args.protocol,
            "scene_token": scene,
            "events": len(scene_units_frame),
            "active_terms": expected_terms,
            "gradient_rows": len(gradient_rows),
            "head_rows": len(head_rows),
            "equivalence_rows": len(equivalence_rows),
            "complete": True,
        })
        update_status()
        print(
            f"completed routing {args.protocol}/{scene} terms={expected_terms} "
            f"gradient_rows={len(gradient_rows)} head_zero_rows={len(head_rows)}",
            flush=True,
        )
        if STOP_REQUESTED:
            print("stop requested; current routing scene saved", flush=True)
            break


if __name__ == "__main__":
    main()
