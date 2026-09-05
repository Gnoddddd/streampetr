#!/usr/bin/env python3
"""Incremental P1 GT-local causal patching for P0-passing real graph taps."""

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
sys.dont_write_bytecode = True
sys.path.insert(0, str(STREAM))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmcv.utils import import_modules_from_strings  # noqa: E402
from mmdet.models.utils.transformer import inverse_sigmoid  # noqa: E402
from mmdet3d.datasets import build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from nuscenes.nuscenes import NuScenes  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402

from analysis.temporal_representation_localization import (  # noqa: E402
    geometry_match,
    local_non_gt_candidates,
)
from scripts.audit_dark_target_recoverability import (  # noqa: E402
    features,
    local_gt,
    physical,
    run_head,
    snapshot,
    unpack,
)
from scripts.audit_temporal_state_counterfactual import output_metrics  # noqa: E402


SOURCE = ROOT / "reports/full_nuscenes/ctep_method_activation"
REPORT = ROOT / "reports/full_nuscenes/temporal_representation_localization_audit"
CONFIG = ROOT / "configs/full_nuscenes/stream_petr_r50_90e_ctep_train_audit.py"
CHECKPOINT = ROOT / "checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth"
DATA = ROOT / "data/nuscenes"
PROTOCOLS = {
    "blur_back": ROOT / "protocols/presets/motion_blur_back_10f_s09.json",
    "crash_back": ROOT / "protocols/presets/camera_crash_back_10f.json",
    "dark_back": ROOT / "protocols/presets/dark_back_10f_s09.json",
}
PAIR_CONDITIONS = {"AC": ("A", "C"), "BD": ("B", "D")}
SCHEMA = 1
STOP_REQUESTED = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=tuple(PROTOCOLS), required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--scene-token")
    parser.add_argument("--batch-scenarios", type=int, default=4)
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
        raise RuntimeError(f"empty P1 scene checkpoint: {path}")
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


def capture_branch(model, meta, data, feats, prev_exists, state):
    """Capture taps and exact replay inputs without modifying the branch."""

    head = model.pts_bbox_head
    layer = head.transformer.decoder.layers[5]
    if "temporal_alignment" in head.__dict__ or "forward" in layer.__dict__:
        raise RuntimeError("unexpected instance method override")
    captured = {"cls_inputs": []}
    original_temporal = head.temporal_alignment
    original_layer_forward = layer.forward

    def temporal_wrapper(query_pos, tgt, reference_points):
        result = original_temporal(query_pos, tgt, reference_points)
        captured["temporal_result"] = result
        captured["temporal_alignment_query_state"] = torch.cat([result[0], result[1]], dim=-1)
        return result

    def layer_wrapper(*arguments, **kwargs):
        captured["layer5_arguments"] = arguments
        captured["layer5_kwargs"] = kwargs
        return original_layer_forward(*arguments, **kwargs)

    def transformer_pre_hook(_module, arguments):
        captured["transformer_arguments"] = arguments

    def temporal_attention_hook(_module, _arguments, output):
        captured["decoder_layer5_temporal_self_attn_output"] = output

    def cls_pre_hook(_module, arguments):
        captured["cls_inputs"].append(arguments[0])

    head.temporal_alignment = temporal_wrapper
    layer.forward = layer_wrapper
    transformer_handle = head.transformer.register_forward_pre_hook(transformer_pre_hook)
    attention_handle = layer.attentions[0].register_forward_hook(temporal_attention_hook)
    cls_handle = head.cls_branches[-1].register_forward_pre_hook(cls_pre_hook)
    try:
        output, next_state, _ = run_head(model, meta, data, feats, prev_exists, state)
    finally:
        transformer_handle.remove()
        attention_handle.remove()
        cls_handle.remove()
        del head.temporal_alignment
        del layer.forward
    if len(captured["cls_inputs"]) != len(head.cls_branches):
        raise RuntimeError("classification capture count mismatch")
    temporal_result = captured["temporal_result"]
    transformer_arguments = captured["transformer_arguments"]
    if len(transformer_arguments) != 7:
        raise RuntimeError(f"unexpected transformer argument count: {len(transformer_arguments)}")
    attention = captured["decoder_layer5_temporal_self_attn_output"]
    result = {
        "representations": {
            "temporal_alignment_query_state": captured["temporal_alignment_query_state"].detach(),
            "decoder_layer5_temporal_self_attn_output": attention.transpose(0, 1).contiguous().detach(),
            "final_decoder_pre_cls_query": captured["cls_inputs"][-1].detach(),
        },
        "transformer_arguments": tuple(
            None if value is None else value.detach() for value in transformer_arguments
        ),
        "reference_points": temporal_result[2].detach(),
        "layer5_arguments": tuple(
            value.detach() if torch.is_tensor(value) else value
            for value in captured["layer5_arguments"]
        ),
        "layer5_kwargs": {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in captured["layer5_kwargs"].items()
        },
    }
    return output, next_state, result


def repeat_batch_first(value, count: int):
    if value is None:
        return None
    repeats = [1] * value.ndim
    repeats[0] = count
    return value.repeat(*repeats)


def repeat_sequence_first(value, count: int):
    if value is None:
        return None
    repeats = [1] * value.ndim
    repeats[1] = count
    return value.repeat(*repeats)


def apply_patches_batch_first(base, source, scenarios):
    result = repeat_batch_first(base, len(scenarios)).clone()
    for batch_index, scenario in enumerate(scenarios):
        for source_index, target_index, _ in scenario["patch_pairs"]:
            result[batch_index, int(target_index)] = source[0, int(source_index)]
    return result


def final_output_from_query(head, query, reference_points):
    logits = head.cls_branches[5](query)
    coordinates = head.reg_branches[5](query)
    reference = inverse_sigmoid(reference_points.clone())
    coordinates[..., 0:3] += reference[..., 0:3]
    coordinates[..., 0:3] = coordinates[..., 0:3].sigmoid()
    coordinates[..., 0:3] = (
        coordinates[..., 0:3] * (head.pc_range[3:6] - head.pc_range[0:3])
        + head.pc_range[0:3]
    )
    return {
        "all_cls_scores": logits.unsqueeze(0),
        "all_bbox_preds": coordinates.unsqueeze(0),
        "dn_mask_dict": None,
    }


def replay_scenarios(head, tap: str, source_capture, target_capture, scenarios):
    count = len(scenarios)
    source_rep = source_capture["representations"][tap]
    reference_points = repeat_batch_first(target_capture["reference_points"], count)
    if tap == "final_decoder_pre_cls_query":
        query = apply_patches_batch_first(
            target_capture["representations"][tap], source_rep, scenarios
        )
        return final_output_from_query(head, query, reference_points)
    if tap == "temporal_alignment_query_state":
        memory, tgt, query_pos, pos_embed, attn_mask, temp_memory, temp_pos = (
            target_capture["transformer_arguments"]
        )
        combined = apply_patches_batch_first(
            target_capture["representations"][tap], source_rep, scenarios
        )
        tgt_batch = combined[..., :256]
        query_pos_batch = combined[..., 256:]
        outs_dec, _ = head.transformer(
            repeat_batch_first(memory, count),
            tgt_batch,
            query_pos_batch,
            repeat_batch_first(pos_embed, count),
            attn_mask,
            repeat_batch_first(temp_memory, count),
            repeat_batch_first(temp_pos, count),
        )
        query = torch.nan_to_num(outs_dec)[-1]
        return final_output_from_query(head, query, reference_points)
    if tap == "decoder_layer5_temporal_self_attn_output":
        layer = head.transformer.decoder.layers[5]
        arguments = list(target_capture["layer5_arguments"])
        kwargs = dict(target_capture["layer5_kwargs"])
        if arguments:
            arguments[0] = repeat_sequence_first(arguments[0], count)
            for index in range(1, len(arguments)):
                if torch.is_tensor(arguments[index]) and arguments[index].ndim >= 3:
                    arguments[index] = repeat_sequence_first(arguments[index], count)
        for key, value in list(kwargs.items()):
            if torch.is_tensor(value) and value.ndim >= 3:
                kwargs[key] = repeat_sequence_first(value, count)

        def patch_hook(_module, _arguments, output):
            result = output.clone()
            for batch_index, scenario in enumerate(scenarios):
                for source_index, target_index, _ in scenario["patch_pairs"]:
                    result[int(target_index), batch_index] = source_rep[0, int(source_index)]
            return result

        handle = layer.attentions[0].register_forward_hook(patch_hook)
        try:
            query = layer(*arguments, **kwargs)
        finally:
            handle.remove()
        # PETRTransformerDecoder appends post_norm to every returned
        # intermediate, including layer 5 (petr_transformer.py:410-418).
        if head.transformer.decoder.post_norm is not None:
            query = head.transformer.decoder.post_norm(query)
        query = query.transpose(0, 1).contiguous()
        return final_output_from_query(head, query, reference_points)
    raise KeyError(tap)


def slice_output(output, index: int):
    return {
        "all_cls_scores": output["all_cls_scores"][:, index:index + 1],
        "all_bbox_preds": output["all_bbox_preds"][:, index:index + 1],
        "dn_mask_dict": None,
    }


def output_exact(left, right) -> tuple[bool, float]:
    exact = True
    differences = []
    for key in ("all_cls_scores", "all_bbox_preds"):
        a = left[key][-1:]
        b = right[key]
        exact &= torch.equal(a, b)
        differences.append(float((a.detach().float() - b.detach().float()).abs().max().item()))
    return bool(exact), max(differences, default=0.0)


def non_gt_pairs(source_boxes, target_boxes, all_gt_centers, target_center, count):
    source_indices = local_non_gt_candidates(
        source_boxes, all_gt_centers, target_center, count, 2.0
    )
    target_indices = local_non_gt_candidates(
        target_boxes, all_gt_centers, target_center, count, 2.0
    )
    if len(source_indices) != count or len(target_indices) != count:
        return []
    return geometry_match(source_boxes, target_boxes, source_indices, target_indices)


def scenario_row(scenario, patched, target, targets_list, pc_range, frame_context):
    metrics = output_metrics(patched, target, targets_list, pc_range, frame_context)
    base_s_pos = float(scenario["base_target_s_pos"])
    source_s_pos = float(scenario["source_s_pos"])
    delta = float(metrics["s_pos"] - base_s_pos) if metrics["candidate"] else math.nan
    original_gap = source_s_pos - base_s_pos
    closure_fraction = delta / original_gap if math.isfinite(delta) and abs(original_gap) > 1e-12 else math.nan
    return {
        **scenario["row"],
        "patched_candidate": metrics["candidate"],
        "patched_s_pos": metrics["s_pos"],
        "delta_s_pos": delta,
        "original_history_gap": original_gap,
        "history_gap_closure": delta,
        "history_gap_closure_fraction": closure_fraction,
        "patched_rank": metrics["rank"],
        "patched_margin": metrics["margin"],
        "patched_topk": metrics["topk"],
        "patched_tp": metrics["tp"],
        "topk_recovery": (not scenario["base_target_topk"]) and bool(metrics["topk"]),
        "tp_recovery": (not scenario["base_target_tp"]) and bool(metrics["tp"]),
        "topk_damage": bool(scenario["base_target_topk"]) and not bool(metrics["topk"]),
        "tp_damage": bool(scenario["base_target_tp"]) and not bool(metrics["tp"]),
    }


def update_status() -> None:
    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    if str(progress.get("status", "")).startswith("FINAL_"):
        return
    population = pd.read_csv(REPORT / "population.csv", usecols=["protocol", "scene_token"])
    lines = [
        "# PARTIAL STATUS", "", "`P1_RUNNING`", "",
        "P1 partial coverage cannot produce a causal tap or final decision.", "",
        "| protocol | scenes | intervention rows |", "|---|---:|---:|",
    ]
    all_complete = True
    stage = {"status": "RUNNING", "taps": json.loads((REPORT / "p0_decision.json").read_text())["p0_passing_taps"]}
    for protocol in PROTOCOLS:
        expected = int(population[population.protocol == protocol].scene_token.nunique())
        directory = REPORT / "incremental/P1" / protocol
        metas = [json.loads(path.read_text()) for path in directory.glob("*.complete.json")]
        tokens = sorted(meta["scene_token"] for meta in metas if meta.get("complete"))
        rows = sum(int(meta["patch_rows"]) for meta in metas if meta.get("complete"))
        stage[protocol] = {"completed_scenes": tokens, "expected_scenes": expected, "rows": rows}
        lines.append(f"| {protocol} | {len(tokens)}/{expected} | {rows} |")
        all_complete &= len(tokens) == expected
    progress["stages"]["P1"] = stage
    progress["status"] = "P1_COMPLETE_ANALYSIS_PENDING" if all_complete else "P1_RUNNING"
    atomic_json(REPORT / "progress_manifest.json", progress)
    lines += ["", "Resume:", "", "```bash"]
    lines += [f"python scripts/run_temporal_representation_p1.py --protocol {p}" for p in PROTOCOLS]
    lines += ["python scripts/analyze_temporal_representation_p1.py", "```", ""]
    temporary = REPORT / "PARTIAL_STATUS.md.tmp"
    temporary.write_text("\n".join(lines))
    os.replace(temporary, REPORT / "PARTIAL_STATUS.md")


def main() -> None:
    args = parse_args()
    if args.batch_scenarios < 1:
        raise ValueError("batch-scenarios must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    decision = json.loads((REPORT / "p0_decision.json").read_text())
    if decision.get("verdict") != "P0_GO_P1_REQUIRED":
        raise RuntimeError(f"P1 locked by {decision.get('verdict')}")
    taps = tuple(decision["p0_passing_taps"])
    population = pd.read_csv(REPORT / "population.csv")
    population = population[population.protocol == args.protocol]
    p0 = pd.read_csv(REPORT / "per_gt_drift.csv")
    p0 = p0[(p0.protocol == args.protocol) & p0.tap_id.isin(taps)]
    scene_order = pd.read_csv(SOURCE / "scene_list.csv").scene_token.tolist()
    output_dir = REPORT / "incremental/P1" / args.protocol
    pending = []
    for scene in scene_order:
        meta_path = output_dir / f"{scene}.complete.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("complete") and meta.get("schema_version") == SCHEMA and meta.get("p0_taps") == list(taps):
                continue
        pending.append(scene)
    if args.scene_token is not None:
        if args.scene_token not in scene_order:
            raise ValueError(f"scene token is not in the frozen scene list: {args.scene_token}")
        pending = [scene for scene in pending if scene == args.scene_token]
    if args.max_scenes is not None:
        pending = pending[:args.max_scenes]
    if not pending:
        print(f"no pending P1 scenes for {args.protocol}")
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
    head.reset_memory()
    initial = snapshot(head)
    pc_range = head.pc_range.detach()
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(DATA), verbose=False)

    for scene in pending:
        scene_population = population[population.scene_token == scene]
        population_by_sample = {
            token: group.set_index("unit_id", drop=False).to_dict("index")
            for token, group in scene_population.groupby("sample_token")
        }
        scene_p0 = p0[p0.scene_token == scene]
        p0_by_sample = {token: group for token, group in scene_p0.groupby("sample_token")}
        clean_state, fault_state = initial, initial
        rows = []
        equivalence_rows = []
        replay_checked = False
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
                output_a, clean_state, capture_a = capture_branch(
                    model, clean_meta, clean_data, clean_feats, frame_idx > 0, clean_pre
                )
            frame_p0 = p0_by_sample.get(token)
            if frame_p0 is None:
                with torch.no_grad():
                    _, fault_state, _ = run_head(
                        model, fault_meta, fault_data, fault_feats, frame_idx > 0, fault_pre
                    )
                continue
            with torch.no_grad():
                output_b, _, capture_b = capture_branch(
                    model, fault_meta, fault_data, fault_feats, frame_idx > 0, clean_pre
                )
                output_c, _, capture_c = capture_branch(
                    model, clean_meta, clean_data, clean_feats, frame_idx > 0, fault_pre
                )
                output_d, fault_state, capture_d = capture_branch(
                    model, fault_meta, fault_data, fault_feats, frame_idx > 0, fault_pre
                )
            outputs = {"A": output_a, "B": output_b, "C": output_c, "D": output_d}
            captures = {"A": capture_a, "B": capture_b, "C": capture_c, "D": capture_d}
            boxes = {
                condition: physical(output, pc_range)[-1, 0].detach().float().cpu().numpy()
                for condition, output in outputs.items()
            }
            targets_list = local_gt(nusc, token)
            for target in targets_list:
                annotation = nusc.get("sample_annotation", target["token"])
                target["global_center"] = np.asarray(annotation["translation"], float)
            targets = {target["token"]: target for target in targets_list}
            all_gt_centers = np.asarray([target["center"] for target in targets_list], float)
            frame_population = population_by_sample[token]
            frame_context = context(info, clean_dataset)

            if not replay_checked:
                with torch.no_grad():
                    for tap in taps:
                        replay = replay_scenarios(
                            head, tap, capture_c, capture_c,
                            [{"patch_pairs": []}],
                        )
                        exact, difference = output_exact(output_c, replay)
                        if not exact:
                            raise RuntimeError(f"no-patch downstream replay changed B0 at {tap}: {difference}")
                        equivalence_rows.append({
                            "protocol": args.protocol,
                            "scene_token": scene,
                            "sample_token": token,
                            "frame_idx": frame_idx,
                            "tap_id": tap,
                            "no_patch_output_bitwise_equal": exact,
                            "no_patch_output_max_abs_diff": difference,
                        })
                replay_checked = True

            for pair, (source_condition, target_condition) in PAIR_CONDITIONS.items():
                source_capture = captures[source_condition]
                target_capture = captures[target_condition]
                for tap in taps:
                    selected = frame_p0[
                        (frame_p0.pair == pair)
                        & (frame_p0.tap_id == tap)
                        & (frame_p0.matched_pair_count > 0)
                    ]
                    scenarios = []
                    for drift in selected.itertuples(index=False):
                        frozen = frame_population[drift.unit_id]
                        target = targets[drift.gt_token]
                        source_queries = json.loads(drift.matched_source_queries)
                        target_queries = json.loads(drift.matched_target_queries)
                        target_pairs = [
                            (int(source_query), int(target_query), 0.0)
                            for source_query, target_query in zip(source_queries, target_queries)
                        ]
                        base = {
                            "unit_id": drift.unit_id,
                            "protocol": args.protocol,
                            "scene_token": scene,
                            "sample_token": token,
                            "frame_idx": frame_idx,
                            "gt_token": drift.gt_token,
                            "instance_token": drift.instance_token,
                            "gt_class": drift.gt_class,
                            "distance_m": drift.distance_m,
                            "visibility_token": drift.visibility_token,
                            "population": drift.population,
                            "tap_id": tap,
                            "pair": pair,
                            "source_condition": source_condition,
                            "target_condition": target_condition,
                            "base_source_s_pos": frozen[f"{source_condition}_s_pos"],
                            "base_target_s_pos": frozen[f"{target_condition}_s_pos"],
                            "base_target_rank": frozen[f"{target_condition}_rank"],
                            "base_target_margin": frozen[f"{target_condition}_margin"],
                            "base_target_topk": frozen[f"{target_condition}_topk"],
                            "base_target_tp": frozen[f"{target_condition}_tp"],
                            "patch_count": len(target_pairs),
                        }
                        scenarios.append({
                            "patch_pairs": target_pairs,
                            "target": target,
                            "source_s_pos": frozen[f"{source_condition}_s_pos"],
                            "base_target_s_pos": frozen[f"{target_condition}_s_pos"],
                            "base_target_topk": bool(frozen[f"{target_condition}_topk"]),
                            "base_target_tp": bool(frozen[f"{target_condition}_tp"]),
                            "row": {**base, "control_type": "gt_target_patch"},
                        })
                        if drift.population == "lost":
                            control_pairs = non_gt_pairs(
                                boxes[source_condition], boxes[target_condition], all_gt_centers,
                                target["center"], len(target_pairs),
                            )
                            if len(control_pairs) != len(target_pairs):
                                raise RuntimeError(f"same-count non-GT control unavailable: {drift.unit_id}")
                            scenarios.append({
                                "patch_pairs": control_pairs,
                                "target": target,
                                "source_s_pos": frozen[f"{source_condition}_s_pos"],
                                "base_target_s_pos": frozen[f"{target_condition}_s_pos"],
                                "base_target_topk": bool(frozen[f"{target_condition}_topk"]),
                                "base_target_tp": bool(frozen[f"{target_condition}_tp"]),
                                "row": {
                                    **base,
                                    "control_type": "non_gt_patch",
                                    "patch_count": len(control_pairs),
                                },
                            })
                    for start in range(0, len(scenarios), args.batch_scenarios):
                        chunk = scenarios[start:start + args.batch_scenarios]
                        with torch.no_grad():
                            patched_batch = replay_scenarios(
                                head, tap, source_capture, target_capture, chunk
                            )
                        for local_index, scenario in enumerate(chunk):
                            rows.append(scenario_row(
                                scenario,
                                slice_output(patched_batch, local_index),
                                scenario["target"],
                                targets_list,
                                pc_range,
                                frame_context,
                            ))
                        del patched_batch
            del output_a, output_b, output_c, output_d, capture_a, capture_b, capture_c, capture_d
            torch.cuda.empty_cache()

        if not rows:
            raise RuntimeError(f"no eligible P1 rows in {args.protocol}/{scene}")
        atomic_csv(output_dir / f"{scene}.csv", rows)
        atomic_csv(output_dir / f"{scene}.equivalence.csv", equivalence_rows)
        atomic_json(output_dir / f"{scene}.complete.json", {
            "schema_version": SCHEMA,
            "protocol": args.protocol,
            "scene_token": scene,
            "p0_taps": list(taps),
            "patch_rows": len(rows),
            "gt_target_rows": sum(row["control_type"] == "gt_target_patch" for row in rows),
            "non_gt_rows": sum(row["control_type"] == "non_gt_patch" for row in rows),
            "replay_equivalence_rows": len(equivalence_rows),
            "complete": True,
        })
        update_status()
        print(f"completed P1 {args.protocol}/{scene} patch_rows={len(rows)}", flush=True)
        if STOP_REQUESTED:
            print("stop requested; current P1 scene saved", flush=True)
            break


if __name__ == "__main__":
    main()
