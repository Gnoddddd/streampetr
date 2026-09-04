#!/usr/bin/env python3
"""Resume-safe canonical trajectory extraction for prospective linear probes."""

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
from mmdet3d.models import build_model  # noqa: E402
from nuscenes.nuscenes import NuScenes  # noqa: E402

from scripts.audit_dark_target_recoverability import (  # noqa: E402
    features, local_gt, physical, run_head, snapshot, unpack,
)
from scripts.run_bd_temporal_support_p0 import (  # noqa: E402
    CHECKPOINT, CONFIG, DATA, atomic_json, compare_outputs, compare_states,
    frame_context, protocol_dataset,
)
REPORT = ROOT / "reports/full_nuscenes/prospective_failure_decodability"
PROTOCOLS = {
    "blur_back": ROOT / "protocols/presets/motion_blur_back_10f_s09.json",
    "crash_back": ROOT / "protocols/presets/camera_crash_back_10f.json",
    "dark_back": ROOT / "protocols/presets/dark_back_10f_s09.json",
}
DISABLED = REPORT / "probes/disabled_empty_frame2.json"
CLASSES = ("car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
           "motorcycle", "bicycle", "pedestrian", "traffic_cone")
POST_RANGE = np.asarray([-61.2, -61.2, -10., 61.2, 61.2, 10.], dtype=float)
TAPS = ("temporal_alignment_query_state", "decoder_layer5_temporal_self_attn_output",
        "final_decoder_pre_cls_query")
OBSERVABLE_DIM = 21
SCHEMA = 2
STOP_REQUESTED = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("probe_train", "probe_val", "probe_test"))
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--scene-token")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def atomic_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def captured_head(model, meta, data, feats, prev_exists, state):
    """Passive real-graph tap capture, retaining tensors on-device until query selection."""
    head = model.pts_bbox_head
    if "temporal_alignment" in head.__dict__:
        raise RuntimeError("unexpected temporal_alignment instance override")
    captured = {tap: [] for tap in TAPS}
    original = head.temporal_alignment

    def temporal_wrapper(query_pos, tgt, reference_points):
        result = original(query_pos, tgt, reference_points)
        captured[TAPS[0]].append(torch.cat([result[0], result[1]], dim=-1))
        return result

    def attention_hook(_module, _arguments, output):
        captured[TAPS[1]].append(output)

    def classifier_pre_hook(_module, arguments):
        captured[TAPS[2]].append(arguments[0])

    head.temporal_alignment = temporal_wrapper
    attention_handle = head.transformer.decoder.layers[5].attentions[0].register_forward_hook(attention_hook)
    classifier_handle = head.cls_branches[-1].register_forward_pre_hook(classifier_pre_hook)
    try:
        output, next_state, _ = run_head(model, meta, data, feats, prev_exists, state)
    finally:
        attention_handle.remove(); classifier_handle.remove(); del head.temporal_alignment
    if len(captured[TAPS[0]]) != 1 or len(captured[TAPS[1]]) != 1 \
            or len(captured[TAPS[2]]) != len(head.cls_branches):
        raise RuntimeError("tap capture count changed")
    result = {
        TAPS[0]: captured[TAPS[0]][0][0].detach(),
        TAPS[1]: captured[TAPS[1]][0].transpose(0, 1).contiguous()[0].detach(),
        TAPS[2]: captured[TAPS[2]][-1][0].detach(),
    }
    if {key: tuple(value.shape) for key, value in result.items()} != {
            TAPS[0]: (900, 512), TAPS[1]: (900, 256), TAPS[2]: (900, 256)}:
        raise RuntimeError("tap tensor layout changed")
    return output, next_state, result


def deployed_match_map(logits, boxes, targets, context):
    scores = torch.as_tensor(logits).float().sigmoid().reshape(-1)
    values, indexes = torch.topk(scores, min(100, scores.numel()))
    predictions = []
    for rank, (score, flat) in enumerate(zip(values.tolist(), indexes.tolist()), start=1):
        if score < .1:
            continue
        query, label = divmod(int(flat), logits.shape[1])
        center = np.asarray(boxes[query, :3], dtype=float)
        if np.any(center < POST_RANGE[:3]) or np.any(center > POST_RANGE[3:]):
            continue
        ego = context["lidar2ego_rotation"] @ center + context["lidar2ego_translation"]
        if np.linalg.norm(ego[:2]) > context["class_range"][CLASSES[label]]:
            continue
        global_center = context["ego2global_rotation"] @ ego + context["ego2global_translation"]
        predictions.append({"query": query, "label": label, "score": float(score),
                            "flat_rank": rank, "box": np.asarray(boxes[query], dtype=float),
                            "global_center": global_center})
    pairs = []
    for gt_index, target in enumerate(targets):
        for prediction_index, prediction in enumerate(predictions):
            if int(target["label"]) != prediction["label"]:
                continue
            distance = np.linalg.norm(target["global_center"][:2] - prediction["global_center"][:2])
            if distance <= 2.:
                pairs.append((float(distance), gt_index, prediction_index))
    used_gt, used_prediction, result = set(), set(), {}
    for distance, gt_index, prediction_index in sorted(pairs):
        if gt_index in used_gt or prediction_index in used_prediction:
            continue
        used_gt.add(gt_index); used_prediction.add(prediction_index)
        prediction = dict(predictions[prediction_index]); prediction["match_distance_m"] = distance
        result[str(targets[gt_index]["token"])] = prediction
    return result


def target_frame(nusc, token):
    targets = local_gt(nusc, token)
    for target in targets:
        annotation = nusc.get("sample_annotation", target["token"])
        target["instance_token"] = str(annotation["instance_token"])
        target["global_center"] = np.asarray(annotation["translation"], dtype=float)
    return targets


def selected_features(logits, prediction, taps, pre_state, timestamp, num_query):
    query = int(prediction["query"])
    class_scores = 1. / (1. + np.exp(-np.clip(np.asarray(logits[query], float), -40., 40.)))
    ordered = np.sort(class_scores)
    top1, top2 = float(ordered[-1]), float(ordered[-2])
    onehot = np.zeros(len(CLASSES), dtype=np.float32)
    onehot[int(prediction["label"])] = 1.
    box = np.asarray(prediction["box"], dtype=float)
    propagated = query >= int(num_query)
    if propagated:
        memory_index = query - int(num_query)
        if pre_state["memory_timestamp"] is None:
            raise RuntimeError("propagated query has no source timestamp")
        source_age = abs(float(pre_state["memory_timestamp"][0, memory_index, 0].item())
                         + float(timestamp))
    else:
        source_age = 0.
    # Use the actual torch.topk deployment order.  Stable NumPy tie ordering is
    # not equivalent when multiple flattened scores are exactly equal.
    rank = int(prediction["flat_rank"])
    observable = np.concatenate([
        np.asarray([prediction["score"], top1, top1 - top2, rank], np.float32), onehot,
        np.asarray([abs(box[3]), abs(box[4]), abs(box[5]), np.linalg.norm(box[:2]),
                    np.linalg.norm(box[7:9]) if len(box) >= 9 else 0., float(propagated),
                    source_age], np.float32),
    ])
    if observable.shape != (OBSERVABLE_DIM,) or not np.isfinite(observable).all():
        raise RuntimeError("invalid observable feature vector")
    representation = {
        tap: (taps[tap][query].detach().float().cpu().numpy().astype(np.float32, copy=False)
              if torch.is_tensor(taps[tap]) else np.asarray(taps[tap][query], np.float32))
        for tap in TAPS
    }
    if {key: value.shape for key, value in representation.items()} != {
            TAPS[0]: (512,), TAPS[1]: (256,), TAPS[2]: (256,)}:
        raise RuntimeError("representation shape changed")
    return observable, representation


def frame_record(output, taps, pre_state, data, targets, context, pc_range, num_query):
    logits = output["all_cls_scores"][-1, 0].detach().float().cpu().numpy()
    boxes = physical(output, pc_range)[-1, 0].detach().float().cpu().numpy()
    matches = deployed_match_map(logits, boxes, targets, context)
    timestamp = float(data["timestamp"].reshape(-1)[0].item())
    candidates = {}
    token_map = {str(target["token"]): target for target in targets}
    for gt_token, prediction in matches.items():
        target = token_map[gt_token]
        observable, representation = selected_features(
            logits, prediction, taps, pre_state, timestamp, num_query)
        candidates[target["instance_token"]] = {
            "target": target, "prediction": prediction, "observable": observable,
            "representation": representation,
        }
    return candidates, matches


def update_status(validation):
    manifest = pd.read_csv(REPORT / "frozen_scene_manifest.csv")
    expected = manifest.groupby("split").size().to_dict()
    completed = {split_name: [] for split_name in expected}
    samples = {protocol: 0 for protocol in PROTOCOLS}
    positives = {protocol: 0 for protocol in PROTOCOLS}
    for marker in (REPORT / "incremental/P0").glob("*.complete.json"):
        value = json.loads(marker.read_text())
        if value.get("complete") and value.get("schema_version") == SCHEMA \
                and value.get("scene_manifest_sha256") == validation["scene_manifest_sha256"]:
            completed[value["split"]].append(value["scene_token"])
            for protocol in PROTOCOLS:
                samples[protocol] += int(value["samples_by_protocol"][protocol])
                positives[protocol] += int(value["positives_by_protocol"][protocol])
    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    coverage = {key: {"completed_scenes": sorted(value), "expected_scenes": expected[key]}
                for key, value in completed.items()}
    complete = all(len(completed[key]) == expected[key] for key in expected)
    progress["stages"]["P0_extraction"] = {"coverage": coverage,
        "samples_by_protocol": samples, "positives_by_protocol": positives}
    progress["status"] = "P0_EXTRACTION_COMPLETE_PROBE_PENDING" if complete else "P0_EXTRACTION_RUNNING"
    atomic_json(REPORT / "progress_manifest.json", progress)
    lines = ["# PARTIAL STATUS", "", f"`{progress['status']}`", "",
             "Coverage is incomplete; no Go/No-Go is permitted.", "",
             "| split | completed scenes | expected scenes |", "|---|---:|---:|"]
    for key in ("probe_train", "probe_val", "probe_test"):
        lines.append(f"| {key} | {len(completed[key])} | {expected[key]} |")
    lines += ["", f"Samples: `{samples}`", "", f"Positive labels: `{positives}`", "",
              "Resume:", "", "```bash", "python scripts/run_prospective_failure_features.py",
              "python scripts/analyze_prospective_failure_decodability.py", "```", ""]
    temporary = REPORT / "PARTIAL_STATUS.md.tmp"; temporary.write_text("\n".join(lines))
    os.replace(temporary, REPORT / "PARTIAL_STATUS.md")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    validation = json.loads((REPORT / "source_validation.json").read_text())
    if validation.get("status") != "VALIDATED_BEFORE_FORWARD":
        raise RuntimeError("pre-forward validation missing")
    manifest = pd.read_csv(REPORT / "frozen_scene_manifest.csv")
    pending = []
    for row in manifest.itertuples(index=False):
        marker = REPORT / "incremental/P0" / f"{row.scene_token}.complete.json"
        if marker.exists():
            value = json.loads(marker.read_text())
            if value.get("complete") and value.get("schema_version") == SCHEMA \
                    and value.get("scene_manifest_sha256") == validation["scene_manifest_sha256"]:
                continue
        if args.split and row.split != args.split:
            continue
        if args.scene_token and row.scene_token != args.scene_token:
            continue
        pending.append(row)
    if args.max_scenes is not None:
        pending = pending[:args.max_scenes]
    if not pending:
        print("no pending scenes"); update_status(validation); return

    def request_stop(_signum, _frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    torch.manual_seed(2026); np.random.seed(2026)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    cfg = Config.fromfile(str(CONFIG)); import_modules_from_strings(**cfg.custom_imports)
    cfg.model.pretrained = None
    clean_dataset = protocol_dataset(cfg, None)
    disabled_dataset = protocol_dataset(cfg, DISABLED)
    fault_datasets = {key: protocol_dataset(cfg, path) for key, path in PROTOCOLS.items()}
    token_index = {str(info["token"]): index for index, info in enumerate(clean_dataset.data_infos)}
    for name, dataset in {"disabled": disabled_dataset, **fault_datasets}.items():
        if token_index != {str(info["token"]): index for index, info in enumerate(dataset.data_infos)}:
            raise RuntimeError(f"paired dataset mismatch: {name}")
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(CHECKPOINT), map_location="cpu")
    model = model.to(device).eval(); head = model.pts_bbox_head; head.reset_memory()
    initial = snapshot(head); pc_range = head.pc_range.detach(); num_query = int(head.num_query)
    if num_query != 644 or int(head.num_propagated) != 256:
        raise RuntimeError("query layout changed")
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(DATA), verbose=False)
    output_dir = REPORT / "incremental/P0"; output_dir.mkdir(parents=True, exist_ok=True)
    sample_fields = ["sample_id", "protocol", "split", "scene_token", "instance_token",
                     "input_frame_idx", "target_frame_idx", "input_sample_token", "target_sample_token",
                     "input_gt_token", "target_gt_token", "y_tp_to_fn", "query_index",
                     "prediction_class", "prediction_score", "query_top1_score",
                     "query_top1_top2_margin", "deployment_flat_rank", "box_width_m", "box_length_m",
                     "box_height_m", "box_radial_distance_m", "box_speed_mps",
                     "query_is_propagated", "query_source_age_seconds", "gt_used_as_input"]
    status_fields = ["protocol", "split", "scene_token", "sample_token", "frame_idx",
                     "instance_token", "gt_token", "gt_class", "tp"]
    equivalence_fields = ["scene_token", "sample_token", "frame_idx", "check",
                          "output_bitwise_equal", "output_max_abs_diff", "memory_bitwise_equal",
                          "memory_max_abs_diff"]

    for scene_row in pending:
        scene = str(scene_row.scene_token); split = str(scene_row.split)
        tokens = json.loads(scene_row.sample_tokens_0_12)
        shared_state = initial; frame2 = None
        for frame_idx in range(3):
            index = token_index[tokens[frame_idx]]; meta, image, data = unpack(clean_dataset[index], device)
            with torch.no_grad():
                _, _, feats = features(model, image); feats = feats.detach(); pre = shared_state
                if frame_idx < 2:
                    output, shared_state, _ = run_head(model, meta, data, feats, frame_idx > 0, pre)
                    del output
                else:
                    output, shared_state, taps = captured_head(model, meta, data, feats, True, pre)
                    targets = target_frame(nusc, tokens[frame_idx])
                    context = frame_context(clean_dataset.data_infos[index], clean_dataset)
                    frame2 = (index, meta, image, data, feats, pre, output, taps, targets, context)
        if frame2 is None:
            raise RuntimeError("shared frame-2 anchor missing")
        index, meta, image, data, feats, pre, output, taps, targets, context = frame2
        with torch.no_grad():
            plain_output, plain_state, _ = run_head(model, meta, data, feats, True, pre)
            disabled_meta, disabled_image, disabled_data = unpack(disabled_dataset[index], device)
            _, _, disabled_feats = features(model, disabled_image)
            disabled_output, disabled_state, _ = run_head(
                model, disabled_meta, disabled_data, disabled_feats.detach(), True, pre)
        hook_out, hook_diff = compare_outputs(output, plain_output)
        hook_state, hook_state_diff = compare_states(shared_state, plain_state)
        disabled_out, disabled_diff = compare_outputs(output, disabled_output)
        disabled_state_equal, disabled_state_diff = compare_states(shared_state, disabled_state)
        if not all((hook_out, hook_state, disabled_out, disabled_state_equal)):
            raise RuntimeError(f"passive hook/disabled exactness failed: {scene}")
        equivalence = [
            {"scene_token": scene, "sample_token": tokens[2], "frame_idx": 2,
             "check": "passive_hooks_vs_unhooked_B0", "output_bitwise_equal": hook_out,
             "output_max_abs_diff": hook_diff, "memory_bitwise_equal": hook_state,
             "memory_max_abs_diff": hook_state_diff},
            {"scene_token": scene, "sample_token": tokens[2], "frame_idx": 2,
             "check": "disabled_empty_schedule_vs_B0", "output_bitwise_equal": disabled_out,
             "output_max_abs_diff": disabled_diff, "memory_bitwise_equal": disabled_state_equal,
             "memory_max_abs_diff": disabled_state_diff},
        ]
        clean_candidates, clean_matches = frame_record(
            output, taps, pre, data, targets, context, pc_range, num_query)
        protocol_state = {protocol: copy.deepcopy(shared_state) for protocol in PROTOCOLS}
        previous = {protocol: clean_candidates for protocol in PROTOCOLS}
        previous_token = {protocol: tokens[2] for protocol in PROTOCOLS}
        sample_rows, status_rows = [], []
        observable_arrays, tap_arrays = [], {tap: [] for tap in TAPS}
        for protocol in PROTOCOLS:
            for target in targets:
                status_rows.append({"protocol": protocol, "split": split, "scene_token": scene,
                    "sample_token": tokens[2], "frame_idx": 2, "instance_token": target["instance_token"],
                    "gt_token": target["token"], "gt_class": target["name"],
                    "tp": target["token"] in clean_matches})
        for frame_idx in range(3, 13):
            index = token_index[tokens[frame_idx]]
            targets = target_frame(nusc, tokens[frame_idx])
            context = frame_context(clean_dataset.data_infos[index], clean_dataset)
            for protocol, dataset in fault_datasets.items():
                fault_meta, fault_image, fault_data = unpack(dataset[index], device)
                with torch.no_grad():
                    _, _, fault_feats = features(model, fault_image); fault_feats = fault_feats.detach()
                    pre_state = protocol_state[protocol]
                    fault_output, protocol_state[protocol], fault_taps = captured_head(
                        model, fault_meta, fault_data, fault_feats, True, pre_state)
                current_candidates, current_matches = frame_record(
                    fault_output, fault_taps, pre_state, fault_data, targets, context, pc_range, num_query)
                current_by_instance = {target["instance_token"]: target for target in targets}
                for target in targets:
                    status_rows.append({"protocol": protocol, "split": split, "scene_token": scene,
                        "sample_token": tokens[frame_idx], "frame_idx": frame_idx,
                        "instance_token": target["instance_token"], "gt_token": target["token"],
                        "gt_class": target["name"], "tp": target["token"] in current_matches})
                for instance_token, anchor in previous[protocol].items():
                    if instance_token not in current_by_instance:
                        continue
                    target = current_by_instance[instance_token]
                    y = int(target["token"] not in current_matches)
                    prediction = anchor["prediction"]; obs = anchor["observable"]
                    sample_id = f"{protocol}:{scene}:{instance_token}:{frame_idx-1}:{frame_idx}"
                    sample_rows.append({
                        "sample_id": sample_id, "protocol": protocol, "split": split,
                        "scene_token": scene, "instance_token": instance_token,
                        "input_frame_idx": frame_idx - 1, "target_frame_idx": frame_idx,
                        "input_sample_token": previous_token[protocol], "target_sample_token": tokens[frame_idx],
                        "input_gt_token": anchor["target"]["token"], "target_gt_token": target["token"],
                        "y_tp_to_fn": y, "query_index": prediction["query"],
                        "prediction_class": CLASSES[int(prediction["label"])],
                        "prediction_score": obs[0], "query_top1_score": obs[1],
                        "query_top1_top2_margin": obs[2], "deployment_flat_rank": int(obs[3]),
                        "box_width_m": obs[14], "box_length_m": obs[15], "box_height_m": obs[16],
                        "box_radial_distance_m": obs[17], "box_speed_mps": obs[18],
                        "query_is_propagated": int(obs[19]), "query_source_age_seconds": obs[20],
                        "gt_used_as_input": False,
                    })
                    observable_arrays.append(obs)
                    for tap in TAPS:
                        tap_arrays[tap].append(anchor["representation"][tap])
                previous[protocol] = current_candidates; previous_token[protocol] = tokens[frame_idx]
                del fault_output, fault_feats, fault_taps, fault_image, fault_data
        tap_dimensions = {TAPS[0]: 512, TAPS[1]: 256, TAPS[2]: 256}
        arrays = {
            "observable": (np.stack(observable_arrays).astype(np.float32)
                           if sample_rows else np.empty((0, OBSERVABLE_DIM), np.float32)),
            "label": np.asarray([row["y_tp_to_fn"] for row in sample_rows], np.int8),
        }
        arrays.update({
            tap: (np.stack(tap_arrays[tap]).astype(np.float32)
                  if sample_rows else np.empty((0, tap_dimensions[tap]), np.float32))
            for tap in TAPS
        })
        if arrays["observable"].shape != (len(sample_rows), OBSERVABLE_DIM):
            raise RuntimeError("observable/sample alignment mismatch")
        prefix = output_dir / scene
        atomic_csv(prefix.with_suffix(".samples.csv"), sample_rows, sample_fields)
        atomic_csv(prefix.with_suffix(".status.csv"), status_rows, status_fields)
        atomic_csv(prefix.with_suffix(".equivalence.csv"), equivalence, equivalence_fields)
        atomic_npz(prefix.with_suffix(".features.npz"), **arrays)
        sample_frame = pd.DataFrame(sample_rows, columns=sample_fields)
        sample_counts = {protocol: int(sum(row["protocol"] == protocol for row in sample_rows))
                         for protocol in PROTOCOLS}
        positive_counts = {protocol: int(sum(
            row["protocol"] == protocol and int(row["y_tp_to_fn"]) == 1 for row in sample_rows))
                           for protocol in PROTOCOLS}
        atomic_json(prefix.with_suffix(".complete.json"), {
            "schema_version": SCHEMA, "scene_manifest_sha256": validation["scene_manifest_sha256"],
            "feature_manifest_sha256": validation["feature_manifest_sha256"],
            "scene_token": scene, "split": split, "rows": len(sample_rows),
            "samples_by_protocol": sample_counts,
            "positives_by_protocol": positive_counts,
            "equivalence_rows": len(equivalence), "complete": True,
        })
        update_status(validation)
        print(f"completed {split}/{scene}: samples={len(sample_rows)} positives={int(arrays['label'].sum())}", flush=True)
        del output, plain_output, disabled_output, taps, feats, image, data, arrays
        if STOP_REQUESTED:
            print("stop requested; current scene checkpoint saved", flush=True); break


if __name__ == "__main__":
    main()
