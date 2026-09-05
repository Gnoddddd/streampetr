#!/usr/bin/env python3
"""Export one-step CARE-3D counterfactual vulnerability supervision.

Every Clean/Fault t+1 branch starts from an identical clean post-state H_t.
Fault branches are discarded; only the clean branch advances the trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
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

from analysis.care3d_counterfactual import (  # noqa: E402
    PROTOCOLS, assert_prospective_payload, assert_unique_sample_ids,
    clone_counterfactual_states, freeze_module, same_query_counterfactual,
    states_exact,
)
from scripts.audit_dark_target_recoverability import (  # noqa: E402
    features, physical, run_head, snapshot, unpack,
)
from scripts.run_bd_temporal_support_p0 import (  # noqa: E402
    CHECKPOINT, CONFIG, DATA, frame_context, protocol_dataset,
)
from scripts.run_prospective_failure_features import (  # noqa: E402
    CLASSES, TAPS, captured_head, deployed_match_map, frame_record, target_frame,
)
from scripts.run_temporal_representation_p0 import compare_outputs, compare_states  # noqa: E402


REPORT = ROOT / "reports/care3d/p0_counterfactual_vulnerability"
PROTOCOL_PATHS = {
    "blur_back": ROOT / "protocols/presets/motion_blur_back_10f_s09.json",
    "crash_back": ROOT / "protocols/presets/camera_crash_back_10f.json",
    "dark_back": ROOT / "protocols/presets/dark_back_10f_s09.json",
}
CAMERA_ORDER = (
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT",
)
SCHEMA = 1
K = 100
STOP_REQUESTED = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("probe_train", "probe_val", "probe_test"))
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--scene-token")
    parser.add_argument("--engineering-scene", action="store_true")
    parser.add_argument("--formal-all", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.engineering_scene and any((args.split, args.scene_token, args.formal_all)):
        parser.error("--engineering-scene cannot be combined with formal selectors")
    if not args.engineering_scene and not any((args.split, args.scene_token, args.formal_all)):
        parser.error("formal extraction requires --split, --scene-token, or --formal-all")
    return args


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def camera_support(info: dict, center_lidar: np.ndarray) -> np.ndarray:
    """Six-camera FOV support from predicted center and calibration only."""
    center = np.asarray(center_lidar, dtype=float).reshape(3)
    cams = info.get("cams")
    if not isinstance(cams, dict):
        raise RuntimeError("nuScenes info has no per-camera calibration dictionary")
    output = []
    for name in CAMERA_ORDER:
        if name not in cams:
            raise RuntimeError(f"camera calibration missing: {name}")
        cam = cams[name]
        if not all(key in cam for key in ("sensor2lidar_rotation", "sensor2lidar_translation", "cam_intrinsic")):
            raise RuntimeError(f"camera calibration schema changed: {name}")
        rotation = np.asarray(cam["sensor2lidar_rotation"], dtype=float).reshape(3, 3)
        translation = np.asarray(cam["sensor2lidar_translation"], dtype=float).reshape(3)
        intrinsic = np.asarray(cam["cam_intrinsic"], dtype=float).reshape(3, 3)
        camera_point = rotation.T @ (center - translation)
        if camera_point[2] <= 1e-3:
            output.append(0.0)
            continue
        pixel = intrinsic @ camera_point
        u, v = pixel[0] / pixel[2], pixel[1] / pixel[2]
        width = float(cam.get("width", 1600))
        height = float(cam.get("height", 900))
        output.append(float(0.0 <= u < width and 0.0 <= v < height))
    value = np.asarray(output, dtype=np.float32)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise RuntimeError("invalid camera support")
    return value


def scene_rows(args) -> pd.DataFrame:
    if args.engineering_scene:
        return pd.read_csv(REPORT / "engineering_scene_manifest.csv")
    frame = pd.read_csv(REPORT / "frozen_scene_manifest.csv")
    if args.split:
        frame = frame[frame.split == args.split]
    if args.scene_token:
        frame = frame[frame.scene_token.astype(str) == args.scene_token]
    return frame.reset_index(drop=True)


def output_directory(engineering: bool) -> Path:
    return REPORT / ("engineering_smoke" if engineering else "incremental/P0")


def pending_rows(frame: pd.DataFrame, out: Path, validation: dict, max_scenes: int | None) -> list:
    pending = []
    for row in frame.itertuples(index=False):
        marker = out / f"{row.scene_token}.complete.json"
        if marker.exists():
            value = json.loads(marker.read_text())
            if value.get("complete") and value.get("schema_version") == SCHEMA \
                    and value.get("scene_manifest_sha256") == validation["scene_manifest_sha256"]:
                continue
        pending.append(row)
    if max_scenes is not None:
        pending = pending[:max_scenes]
    return pending


def update_formal_status(validation: dict) -> None:
    manifest = pd.read_csv(REPORT / "frozen_scene_manifest.csv")
    complete = {}
    for marker in (REPORT / "incremental/P0").glob("*.complete.json"):
        value = json.loads(marker.read_text())
        if value.get("complete") and value.get("schema_version") == SCHEMA \
                and value.get("scene_manifest_sha256") == validation["scene_manifest_sha256"]:
            complete[str(value["scene_token"])] = value
    coverage = {}
    all_complete = True
    for split, group in manifest.groupby("split"):
        expected = set(group.scene_token.astype(str))
        observed = expected & set(complete)
        coverage[str(split)] = {"completed_scenes": len(observed), "expected_scenes": len(expected)}
        all_complete &= observed == expected
    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    progress["stages"]["formal_extraction"] = coverage
    progress["status"] = ("P0_COUNTERFACTUAL_EXTRACTION_COMPLETE_TRAINING_PENDING"
                          if all_complete else "P0_COUNTERFACTUAL_EXTRACTION_RUNNING")
    if all_complete:
        progress["stages"]["training"] = "ELIGIBLE"
    atomic_json(REPORT / "progress_manifest.json", progress)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    validation = json.loads((REPORT / "source_validation.json").read_text())
    if validation.get("status") != "VALIDATED_BEFORE_FORWARD":
        raise RuntimeError("run scripts/prepare_care3d_p0.py first")

    rows = scene_rows(args)
    out = output_directory(args.engineering_scene)
    out.mkdir(parents=True, exist_ok=True)
    pending = pending_rows(rows, out, validation, args.max_scenes)
    if not pending:
        print("no pending CARE-3D P0 scenes")
        if not args.engineering_scene:
            update_formal_status(validation)
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
    fault_datasets = {name: protocol_dataset(cfg, path) for name, path in PROTOCOL_PATHS.items()}
    token_index = {str(info["token"]): index for index, info in enumerate(clean_dataset.data_infos)}
    for name, dataset in fault_datasets.items():
        if token_index != {str(info["token"]): index for index, info in enumerate(dataset.data_infos)}:
            raise RuntimeError(f"paired dataset mismatch: {name}")

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(CHECKPOINT), map_location="cpu")
    model = freeze_module(model.to(device))
    head = model.pts_bbox_head
    head.reset_memory()
    initial = snapshot(head)
    pc_range = head.pc_range.detach()
    if int(head.num_query) != 644 or int(head.num_propagated) != 256:
        raise RuntimeError("StreamPETR query layout changed")
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(DATA), verbose=False)

    metadata_fields = [
        "sample_id", "split", "scene_token", "instance_token", "anchor_frame_idx",
        "target_frame_idx", "anchor_sample_token", "target_sample_token", "anchor_gt_token",
        "target_gt_token", "anchor_query_index", "target_clean_query_index", "prediction_class",
        "gt_used_as_input",
    ]
    for protocol in PROTOCOLS:
        metadata_fields += [
            f"{protocol}_clean_score", f"{protocol}_fault_score",
            f"{protocol}_clean_flat_rank", f"{protocol}_fault_flat_rank",
            f"{protocol}_clean_topk", f"{protocol}_fault_topk",
            f"{protocol}_evidence_drop", f"{protocol}_cross_topk", f"{protocol}_tp_to_fn",
        ]

    for scene_row in pending:
        started = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        scene = str(scene_row.scene_token)
        split = str(scene_row.split)
        tokens = json.loads(scene_row.sample_tokens_0_12)
        clean_state = initial
        anchor = None
        equivalence = []

        # Warm frames 0..2 on clean only.  Frame 2 is the first anchor.
        for frame_idx in range(3):
            index = token_index[tokens[frame_idx]]
            meta, image, data = unpack(clean_dataset[index], device)
            with torch.no_grad():
                _, _, feats = features(model, image)
                feats = feats.detach()
                pre_state = clean_state
                if frame_idx < 2:
                    output, clean_state, _ = run_head(model, meta, data, feats, frame_idx > 0, pre_state)
                    del output
                else:
                    output, clean_state, taps = captured_head(model, meta, data, feats, True, pre_state)
                    plain_output, plain_state, _ = run_head(model, meta, data, feats, True, pre_state)
                    out_equal, out_diff = compare_outputs(output, plain_output)
                    state_equal, state_diff = compare_states(clean_state, plain_state)
                    if not out_equal or not state_equal:
                        raise RuntimeError(f"passive tap changed B0 output/state: {scene}")
                    equivalence.append({
                        "scene_token": scene, "sample_token": tokens[frame_idx], "frame_idx": frame_idx,
                        "check": "passive_hooks_vs_unhooked_B0", "output_bitwise_equal": out_equal,
                        "output_max_abs_diff": out_diff, "memory_bitwise_equal": state_equal,
                        "memory_max_abs_diff": state_diff,
                    })
            targets = target_frame(nusc, tokens[frame_idx])
            context = frame_context(clean_dataset.data_infos[index], clean_dataset)
            candidates, matches = frame_record(output, taps, pre_state, data, targets, context, pc_range,
                                               int(head.num_query))
            anchor = {
                "frame_idx": frame_idx, "index": index, "data": data, "output": output,
                "taps": taps, "targets": targets, "candidates": candidates, "matches": matches,
            }

        if anchor is None:
            raise RuntimeError("clean anchor was not created")

        sample_rows = []
        object_arrays, temporal_arrays, decision_arrays = [], [], []
        support_arrays, quality_arrays = [], []
        drop_arrays, cross_arrays, fn_arrays, valid_arrays = [], [], [], []

        for target_frame_idx in range(3, 13):
            anchor_frame_idx = target_frame_idx - 1
            if anchor["frame_idx"] != anchor_frame_idx:
                raise RuntimeError("clean anchor progression changed")
            target_index = token_index[tokens[target_frame_idx]]
            branch_states = clone_counterfactual_states(clean_state, 1 + len(PROTOCOLS))
            if not all(states_exact(branch_states[0], value) for value in branch_states[1:]):
                raise RuntimeError("counterfactual branches do not start from identical H_t")

            clean_meta, clean_image, clean_data = unpack(clean_dataset[target_index], device)
            with torch.no_grad():
                _, _, clean_feats = features(model, clean_image)
                clean_feats = clean_feats.detach()
                clean_next_output, clean_next_state, clean_next_taps = captured_head(
                    model, clean_meta, clean_data, clean_feats, True, branch_states[0])
            next_targets = target_frame(nusc, tokens[target_frame_idx])
            next_context = frame_context(clean_dataset.data_infos[target_index], clean_dataset)
            clean_next_candidates, clean_next_matches = frame_record(
                clean_next_output, clean_next_taps, branch_states[0], clean_data, next_targets,
                next_context, pc_range, int(head.num_query))
            clean_logits = clean_next_output["all_cls_scores"][-1, 0].detach().float().cpu().numpy()

            fault_outputs, fault_matches = {}, {}
            for protocol_index, protocol in enumerate(PROTOCOLS, start=1):
                fault_meta, fault_image, fault_data = unpack(fault_datasets[protocol][target_index], device)
                with torch.no_grad():
                    _, _, fault_feats = features(model, fault_image)
                    fault_output, _, _ = run_head(
                        model, fault_meta, fault_data, fault_feats.detach(), True, branch_states[protocol_index])
                fault_outputs[protocol] = fault_output
                fault_logits = fault_output["all_cls_scores"][-1, 0].detach().float().cpu().numpy()
                fault_boxes = physical(fault_output, pc_range)[-1, 0].detach().float().cpu().numpy()
                fault_matches[protocol] = deployed_match_map(
                    fault_logits, fault_boxes, next_targets, next_context)
                del fault_feats, fault_image, fault_data

            next_by_instance = {target["instance_token"]: target for target in next_targets}
            for instance_token, anchor_candidate in anchor["candidates"].items():
                if instance_token not in next_by_instance or instance_token not in clean_next_candidates:
                    continue
                next_target = next_by_instance[instance_token]
                clean_candidate = clean_next_candidates[instance_token]
                clean_prediction = clean_candidate["prediction"]
                query = int(clean_prediction["query"])
                label = int(clean_prediction["label"])

                anchor_rep = anchor_candidate["representation"]
                predictor_payload = {
                    "object_features": np.asarray(anchor_rep[TAPS[2]], np.float32),
                    "temporal_features": np.asarray(anchor_rep[TAPS[1]], np.float32),
                    "decision_features": np.asarray(anchor_candidate["observable"], np.float32),
                    "camera_support": camera_support(
                        clean_dataset.data_infos[anchor["index"]], anchor_candidate["prediction"]["box"][:3]),
                    "camera_quality": np.ones(6, dtype=np.float32),
                }
                assert_prospective_payload(predictor_payload)
                if predictor_payload["object_features"].shape != (256,) \
                        or predictor_payload["temporal_features"].shape != (256,) \
                        or predictor_payload["decision_features"].shape != (21,):
                    raise RuntimeError("CARE predictor feature layout changed")

                row = {
                    "sample_id": f"{scene}:{instance_token}:{anchor_frame_idx}:{target_frame_idx}",
                    "split": split, "scene_token": scene, "instance_token": instance_token,
                    "anchor_frame_idx": anchor_frame_idx, "target_frame_idx": target_frame_idx,
                    "anchor_sample_token": tokens[anchor_frame_idx],
                    "target_sample_token": tokens[target_frame_idx],
                    "anchor_gt_token": anchor_candidate["target"]["token"],
                    "target_gt_token": next_target["token"],
                    "anchor_query_index": int(anchor_candidate["prediction"]["query"]),
                    "target_clean_query_index": query,
                    "prediction_class": CLASSES[label], "gt_used_as_input": False,
                }
                drops, crosses, fns, valids = [], [], [], []
                for protocol in PROTOCOLS:
                    fault_logits = fault_outputs[protocol]["all_cls_scores"][-1, 0].detach().float().cpu().numpy()
                    label_values = same_query_counterfactual(clean_logits, fault_logits, query, label, K)
                    tp_to_fn = int(str(next_target["token"]) not in fault_matches[protocol])
                    row.update({
                        f"{protocol}_clean_score": label_values["clean_score"],
                        f"{protocol}_fault_score": label_values["fault_score"],
                        f"{protocol}_clean_flat_rank": label_values["clean_flat_rank"],
                        f"{protocol}_fault_flat_rank": label_values["fault_flat_rank"],
                        f"{protocol}_clean_topk": int(label_values["clean_topk"]),
                        f"{protocol}_fault_topk": int(label_values["fault_topk"]),
                        f"{protocol}_evidence_drop": label_values["evidence_drop"],
                        f"{protocol}_cross_topk": int(label_values["cross_topk"]),
                        f"{protocol}_tp_to_fn": tp_to_fn,
                    })
                    drops.append(float(label_values["evidence_drop"]))
                    crosses.append(int(label_values["cross_topk"]))
                    fns.append(tp_to_fn)
                    valids.append(1)

                sample_rows.append(row)
                object_arrays.append(predictor_payload["object_features"])
                temporal_arrays.append(predictor_payload["temporal_features"])
                decision_arrays.append(predictor_payload["decision_features"])
                support_arrays.append(predictor_payload["camera_support"])
                quality_arrays.append(predictor_payload["camera_quality"])
                drop_arrays.append(drops); cross_arrays.append(crosses)
                fn_arrays.append(fns); valid_arrays.append(valids)

            # Only the clean branch advances.  Fault states are discarded here.
            clean_state = clean_next_state
            anchor = {
                "frame_idx": target_frame_idx, "index": target_index, "data": clean_data,
                "output": clean_next_output, "taps": clean_next_taps, "targets": next_targets,
                "candidates": clean_next_candidates, "matches": clean_next_matches,
            }
            for output in fault_outputs.values():
                del output

        assert_unique_sample_ids([row["sample_id"] for row in sample_rows])
        n = len(sample_rows)
        arrays = {
            "object_features": np.stack(object_arrays).astype(np.float32) if n else np.empty((0, 256), np.float32),
            "temporal_features": np.stack(temporal_arrays).astype(np.float32) if n else np.empty((0, 256), np.float32),
            "decision_features": np.stack(decision_arrays).astype(np.float32) if n else np.empty((0, 21), np.float32),
            "camera_support": np.stack(support_arrays).astype(np.float32) if n else np.empty((0, 6), np.float32),
            "camera_quality": np.stack(quality_arrays).astype(np.float32) if n else np.empty((0, 6), np.float32),
            "evidence_drop": np.asarray(drop_arrays, np.float32).reshape(n, 3),
            "cross_topk": np.asarray(cross_arrays, np.int8).reshape(n, 3),
            "tp_to_fn": np.asarray(fn_arrays, np.int8).reshape(n, 3),
            "valid_mask": np.asarray(valid_arrays, np.int8).reshape(n, 3),
        }
        prefix = out / scene
        atomic_csv(prefix.with_suffix(".samples.csv"), sample_rows, metadata_fields)
        atomic_csv(prefix.with_suffix(".equivalence.csv"), equivalence,
                   ["scene_token", "sample_token", "frame_idx", "check", "output_bitwise_equal",
                    "output_max_abs_diff", "memory_bitwise_equal", "memory_max_abs_diff"])
        atomic_npz(prefix.with_suffix(".features.npz"), **arrays)
        elapsed = time.time() - started
        peak_gib = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
        summary = {
            "schema_version": SCHEMA,
            "scene_manifest_sha256": validation["scene_manifest_sha256"],
            "scene_token": scene, "split": split, "rows": n,
            "drop_mean": {p: float(arrays["evidence_drop"][:, i].mean()) if n else 0.0
                          for i, p in enumerate(PROTOCOLS)},
            "cross_topk_positives": {p: int(arrays["cross_topk"][:, i].sum())
                                     for i, p in enumerate(PROTOCOLS)},
            "tp_to_fn_positives": {p: int(arrays["tp_to_fn"][:, i].sum())
                                   for i, p in enumerate(PROTOCOLS)},
            "equivalence_pass": all(bool(row["output_bitwise_equal"]) and bool(row["memory_bitwise_equal"])
                                    for row in equivalence),
            "peak_cuda_gib": peak_gib, "elapsed_seconds": elapsed, "complete": True,
        }
        atomic_json(prefix.with_suffix(".complete.json"), summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

        if args.engineering_scene:
            progress = json.loads((REPORT / "progress_manifest.json").read_text())
            if not summary["equivalence_pass"]:
                progress["status"] = "ENGINEERING_SMOKE_FAILED"
                progress["stages"]["engineering_smoke"] = "FAILED"
            else:
                progress["status"] = "ENGINEERING_SMOKE_PASSED_FORMAL_EXTRACTION_ELIGIBLE"
                progress["stages"]["engineering_smoke"] = "PASSED"
                progress["stages"]["formal_extraction"] = "ELIGIBLE"
            atomic_json(REPORT / "progress_manifest.json", progress)
        else:
            update_formal_status(validation)

        if STOP_REQUESTED:
            print("stop requested; current CARE scene saved", flush=True)
            break


if __name__ == "__main__":
    main()
