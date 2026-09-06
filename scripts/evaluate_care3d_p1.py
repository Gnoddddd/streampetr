#!/usr/bin/env python3
"""Evaluate frozen CARE-3D P1 routers on the locked 132-scene probe-test split."""

from __future__ import annotations

import argparse
import hashlib
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

from analysis.care3d_counterfactual import clone_counterfactual_states, freeze_module, states_exact  # noqa: E402
from analysis.care3d_p1 import (  # noqa: E402
    PROTOCOLS,
    SOURCE_NAMES,
    assert_unique_queries,
    build_source_bank,
    fixed_target_metrics,
    sample_projected_camera_tokens,
)
from models.care3d import CARE3DCore  # noqa: E402
from models.care3d_p1 import CARE3DP1ScoreRouter  # noqa: E402
from scripts.audit_dark_target_recoverability import (
    features, physical, run_head, snapshot, unpack,
)  # noqa: E402
from scripts.export_care3d_counterfactual_pairs import atomic_json  # noqa: E402
from scripts.export_care3d_p1_supervision import main_p0_source  # noqa: E402
from scripts.run_bd_temporal_support_p0 import (
    CHECKPOINT, CONFIG, DATA, frame_context, protocol_dataset,
)  # noqa: E402
from scripts.run_prospective_failure_features import (
    CLASSES, POST_RANGE, TAPS, captured_head, target_frame,
)  # noqa: E402

REPORT = ROOT / "reports/care3d/p1_sparse_evidence_router"
P0 = ROOT / "reports/care3d/p0_counterfactual_vulnerability"
PROTOCOL_PATHS = {
    "blur_back": ROOT / "protocols/presets/motion_blur_back_10f_s09.json",
    "crash_back": ROOT / "protocols/presets/camera_crash_back_10f.json",
    "dark_back": ROOT / "protocols/presets/dark_back_10f_s09.json",
}
SEEDS = (42, 2027, 2028)
SCHEMA = 1
STOP_REQUESTED = False
PREDICTOR_KEYS = (
    "object_features", "temporal_features", "decision_features",
    "camera_support", "camera_quality",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-test", action="store_true")
    parser.add_argument("--scene-token")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if int(args.formal_test) + int(args.scene_token is not None) != 1:
        parser.error("choose exactly one of --formal-test or --scene-token")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_training_complete() -> tuple[dict, dict]:
    validation = json.loads((REPORT / "source_validation.json").read_text())
    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    if progress.get("status") not in (
        "P1_TRAINING_COMPLETE_TEST_EVALUATION_ELIGIBLE",
        "P1_TEST_EVALUATION_RUNNING",
        "P1_TEST_EVALUATION_COMPLETE_ANALYSIS_PENDING",
    ):
        raise RuntimeError(f"P1 probe-test remains locked: {progress.get('status')}")
    for seed in SEEDS:
        directory = REPORT / "training" / f"seed_{seed}"
        manifest_path = directory / "training_manifest.json"
        checkpoint = directory / "best.pth"
        if not manifest_path.exists() or not checkpoint.exists():
            raise RuntimeError(f"missing frozen P1 seed {seed}")
        meta = json.loads(manifest_path.read_text())
        if meta.get("status") != "P1_TRAINING_COMPLETE_TEST_UNSEEN" \
                or meta.get("probe_test_read") is not False:
            raise RuntimeError(f"P1 seed {seed} is not test-blind")
        if sha256(checkpoint) != meta.get("checkpoint_sha256"):
            raise RuntimeError(f"P1 seed {seed} checkpoint changed after training")
    return validation, progress


def build_p0(seed: int, device: torch.device, validation: dict):
    path = P0 / "training" / f"seed_{seed}" / "best.pth"
    if sha256(path) != validation["p0_checkpoint_sha256"][str(seed)]:
        raise RuntimeError(f"P0 seed {seed} checkpoint changed")
    payload = torch.load(path, map_location="cpu")
    model = CARE3DCore(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if model.router is not None:
        raise RuntimeError(f"P0 seed {seed} unexpectedly contains a router")
    return freeze_module(model.to(device))


def build_p1(seed: int, device: torch.device):
    directory = REPORT / "training" / f"seed_{seed}"
    payload = torch.load(directory / "best.pth", map_location="cpu")
    model = CARE3DP1ScoreRouter(**payload["router_config"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if tuple(payload.get("source_names", ())) != tuple(SOURCE_NAMES):
        raise RuntimeError(f"P1 seed {seed} source bank changed")
    return model.to(device).eval()


def p0_forward(model, predictor_inputs):
    output = model(
        object_features=predictor_inputs["object_features"].unsqueeze(1),
        camera_support=predictor_inputs["camera_support"].unsqueeze(1),
        camera_quality=predictor_inputs["camera_quality"].unsqueeze(1),
        temporal_features=predictor_inputs["temporal_features"].unsqueeze(1),
        decision_features=predictor_inputs["decision_features"].unsqueeze(1),
    )
    return {
        "vulnerability": output["vulnerability"][:, 0],
        "boundary_crossing_logits": output["boundary_crossing_logits"][:, 0],
    }


def deployment_stats(logits, boxes, targets, context):
    scores = torch.as_tensor(logits).float().sigmoid().reshape(-1)
    values, indexes = torch.topk(scores, min(100, scores.numel()))
    predictions = []
    class_count = int(logits.shape[1])
    for rank, (score, flat) in enumerate(zip(values.tolist(), indexes.tolist()), start=1):
        if score < 0.1:
            continue
        query, label = divmod(int(flat), class_count)
        center = np.asarray(boxes[query, :3], dtype=float)
        if np.any(center < POST_RANGE[:3]) or np.any(center > POST_RANGE[3:]):
            continue
        ego = context["lidar2ego_rotation"] @ center + context["lidar2ego_translation"]
        if np.linalg.norm(ego[:2]) > context["class_range"][CLASSES[label]]:
            continue
        global_center = context["ego2global_rotation"] @ ego + context["ego2global_translation"]
        predictions.append({
            "query": query,
            "label": label,
            "score": float(score),
            "flat_rank": rank,
            "global_center": global_center,
        })
    pairs = []
    for gt_index, target in enumerate(targets):
        for prediction_index, prediction in enumerate(predictions):
            if int(target["label"]) != int(prediction["label"]):
                continue
            distance = np.linalg.norm(
                np.asarray(target["global_center"][:2], float)
                - np.asarray(prediction["global_center"][:2], float)
            )
            if distance <= 2.0:
                pairs.append((float(distance), gt_index, prediction_index))
    used_gt, used_prediction, matches = set(), set(), {}
    for distance, gt_index, prediction_index in sorted(pairs):
        if gt_index in used_gt or prediction_index in used_prediction:
            continue
        used_gt.add(gt_index)
        used_prediction.add(prediction_index)
        value = dict(predictions[prediction_index])
        value["match_distance_m"] = distance
        matches[str(targets[gt_index]["token"])] = value
    return {
        "matches": matches,
        "prediction_count": len(predictions),
        "tp": len(matches),
        "fp": len(predictions) - len(matches),
    }


def to_tensor(value, device, dtype=torch.float32):
    return torch.as_tensor(value, device=device, dtype=dtype)


def update_progress(validation: dict) -> None:
    manifest = pd.read_csv(REPORT / "frozen_scene_manifest.csv")
    expected = set(
        manifest[manifest.split.astype(str) == "probe_test"].scene_token.astype(str)
    )
    observed = set()
    for marker in (REPORT / "evaluation/probe_test").glob("*.complete.json"):
        value = json.loads(marker.read_text())
        if value.get("complete") and value.get("schema_version") == SCHEMA \
                and value.get("scene_manifest_sha256") == validation["scene_manifest_sha256"]:
            observed.add(str(value["scene_token"]))
    progress_path = REPORT / "progress_manifest.json"
    progress = json.loads(progress_path.read_text())
    progress["stages"]["probe_test_evaluation"] = {
        "completed_scenes": len(expected & observed),
        "expected_scenes": len(expected),
    }
    if len(expected & observed) == len(expected):
        progress["status"] = "P1_TEST_EVALUATION_COMPLETE_ANALYSIS_PENDING"
        progress["stages"]["analysis"] = "ELIGIBLE"
    else:
        progress["status"] = "P1_TEST_EVALUATION_RUNNING"
    atomic_json(progress_path, progress)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    validation, _ = require_training_complete()
    manifest = pd.read_csv(REPORT / "frozen_scene_manifest.csv")
    rows = manifest[manifest.split.astype(str) == "probe_test"].reset_index(drop=True)
    if len(rows) != 132:
        raise RuntimeError("P1 frozen probe-test split is not 132 scenes")
    if args.scene_token:
        rows = rows[rows.scene_token.astype(str) == str(args.scene_token)].reset_index(drop=True)
        if len(rows) != 1:
            raise RuntimeError("scene-token is not in frozen probe_test")
    if args.max_scenes is not None:
        rows = rows.iloc[: int(args.max_scenes)].reset_index(drop=True)
    out = REPORT / "evaluation/probe_test"
    out.mkdir(parents=True, exist_ok=True)
    pending = []
    for row in rows.itertuples(index=False):
        marker = out / f"{row.scene_token}.complete.json"
        if marker.exists():
            value = json.loads(marker.read_text())
            if value.get("complete") and value.get("scene_manifest_sha256") == validation["scene_manifest_sha256"]:
                continue
        pending.append(row)
    if not pending:
        print("no pending CARE-3D P1 probe-test scenes")
        update_progress(validation)
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
    fault_datasets = {
        protocol: protocol_dataset(cfg, path) for protocol, path in PROTOCOL_PATHS.items()
    }
    token_index = {str(info["token"]): index for index, info in enumerate(clean_dataset.data_infos)}
    for protocol, dataset in fault_datasets.items():
        other = {str(info["token"]): index for index, info in enumerate(dataset.data_infos)}
        if token_index != other:
            raise RuntimeError(f"P1 test dataset mismatch: {protocol}")

    detector = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(detector, str(CHECKPOINT), map_location="cpu")
    detector = freeze_module(detector.to(device))
    head = detector.pts_bbox_head
    head.reset_memory()
    initial = snapshot(head)
    pc_range = head.pc_range.detach()
    classifier = head.cls_branches[-1]
    p0_models = {seed: build_p0(seed, device, validation) for seed in SEEDS}
    p1_models = {seed: build_p1(seed, device) for seed in SEEDS}
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(DATA), verbose=False)

    for scene_row in pending:
        started = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        scene = str(scene_row.scene_token)
        tokens = json.loads(scene_row.sample_tokens_0_12)
        main_frame, main_arrays = main_p0_source(scene, False)
        clean_state = initial
        anchor = None
        clean_identity_pass = True
        object_rows, frame_rows = [], []

        for frame_idx in range(3):
            index = token_index[tokens[frame_idx]]
            meta, image, data = unpack(clean_dataset[index], device)
            with torch.no_grad():
                _, _, feats = features(detector, image)
                if frame_idx < 2:
                    output, clean_state, _ = run_head(
                        detector, meta, data, feats.detach(), frame_idx > 0, clean_state
                    )
                    del output, image, data, feats
                    continue
                output, clean_state, taps = captured_head(
                    detector, meta, data, feats.detach(), True, clean_state
                )
            anchor = {"frame_idx": frame_idx, "output": output, "taps": taps}

        if anchor is None:
            raise RuntimeError("P1 test clean anchor missing")

        for target_frame_idx in range(3, 13):
            row_indices = np.flatnonzero(
                main_frame.target_frame_idx.to_numpy(dtype=int) == target_frame_idx
            )
            queries = main_frame.target_clean_query_index.to_numpy(dtype=int)[row_indices]
            if row_indices.size:
                assert_unique_queries(queries.tolist())
            target_index = token_index[tokens[target_frame_idx]]
            branch_states = clone_counterfactual_states(clean_state, 1 + len(PROTOCOLS))
            if not all(states_exact(branch_states[0], state) for state in branch_states[1:]):
                raise RuntimeError("P1 test counterfactual branches do not share H_t")

            clean_meta, clean_image, clean_data = unpack(clean_dataset[target_index], device)
            with torch.no_grad():
                _, _, clean_feats = features(detector, clean_image)
                clean_output, clean_next_state, clean_taps = captured_head(
                    detector, clean_meta, clean_data, clean_feats.detach(), True, branch_states[0]
                )
            targets = target_frame(nusc, tokens[target_frame_idx])
            context = frame_context(clean_dataset.data_infos[target_index], clean_dataset)

            for protocol_index, protocol in enumerate(PROTOCOLS, start=1):
                fault_meta, fault_image, fault_data = unpack(
                    fault_datasets[protocol][target_index], device
                )
                with torch.no_grad():
                    _, fault_pyramid, fault_feats = features(detector, fault_image)
                    fault_output, _, fault_taps = captured_head(
                        detector, fault_meta, fault_data, fault_feats.detach(), True,
                        branch_states[protocol_index],
                    )
                p_index = protocol_index - 1
                base_logits = fault_output["all_cls_scores"][-1, 0].detach().float().cpu().numpy()
                boxes_tensor = physical(fault_output, pc_range)[-1, 0].detach().float()
                boxes = boxes_tensor.cpu().numpy()
                base = deployment_stats(base_logits, boxes, targets, context)

                if row_indices.size:
                    q_tensor = torch.as_tensor(queries, device=device, dtype=torch.long)
                    fault_query = fault_taps[TAPS[2]][q_tensor].detach().float()
                    centers = boxes_tensor[q_tensor, :3]
                    camera_tokens, camera_reliability = sample_projected_camera_tokens(
                        fault_pyramid[0],
                        fault_data["lidar2img"][0].detach().float(),
                        centers,
                        tuple(int(value) for value in fault_image.shape[-2:]),
                    )
                    temporal = to_tensor(main_arrays["object_features"][row_indices], device)
                    sources, reliability = build_source_bank(camera_tokens, camera_reliability, temporal)
                    predictor_inputs = {
                        key: to_tensor(main_arrays[key][row_indices], device)
                        for key in PREDICTOR_KEYS
                    }
                    class_map = {name: index for index, name in enumerate(CLASSES)}
                    target_class = main_frame.prediction_class.iloc[row_indices].map(class_map).to_numpy(dtype=int)
                else:
                    fault_query = None
                    sources = reliability = None
                    predictor_inputs = None
                    target_class = np.empty((0,), dtype=int)

                for seed in SEEDS:
                    patched_logits = base_logits.copy()
                    risks = np.empty((0,), dtype=float)
                    selected_sources = []
                    if row_indices.size:
                        with torch.no_grad():
                            p0_output = p0_forward(p0_models[seed], predictor_inputs)
                            protocol_tensor = torch.full(
                                (len(row_indices),), p_index, device=device, dtype=torch.long
                            )
                            routed, aux = p1_models[seed](
                                fault_query, sources, reliability,
                                p0_output["vulnerability"],
                                p0_output["boundary_crossing_logits"],
                                protocol_tensor, fault_active=True,
                            )
                            routed_logits = classifier(routed).detach().float().cpu().numpy()
                            risks = aux["risk_probability"].detach().float().cpu().numpy()
                            topk_indices = aux.get("topk_indices")
                            if topk_indices is not None:
                                topk_indices = topk_indices.detach().cpu().numpy()
                                selected_sources = [
                                    ";".join(SOURCE_NAMES[int(value)] for value in values)
                                    for values in topk_indices
                                ]
                            else:
                                selected_sources = [""] * len(row_indices)
                        for local, query in enumerate(queries):
                            patched_logits[int(query), :] = routed_logits[local]
                    patched = deployment_stats(patched_logits, boxes, targets, context)
                    frame_rows.append({
                        "seed": seed, "protocol": protocol, "scene_token": scene,
                        "target_frame_idx": target_frame_idx,
                        "target_sample_token": tokens[target_frame_idx],
                        "base_predictions": base["prediction_count"],
                        "patched_predictions": patched["prediction_count"],
                        "base_tp": base["tp"], "patched_tp": patched["tp"],
                        "base_fp": base["fp"], "patched_fp": patched["fp"],
                    })
                    for local, row_index in enumerate(row_indices):
                        row = main_frame.iloc[int(row_index)]
                        query = int(queries[local])
                        label = int(target_class[local])
                        before = fixed_target_metrics(base_logits, query, label, 100)
                        after = fixed_target_metrics(patched_logits, query, label, 100)
                        gt_token = str(row.target_gt_token)
                        base_tp = int(gt_token in base["matches"])
                        patched_tp = int(gt_token in patched["matches"])
                        base_cross = int(main_arrays["cross_topk"][row_index, p_index])
                        object_rows.append({
                            "seed": seed, "protocol": protocol, "scene_token": scene,
                            "instance_token": str(row.instance_token),
                            "sample_id": str(row.sample_id),
                            "target_frame_idx": target_frame_idx,
                            "target_sample_token": tokens[target_frame_idx],
                            "target_gt_token": gt_token, "query_index": query,
                            "prediction_class": str(row.prediction_class),
                            "base_tp": base_tp, "patched_tp": patched_tp,
                            "lost_recovered": int((not base_tp) and patched_tp),
                            "retained_damaged": int(base_tp and (not patched_tp)),
                            "tp_delta": patched_tp - base_tp,
                            "base_cross_topk": base_cross,
                            "base_target_score": before["score"],
                            "patched_target_score": after["score"],
                            "target_score_delta": after["score"] - before["score"],
                            "base_target_rank": before["rank"],
                            "patched_target_rank": after["rank"],
                            "cross_topk_recovered": int(base_cross and bool(after["topk"])),
                            "risk_probability": float(risks[local]),
                            "selected_sources": selected_sources[local],
                            "gt_used_as_router_input": False,
                            "clean_future_used_as_router_input": False,
                        })

                del fault_output, fault_taps, fault_pyramid, fault_feats, fault_image, fault_data

            if row_indices.size:
                seed = SEEDS[0]
                query = int(queries[0])
                clean_query = clean_taps[TAPS[2]][query : query + 1].detach().float()
                dummy_sources = clean_query[:, None, :].repeat(1, 3, 1)
                dummy_reliability = clean_query.new_ones((1, 3))
                dummy_vulnerability = clean_query.new_zeros((1, 3))
                dummy_logits = clean_query.new_zeros((1, 3))
                dummy_protocol = torch.zeros((1,), dtype=torch.long, device=device)
                with torch.no_grad():
                    bypass, _ = p1_models[seed](
                        clean_query, dummy_sources, dummy_reliability,
                        dummy_vulnerability, dummy_logits, dummy_protocol,
                        fault_active=False,
                    )
                clean_identity_pass &= bool(torch.equal(bypass, clean_query))

            clean_state = clean_next_state
            anchor = {"frame_idx": target_frame_idx, "output": clean_output, "taps": clean_taps}
            del clean_image, clean_data, clean_feats

        prefix = out / scene
        pd.DataFrame(object_rows).to_csv(prefix.with_suffix(".objects.csv"), index=False)
        pd.DataFrame(frame_rows).to_csv(prefix.with_suffix(".frames.csv"), index=False)
        summary = {
            "schema_version": SCHEMA,
            "scene_manifest_sha256": validation["scene_manifest_sha256"],
            "scene_token": scene,
            "split": "probe_test",
            "object_rows": len(object_rows),
            "frame_rows": len(frame_rows),
            "clean_identity_pass": bool(clean_identity_pass),
            "seeds": list(SEEDS),
            "protocols": list(PROTOCOLS),
            "peak_cuda_gib": float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)),
            "elapsed_seconds": time.time() - started,
            "complete": True,
        }
        atomic_json(prefix.with_suffix(".complete.json"), summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        update_progress(validation)
        if STOP_REQUESTED:
            print("stop requested; current P1 probe-test scene saved", flush=True)
            break


if __name__ == "__main__":
    main()
