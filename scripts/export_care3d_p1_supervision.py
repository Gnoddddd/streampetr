#!/usr/bin/env python3
"""Export CARE-3D P1 sparse-router supervision from full nuScenes.

Only the frozen discovery smoke scene plus probe_train/probe_val are permitted.
Probe-test is intentionally absent from this script and remains locked until all
three P1 router checkpoints are frozen.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STREAM = ROOT / "repos/StreamPETR"
sys.dont_writebytecode = True
sys.path.insert(0, str(STREAM))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmcv.utils import import_modules_from_strings  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402

from analysis.care3d_counterfactual import (  # noqa: E402
    clone_counterfactual_states,
    freeze_module,
    states_exact,
)
from analysis.care3d_p1 import (  # noqa: E402
    PROTOCOLS,
    QUERY_COLLISION_POLICY,
    SOURCE_NAMES,
    assert_exact_sample_alignment,
    assert_source_contract,
    assert_unique_queries,
    build_source_bank,
    filter_aligned_rows,
    sample_projected_camera_tokens,
    topk_score_threshold,
)
from scripts.audit_dark_target_recoverability import (  # noqa: E402
    features,
    physical,
    run_head,
    snapshot,
    unpack,
)
from scripts.export_care3d_counterfactual_pairs import (  # noqa: E402
    atomic_json,
    atomic_npz,
)
from scripts.run_bd_temporal_support_p0 import (  # noqa: E402
    CHECKPOINT,
    CONFIG,
    protocol_dataset,
)
from scripts.run_prospective_failure_features import CLASSES, TAPS, captured_head  # noqa: E402
from scripts.run_temporal_representation_p0 import compare_outputs, compare_states  # noqa: E402


P0 = ROOT / "reports/care3d/p0_counterfactual_vulnerability"
REPORT = ROOT / "reports/care3d/p1_sparse_evidence_router"
PROTOCOL_PATHS = {
    "blur_back": ROOT / "protocols/presets/motion_blur_back_10f_s09.json",
    "crash_back": ROOT / "protocols/presets/camera_crash_back_10f.json",
    "dark_back": ROOT / "protocols/presets/dark_back_10f_s09.json",
}
PREDICTOR_KEYS = (
    "object_features",
    "temporal_features",
    "decision_features",
    "camera_support",
    "camera_quality",
)
SCHEMA = 1
STOP_REQUESTED = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engineering-scene", action="store_true")
    parser.add_argument("--split", choices=("probe_train", "probe_val"))
    parser.add_argument("--formal-train-val", action="store_true")
    parser.add_argument("--scene-token")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    selectors = int(args.engineering_scene) + int(args.formal_train_val) \
        + int(args.split is not None) + int(args.scene_token is not None)
    if selectors != 1:
        parser.error(
            "choose exactly one of --engineering-scene, --formal-train-val, "
            "--split, --scene-token"
        )
    if args.engineering_scene and args.max_scenes is not None:
        parser.error("--max-scenes is not valid for engineering smoke")
    return args


def load_validation() -> dict:
    path = REPORT / "source_validation.json"
    if not path.exists():
        raise RuntimeError("run scripts/prepare_care3d_p1.py first")
    value = json.loads(path.read_text())
    if value.get("status") != "VALIDATED_BEFORE_P1_FORWARD":
        raise RuntimeError("P1 source validation is not frozen")
    return value


def selected_scenes(args) -> pd.DataFrame:
    if args.engineering_scene:
        frame = pd.read_csv(REPORT / "engineering_scene_manifest.csv")
        if len(frame) != 1:
            raise RuntimeError("expected one P1 engineering scene")
        return frame
    frame = pd.read_csv(REPORT / "frozen_scene_manifest.csv")
    frame = frame[frame.split.astype(str).isin(("probe_train", "probe_val"))]
    if args.formal_train_val:
        pass
    elif args.split:
        frame = frame[frame.split.astype(str) == args.split]
    elif args.scene_token:
        frame = frame[frame.scene_token.astype(str) == str(args.scene_token)]
        if len(frame) != 1:
            raise RuntimeError("scene-token is not a frozen probe_train/probe_val scene")
    frame = frame.reset_index(drop=True)
    if args.max_scenes is not None:
        frame = frame.iloc[: int(args.max_scenes)].reset_index(drop=True)
    return frame


def output_dir(engineering: bool) -> Path:
    return REPORT / ("engineering_smoke" if engineering else "incremental/supervision")


def main_p0_source(scene: str, engineering: bool):
    directory = P0 / ("engineering_smoke" if engineering else "incremental/P0")
    prefix = directory / scene
    sample_path = prefix.with_suffix(".samples.csv")
    feature_path = prefix.with_suffix(".features.npz")
    marker_path = prefix.with_suffix(".complete.json")
    for path in (sample_path, feature_path, marker_path):
        if not path.exists():
            raise RuntimeError(f"missing frozen P0 source: {path}")
    marker = json.loads(marker_path.read_text())
    if not marker.get("complete") or not marker.get("equivalence_pass"):
        raise RuntimeError(f"frozen P0 source failed equivalence: {scene}")
    frame = pd.read_csv(sample_path)
    with np.load(feature_path) as packed:
        arrays = {key: np.asarray(packed[key]).copy() for key in packed.files}
    for key in (*PREDICTOR_KEYS, "cross_topk", "tp_to_fn", "valid_mask"):
        if key not in arrays:
            raise RuntimeError(f"P0 source missing {key}: {scene}")
    if len(frame) != len(arrays["object_features"]):
        raise RuntimeError(f"P0 metadata/feature mismatch: {scene}")
    frame, arrays, audit = filter_aligned_rows(frame, arrays)
    if int(audit["query_collision_excluded_rows"]) > 0:
        print(json.dumps({
            "event": "P1_QUERY_COLLISION_EXCLUSION",
            "scene_token": scene,
            "policy": QUERY_COLLISION_POLICY,
            "p0_rows_total": int(audit["p0_rows_total"]),
            "p1_eligible_rows": int(audit["p1_eligible_rows"]),
            "query_collision_excluded_rows": int(audit["query_collision_excluded_rows"]),
            "query_collision_groups": int(audit["query_collision_groups"]),
        }, sort_keys=True), flush=True)
    return frame, arrays


def marker_valid(path: Path, validation: dict) -> bool:
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    return bool(
        value.get("complete")
        and value.get("schema_version") == SCHEMA
        and value.get("scene_manifest_sha256") == validation["scene_manifest_sha256"]
        and value.get("query_collision_policy") == QUERY_COLLISION_POLICY
    )


def require_smoke(validation: dict) -> None:
    engineering = pd.read_csv(REPORT / "engineering_scene_manifest.csv")
    scene = str(engineering.iloc[0].scene_token)
    marker = REPORT / "engineering_smoke" / f"{scene}.complete.json"
    if not marker_valid(marker, validation):
        raise RuntimeError("rerun P1 engineering smoke with the frozen query-collision policy")
    value = json.loads(marker.read_text())
    required = (
        "equivalence_pass",
        "predictor_input_identity_pass",
        "label_identity_pass",
        "source_contract_pass",
    )
    if not all(bool(value.get(key)) for key in required):
        raise RuntimeError(f"P1 engineering smoke did not pass all invariants: {value}")


def update_progress(validation: dict) -> None:
    manifest = pd.read_csv(REPORT / "frozen_scene_manifest.csv")
    progress_path = REPORT / "progress_manifest.json"
    progress = json.loads(progress_path.read_text())
    coverage = {}
    complete = True
    for split, expected in (("probe_train", 419), ("probe_val", 133)):
        wanted = set(manifest[manifest.split.astype(str) == split].scene_token.astype(str))
        observed = set()
        rows = 0
        p0_rows_total = 0
        excluded_rows = 0
        collision_groups = 0
        positives = {protocol: 0 for protocol in PROTOCOLS}
        for marker in (REPORT / "incremental/supervision").glob("*.complete.json"):
            value = json.loads(marker.read_text())
            if not marker_valid(marker, validation) or value.get("split") != split:
                continue
            scene = str(value["scene_token"])
            if scene in wanted:
                observed.add(scene)
                rows += int(value.get("rows", 0))
                p0_rows_total += int(value.get("p0_rows_total", value.get("rows", 0)))
                excluded_rows += int(value.get("query_collision_excluded_rows", 0))
                collision_groups += int(value.get("query_collision_groups", 0))
                for protocol in PROTOCOLS:
                    positives[protocol] += int(value.get("cross_topk_positives", {}).get(protocol, 0))
        coverage[split] = {
            "completed_scenes": len(observed),
            "expected_scenes": expected,
            "p0_rows_total": p0_rows_total,
            "rows": rows,
            "query_collision_excluded_rows": excluded_rows,
            "query_collision_groups": collision_groups,
            "query_collision_policy": QUERY_COLLISION_POLICY,
            "cross_topk_positives": positives,
        }
        complete &= len(observed) == expected
    progress["stages"]["supervision_extraction"] = coverage
    if complete:
        progress["status"] = "P1_SUPERVISION_COMPLETE_TRAINING_ELIGIBLE"
        progress["stages"]["training"] = "ELIGIBLE"
    else:
        progress["status"] = "P1_SUPERVISION_EXTRACTION_RUNNING"
    atomic_json(progress_path, progress)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    validation = load_validation()
    if not args.engineering_scene:
        require_smoke(validation)
    rows = selected_scenes(args)
    out = output_dir(args.engineering_scene)
    out.mkdir(parents=True, exist_ok=True)
    pending = []
    for row in rows.itertuples(index=False):
        marker = out / f"{row.scene_token}.complete.json"
        if not marker_valid(marker, validation):
            pending.append(row)
    if not pending:
        print("no pending CARE-3D P1 supervision scenes")
        if not args.engineering_scene:
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
            raise RuntimeError(f"paired P1 dataset mismatch: {protocol}")

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(CHECKPOINT), map_location="cpu")
    model = freeze_module(model.to(device))
    head = model.pts_bbox_head
    head.reset_memory()
    initial = snapshot(head)
    pc_range = head.pc_range.detach()
    if int(head.num_query) != 644 or int(head.num_propagated) != 256:
        raise RuntimeError("StreamPETR P1 query layout changed")

    for scene_row in pending:
        started = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        scene = str(scene_row.scene_token)
        split = str(scene_row.split)
        tokens = json.loads(scene_row.sample_tokens_0_12)
        main_frame, main_arrays = main_p0_source(scene, args.engineering_scene)
        expected_ids = main_frame.sample_id.astype(str).tolist()
        n = len(main_frame)
        protocol_count = len(PROTOCOLS)
        clean_query = np.zeros((n, 256), dtype=np.float16)
        fault_query = np.zeros((n, protocol_count, 256), dtype=np.float16)
        source_features = np.zeros((n, protocol_count, 3, 256), dtype=np.float16)
        source_reliability = np.zeros((n, protocol_count, 3), dtype=np.float16)
        clean_score = np.zeros((n,), dtype=np.float32)
        fault_score = np.zeros((n, protocol_count), dtype=np.float32)
        fault_topk_threshold = np.zeros((n, protocol_count), dtype=np.float32)
        target_class = np.asarray(
            [CLASSES.index(str(value)) for value in main_frame.prediction_class],
            dtype=np.int64,
        )
        target_query = main_frame.target_clean_query_index.to_numpy(dtype=np.int64)
        filled = np.zeros(n, dtype=bool)
        label_identity = True
        equivalence = []

        clean_state = initial
        anchor = None
        for frame_idx in range(3):
            index = token_index[tokens[frame_idx]]
            meta, image, data = unpack(clean_dataset[index], device)
            with torch.no_grad():
                _, _, feats = features(model, image)
                feats = feats.detach()
                pre_state = clean_state
                if frame_idx < 2:
                    output, clean_state, _ = run_head(
                        model, meta, data, feats, frame_idx > 0, pre_state
                    )
                    del output, image, data, feats
                    continue
                output, clean_state, taps = captured_head(
                    model, meta, data, feats, True, pre_state
                )
                plain_output, plain_state, _ = run_head(
                    model, meta, data, feats, True, pre_state
                )
                out_equal, out_diff = compare_outputs(output, plain_output)
                state_equal, state_diff = compare_states(clean_state, plain_state)
                if not out_equal or not state_equal:
                    raise RuntimeError(f"P1 passive capture changed B0: {scene}")
                equivalence.append({
                    "scene_token": scene,
                    "sample_token": tokens[frame_idx],
                    "frame_idx": frame_idx,
                    "output_bitwise_equal": out_equal,
                    "output_max_abs_diff": out_diff,
                    "memory_bitwise_equal": state_equal,
                    "memory_max_abs_diff": state_diff,
                })
            anchor = {"frame_idx": frame_idx, "index": index, "output": output, "taps": taps}

        if anchor is None:
            raise RuntimeError("P1 clean anchor missing")

        for target_frame_idx in range(3, 13):
            anchor_frame_idx = target_frame_idx - 1
            if anchor["frame_idx"] != anchor_frame_idx:
                raise RuntimeError("P1 clean anchor progression changed")
            row_indices = np.flatnonzero(
                main_frame.target_frame_idx.to_numpy(dtype=int) == target_frame_idx
            )
            if row_indices.size:
                assert_unique_queries(target_query[row_indices].tolist())
            target_index = token_index[tokens[target_frame_idx]]
            branch_states = clone_counterfactual_states(clean_state, 1 + protocol_count)
            if not all(states_exact(branch_states[0], state) for state in branch_states[1:]):
                raise RuntimeError("P1 counterfactual branches do not share H_t")

            clean_meta, clean_image, clean_data = unpack(clean_dataset[target_index], device)
            with torch.no_grad():
                _, _, clean_feats = features(model, clean_image)
                clean_output, clean_next_state, clean_taps = captured_head(
                    model, clean_meta, clean_data, clean_feats.detach(), True, branch_states[0]
                )
            clean_logits = clean_output["all_cls_scores"][-1, 0].detach().float().cpu().numpy()
            if row_indices.size:
                q = target_query[row_indices]
                c = target_class[row_indices]
                clean_query[row_indices] = (
                    clean_taps[TAPS[2]][torch.as_tensor(q, device=device)]
                    .detach().float().cpu().numpy().astype(np.float16)
                )
                computed_clean = 1.0 / (
                    1.0 + np.exp(-np.clip(clean_logits[q, c], -40.0, 40.0))
                )
                clean_score[row_indices] = computed_clean.astype(np.float32)
                for protocol in PROTOCOLS:
                    reference = main_frame.iloc[row_indices][
                        f"{protocol}_clean_score"
                    ].to_numpy(float)
                    label_identity &= bool(
                        np.allclose(computed_clean, reference, rtol=0.0, atol=2e-6)
                    )

            for protocol_index, protocol in enumerate(PROTOCOLS, start=1):
                fault_meta, fault_image, fault_data = unpack(
                    fault_datasets[protocol][target_index], device
                )
                with torch.no_grad():
                    _, fault_pyramid, fault_feats = features(model, fault_image)
                    fault_output, _, fault_taps = captured_head(
                        model,
                        fault_meta,
                        fault_data,
                        fault_feats.detach(),
                        True,
                        branch_states[protocol_index],
                    )
                if row_indices.size:
                    p_index = protocol_index - 1
                    q = target_query[row_indices]
                    c = target_class[row_indices]
                    q_tensor = torch.as_tensor(q, device=device, dtype=torch.long)
                    fq = fault_taps[TAPS[2]][q_tensor]
                    fault_query[row_indices, p_index] = (
                        fq.detach().float().cpu().numpy().astype(np.float16)
                    )
                    fault_logits = (
                        fault_output["all_cls_scores"][-1, 0]
                        .detach().float().cpu().numpy()
                    )
                    computed_fault = 1.0 / (
                        1.0 + np.exp(-np.clip(fault_logits[q, c], -40.0, 40.0))
                    )
                    fault_score[row_indices, p_index] = computed_fault.astype(np.float32)
                    reference = main_frame.iloc[row_indices][
                        f"{protocol}_fault_score"
                    ].to_numpy(float)
                    label_identity &= bool(
                        np.allclose(computed_fault, reference, rtol=0.0, atol=2e-6)
                    )
                    threshold = topk_score_threshold(fault_logits, 100)
                    fault_topk_threshold[row_indices, p_index] = threshold

                    fault_boxes = physical(fault_output, pc_range)[-1, 0]
                    centers = fault_boxes[q_tensor, :3].detach().float()
                    p0 = fault_pyramid[0]
                    matrices = fault_data["lidar2img"][0].detach().float()
                    camera_tokens, camera_reliability = sample_projected_camera_tokens(
                        p0,
                        matrices,
                        centers,
                        tuple(int(value) for value in fault_image.shape[-2:]),
                    )
                    temporal = torch.as_tensor(
                        main_arrays["object_features"][row_indices],
                        device=device,
                        dtype=camera_tokens.dtype,
                    )
                    sources, reliability = build_source_bank(
                        camera_tokens, camera_reliability, temporal
                    )
                    source_features[row_indices, p_index] = (
                        sources.detach().float().cpu().numpy().astype(np.float16)
                    )
                    source_reliability[row_indices, p_index] = (
                        reliability.detach().float().cpu().numpy().astype(np.float16)
                    )
                    filled[row_indices] = True
                del fault_output, fault_taps, fault_pyramid, fault_feats, fault_image, fault_data

            clean_state = clean_next_state
            anchor = {
                "frame_idx": target_frame_idx,
                "index": target_index,
                "output": clean_output,
                "taps": clean_taps,
            }
            del clean_image, clean_data, clean_feats

        if n and not filled.all():
            missing = np.flatnonzero(~filled)[:10].tolist()
            raise RuntimeError(f"P1 rows were not populated: {missing}")
        assert_exact_sample_alignment(expected_ids, main_frame.sample_id.astype(str).tolist())
        assert_source_contract(
            source_features.astype(np.float32), source_reliability.astype(np.float32)
        )
        if not label_identity:
            raise RuntimeError(f"P1 recomputed P0 score labels diverged: {scene}")

        arrays = {
            "object_features": main_arrays["object_features"].astype(np.float32, copy=False),
            "temporal_features": main_arrays["temporal_features"].astype(np.float32, copy=False),
            "decision_features": main_arrays["decision_features"].astype(np.float32, copy=False),
            "camera_support": main_arrays["camera_support"].astype(np.float32, copy=False),
            "camera_quality": main_arrays["camera_quality"].astype(np.float32, copy=False),
            "target_query": target_query,
            "target_class": target_class,
            "clean_query": clean_query,
            "fault_query": fault_query,
            "source_features": source_features,
            "source_reliability": source_reliability,
            "clean_score": clean_score,
            "fault_score": fault_score,
            "fault_topk_threshold": fault_topk_threshold,
            "cross_topk": main_arrays["cross_topk"].astype(np.int8, copy=False),
            "tp_to_fn": main_arrays["tp_to_fn"].astype(np.int8, copy=False),
            "valid_mask": main_arrays["valid_mask"].astype(np.int8, copy=False),
        }
        prefix = out / scene
        main_frame.to_csv(prefix.with_suffix(".samples.csv"), index=False)
        atomic_npz(prefix.with_suffix(".features.npz"), **arrays)
        pd.DataFrame(equivalence).to_csv(prefix.with_suffix(".equivalence.csv"), index=False)
        summary = {
            "schema_version": SCHEMA,
            "scene_manifest_sha256": validation["scene_manifest_sha256"],
            "scene_token": scene,
            "split": split,
            "query_collision_policy": QUERY_COLLISION_POLICY,
            "p0_rows_total": int(main_arrays["_p1_p0_rows_total"]),
            "rows": n,
            "query_collision_excluded_rows": int(
                main_arrays["_p1_query_collision_excluded_rows"]
            ),
            "query_collision_groups": int(main_arrays["_p1_query_collision_groups"]),
            "source_names": list(SOURCE_NAMES),
            "cross_topk_positives": {
                protocol: int(arrays["cross_topk"][:, index].sum())
                for index, protocol in enumerate(PROTOCOLS)
            },
            "equivalence_pass": all(
                bool(row["output_bitwise_equal"]) and bool(row["memory_bitwise_equal"])
                for row in equivalence
            ),
            "predictor_input_identity_pass": True,
            "label_identity_pass": bool(label_identity),
            "source_contract_pass": True,
            "peak_cuda_gib": float(
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            ),
            "elapsed_seconds": time.time() - started,
            "complete": True,
        }
        atomic_json(prefix.with_suffix(".complete.json"), summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

        progress_path = REPORT / "progress_manifest.json"
        if args.engineering_scene:
            progress = json.loads(progress_path.read_text())
            passed = all(bool(summary[key]) for key in (
                "equivalence_pass",
                "predictor_input_identity_pass",
                "label_identity_pass",
                "source_contract_pass",
            ))
            progress["stages"]["engineering_smoke"] = "PASSED" if passed else "FAILED"
            progress["stages"]["supervision_extraction"] = (
                "ELIGIBLE" if passed else "LOCKED_ENGINEERING_SMOKE_FAILED"
            )
            progress["status"] = (
                "P1_ENGINEERING_SMOKE_PASSED"
                if passed else "P1_ENGINEERING_SMOKE_FAILED"
            )
            atomic_json(progress_path, progress)
        else:
            update_progress(validation)

        if STOP_REQUESTED:
            print("stop requested; current P1 supervision scene saved", flush=True)
            break


if __name__ == "__main__":
    main()
