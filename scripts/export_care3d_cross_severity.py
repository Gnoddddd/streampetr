#!/usr/bin/env python3
"""Export CARE-3D P0 cross-severity labels on the frozen probe-test cohort.

This is a confirmatory transfer experiment.  The already-frozen CARE P0
predictor was trained using Blur/Dark severity 0.9.  This exporter changes only
the one-step t+1 intervention to severity 0.3 for Blur and Dark.  Clean-anchor
predictor inputs must match the main P0 export bit-for-bit; no predictor is
trained or calibrated here.
"""

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

from analysis.care3d_counterfactual import (  # noqa: E402
    assert_prospective_payload,
    clone_counterfactual_states,
    freeze_module,
    same_query_counterfactual,
    states_exact,
)
from analysis.care3d_cross_severity import (  # noqa: E402
    TRANSFER_PROTOCOLS,
    assert_exact_sample_alignment,
    assert_predictor_inputs_exact,
)
from scripts.audit_dark_target_recoverability import (  # noqa: E402
    features,
    physical,
    run_head,
    snapshot,
    unpack,
)
from scripts.export_care3d_counterfactual_pairs import (  # noqa: E402
    K,
    atomic_csv,
    atomic_json,
    atomic_npz,
    camera_support,
)
from scripts.run_bd_temporal_support_p0 import (  # noqa: E402
    CHECKPOINT,
    CONFIG,
    DATA,
    frame_context,
    protocol_dataset,
)
from scripts.run_prospective_failure_features import (  # noqa: E402
    CLASSES,
    TAPS,
    captured_head,
    deployed_match_map,
    frame_record,
    target_frame,
)
from scripts.run_temporal_representation_p0 import compare_outputs, compare_states  # noqa: E402


MAIN_REPORT = ROOT / "reports/care3d/p0_counterfactual_vulnerability"
REPORT = ROOT / "reports/care3d/p0_cross_severity"
PROTOCOL_PATHS = {
    "blur_s03": ROOT / "protocols/presets/motion_blur_back_10f_s03.json",
    "dark_s03": ROOT / "protocols/presets/dark_back_10f_s03.json",
}
EXPECTED_PROTOCOL_SHA256 = {
    "blur_s03": "a571b414f1466c60fc0236e10fd3beeadc8cae595fec5f87819daf2b5992ebec",
    "dark_s03": "b0409402873877059e0b8653588dddd8d766b1b43fc0e51c783758f50111555e",
}
SCHEMA = 1
STOP_REQUESTED = False
PREDICTOR_KEYS = (
    "object_features",
    "temporal_features",
    "decision_features",
    "camera_support",
    "camera_quality",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engineering-scene", action="store_true")
    parser.add_argument("--formal-test", action="store_true")
    parser.add_argument("--scene-token")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    selectors = int(args.engineering_scene) + int(args.formal_test) + int(bool(args.scene_token))
    if selectors != 1:
        parser.error("choose exactly one of --engineering-scene, --formal-test, --scene-token")
    if args.engineering_scene and args.max_scenes is not None:
        parser.error("--max-scenes is only for formal probe-test extraction")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_sources() -> dict:
    decision_path = MAIN_REPORT / "decision.json"
    progress_path = MAIN_REPORT / "progress_manifest.json"
    validation_path = MAIN_REPORT / "source_validation.json"
    manifest_path = MAIN_REPORT / "frozen_scene_manifest.csv"
    engineering_path = MAIN_REPORT / "engineering_scene_manifest.csv"
    for path in (decision_path, progress_path, validation_path, manifest_path, engineering_path):
        if not path.exists():
            raise FileNotFoundError(f"missing frozen main P0 source: {path}")

    decision = json.loads(decision_path.read_text())
    if decision.get("decision") != "GO_CARE3D_COUNTERFACTUAL_P0":
        raise RuntimeError("cross-severity transfer is locked until main CARE P0 is GO")
    if decision.get("cross_severity_status") != "ELIGIBLE_PENDING_PROTOCOL_CREATION":
        raise RuntimeError("main CARE P0 did not mark cross-severity transfer eligible")

    progress = json.loads(progress_path.read_text())
    if progress.get("status") != "GO_CARE3D_COUNTERFACTUAL_P0":
        raise RuntimeError("main CARE P0 progress is not frozen at GO")

    validation = json.loads(validation_path.read_text())
    manifest_sha = sha256(manifest_path)
    if validation.get("scene_manifest_sha256") != manifest_sha:
        raise RuntimeError("main CARE P0 scene manifest hash changed")

    protocols = {name: sha256(path) for name, path in PROTOCOL_PATHS.items()}
    if protocols != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(f"cross-severity protocol hash mismatch: {protocols}")

    checkpoint_hashes = {}
    for seed in (42, 2027, 2028):
        training_manifest = MAIN_REPORT / "training" / f"seed_{seed}" / "training_manifest.json"
        checkpoint = MAIN_REPORT / "training" / f"seed_{seed}" / "best.pth"
        if not training_manifest.exists() or not checkpoint.exists():
            raise RuntimeError(f"missing frozen CARE P0 seed {seed}")
        meta = json.loads(training_manifest.read_text())
        if meta.get("status") != "TRAINING_COMPLETE_TEST_UNSEEN" or meta.get("probe_test_read") is not False:
            raise RuntimeError(f"seed {seed} training provenance is not test-blind")
        if meta.get("routing_enabled") is not False or int(meta.get("detector_parameters_in_optimizer", -1)) != 0:
            raise RuntimeError(f"seed {seed} violates frozen P0 training invariants")
        checkpoint_hashes[str(seed)] = sha256(checkpoint)

    source = {
        "schema_version": SCHEMA,
        "status": "VALIDATED_BEFORE_CROSS_SEVERITY_FORWARD",
        "main_p0_decision": decision["decision"],
        "main_p0_decision_sha256": sha256(decision_path),
        "main_scene_manifest_sha256": manifest_sha,
        "main_source_validation_sha256": sha256(validation_path),
        "protocol_sha256": protocols,
        "p0_checkpoint_sha256": checkpoint_hashes,
        "probe_test_only": True,
        "retrain_predictor": False,
        "recalibrate_predictor": False,
        "main_train_severity": 0.9,
        "transfer_severity": 0.3,
    }

    REPORT.mkdir(parents=True, exist_ok=True)
    source_path = REPORT / "source_validation.json"
    if source_path.exists():
        previous = json.loads(source_path.read_text())
        if previous != source:
            raise RuntimeError("cross-severity frozen source identity changed")
    else:
        atomic_json(source_path, source)

    progress_out = REPORT / "progress_manifest.json"
    if not progress_out.exists():
        atomic_json(progress_out, {
            "schema_version": SCHEMA,
            "status": "CROSS_SEVERITY_ENGINEERING_SMOKE_PENDING",
            "main_scene_manifest_sha256": manifest_sha,
            "stages": {
                "engineering_smoke": "PENDING",
                "probe_test_extraction": "LOCKED_PENDING_ENGINEERING_SMOKE",
                "analysis": "LOCKED_PENDING_EXTRACTION",
            },
        })
    else:
        previous = json.loads(progress_out.read_text())
        if previous.get("main_scene_manifest_sha256") != manifest_sha:
            raise RuntimeError("cross-severity resume manifest mismatch")
    return source


def selected_rows(args) -> pd.DataFrame:
    if args.engineering_scene:
        frame = pd.read_csv(MAIN_REPORT / "engineering_scene_manifest.csv")
        if len(frame) != 1:
            raise RuntimeError("expected exactly one frozen engineering scene")
        return frame
    frame = pd.read_csv(MAIN_REPORT / "frozen_scene_manifest.csv")
    frame = frame[frame.split.astype(str) == "probe_test"].reset_index(drop=True)
    if len(frame) != 132:
        raise RuntimeError(f"expected 132 frozen probe-test scenes, got {len(frame)}")
    if args.scene_token:
        frame = frame[frame.scene_token.astype(str) == str(args.scene_token)].reset_index(drop=True)
        if len(frame) != 1:
            raise RuntimeError("requested scene-token is not in frozen probe-test")
    if args.max_scenes is not None:
        frame = frame.iloc[: int(args.max_scenes)].reset_index(drop=True)
    return frame


def output_dir(engineering: bool) -> Path:
    return REPORT / ("engineering_smoke" if engineering else "incremental/probe_test")


def marker_valid(path: Path, source: dict) -> bool:
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    return bool(
        value.get("complete")
        and value.get("schema_version") == SCHEMA
        and value.get("main_scene_manifest_sha256") == source["main_scene_manifest_sha256"]
        and value.get("protocol_sha256") == source["protocol_sha256"]
    )


def require_engineering_smoke(source: dict) -> None:
    engineering = pd.read_csv(MAIN_REPORT / "engineering_scene_manifest.csv")
    scene = str(engineering.iloc[0].scene_token)
    marker = REPORT / "engineering_smoke" / f"{scene}.complete.json"
    if not marker_valid(marker, source):
        raise RuntimeError("run cross-severity --engineering-scene before formal probe-test")
    value = json.loads(marker.read_text())
    if not value.get("equivalence_pass") or not value.get("predictor_input_identity_pass"):
        raise RuntimeError("cross-severity engineering smoke failed identity/equivalence")


def update_progress(source: dict) -> None:
    frozen = pd.read_csv(MAIN_REPORT / "frozen_scene_manifest.csv")
    expected = set(frozen[frozen.split.astype(str) == "probe_test"].scene_token.astype(str))
    observed = set()
    for marker in (REPORT / "incremental/probe_test").glob("*.complete.json"):
        if marker_valid(marker, source):
            observed.add(str(json.loads(marker.read_text())["scene_token"]))
    completed = len(expected & observed)
    progress_path = REPORT / "progress_manifest.json"
    progress = json.loads(progress_path.read_text())
    progress["stages"]["probe_test_extraction"] = {
        "completed_scenes": completed,
        "expected_scenes": len(expected),
    }
    if completed == len(expected):
        progress["status"] = "CROSS_SEVERITY_EXTRACTION_COMPLETE_ANALYSIS_PENDING"
        progress["stages"]["analysis"] = "ELIGIBLE"
    else:
        progress["status"] = "CROSS_SEVERITY_EXTRACTION_RUNNING"
    atomic_json(progress_path, progress)


def main_p0_inputs(scene: str, engineering: bool = False):
    subdir = "engineering_smoke" if engineering else "incremental/P0"
    prefix = MAIN_REPORT / subdir / scene
    samples_path = prefix.with_suffix(".samples.csv")
    feature_path = prefix.with_suffix(".features.npz")
    marker_path = prefix.with_suffix(".complete.json")
    if not samples_path.exists() or not feature_path.exists() or not marker_path.exists():
        raise RuntimeError(f"main P0 source scene is incomplete ({subdir}): {scene}")
    marker = json.loads(marker_path.read_text())
    if not marker.get("complete") or not marker.get("equivalence_pass"):
        raise RuntimeError(f"main P0 scene did not pass equivalence: {scene}")
    frame = pd.read_csv(samples_path)
    with np.load(feature_path) as packed:
        arrays = {key: np.asarray(packed[key]).copy() for key in PREDICTOR_KEYS}
    if len(frame) != len(arrays["object_features"]):
        raise RuntimeError(f"main P0 metadata/input misalignment: {scene}")
    return frame, arrays


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    source = frozen_sources()
    if not args.engineering_scene:
        require_engineering_smoke(source)

    rows = selected_rows(args)
    out = output_dir(args.engineering_scene)
    out.mkdir(parents=True, exist_ok=True)
    pending = []
    for row in rows.itertuples(index=False):
        marker = out / f"{row.scene_token}.complete.json"
        if not marker_valid(marker, source):
            pending.append(row)
    if not pending:
        print("no pending CARE-3D cross-severity scenes")
        if not args.engineering_scene:
            update_progress(source)
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
        other = {str(info["token"]): index for index, info in enumerate(dataset.data_infos)}
        if token_index != other:
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
        "target_frame_idx", "anchor_sample_token", "target_sample_token",
        "target_clean_query_index", "prediction_class", "gt_used_as_input",
    ]
    for protocol in TRANSFER_PROTOCOLS:
        metadata_fields += [
            f"{protocol}_clean_score", f"{protocol}_fault_score",
            f"{protocol}_clean_flat_rank", f"{protocol}_fault_flat_rank",
            f"{protocol}_clean_topk", f"{protocol}_fault_topk",
            f"{protocol}_evidence_drop", f"{protocol}_cross_topk", f"{protocol}_tp_to_fn",
        ]

    for scene_row in pending:
        started = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        scene = str(scene_row.scene_token)
        split = str(scene_row.split)
        tokens = json.loads(scene_row.sample_tokens_0_12)
        clean_state = initial
        anchor = None
        equivalence = []

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
                    del output, feats, image, data
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
                    raise RuntimeError(f"passive tap changed B0 output/state: {scene}")
                equivalence.append({
                    "scene_token": scene,
                    "sample_token": tokens[frame_idx],
                    "frame_idx": frame_idx,
                    "check": "passive_hooks_vs_unhooked_B0",
                    "output_bitwise_equal": out_equal,
                    "output_max_abs_diff": out_diff,
                    "memory_bitwise_equal": state_equal,
                    "memory_max_abs_diff": state_diff,
                })
            targets = target_frame(nusc, tokens[frame_idx])
            context = frame_context(clean_dataset.data_infos[index], clean_dataset)
            candidates, matches = frame_record(
                output, taps, pre_state, data, targets, context, pc_range, int(head.num_query)
            )
            anchor = {
                "frame_idx": frame_idx,
                "index": index,
                "output": output,
                "taps": taps,
                "targets": targets,
                "candidates": candidates,
                "matches": matches,
            }

        if anchor is None:
            raise RuntimeError("clean anchor was not created")

        sample_rows = []
        current_inputs = {key: [] for key in PREDICTOR_KEYS}
        drop_arrays, cross_arrays, fn_arrays, valid_arrays = [], [], [], []

        for target_frame_idx in range(3, 13):
            anchor_frame_idx = target_frame_idx - 1
            if anchor["frame_idx"] != anchor_frame_idx:
                raise RuntimeError("clean anchor progression changed")
            target_index = token_index[tokens[target_frame_idx]]
            branch_states = clone_counterfactual_states(clean_state, 1 + len(TRANSFER_PROTOCOLS))
            if not all(states_exact(branch_states[0], value) for value in branch_states[1:]):
                raise RuntimeError("cross-severity branches do not start from identical H_t")

            clean_meta, clean_image, clean_data = unpack(clean_dataset[target_index], device)
            with torch.no_grad():
                _, _, clean_feats = features(model, clean_image)
                clean_feats = clean_feats.detach()
                clean_next_output, clean_next_state, clean_next_taps = captured_head(
                    model, clean_meta, clean_data, clean_feats, True, branch_states[0]
                )
            next_targets = target_frame(nusc, tokens[target_frame_idx])
            next_context = frame_context(clean_dataset.data_infos[target_index], clean_dataset)
            clean_next_candidates, clean_next_matches = frame_record(
                clean_next_output,
                clean_next_taps,
                branch_states[0],
                clean_data,
                next_targets,
                next_context,
                pc_range,
                int(head.num_query),
            )
            clean_logits = clean_next_output["all_cls_scores"][-1, 0].detach().float().cpu().numpy()

            fault_outputs, fault_matches = {}, {}
            for protocol_index, protocol in enumerate(TRANSFER_PROTOCOLS, start=1):
                fault_meta, fault_image, fault_data = unpack(
                    fault_datasets[protocol][target_index], device
                )
                with torch.no_grad():
                    _, _, fault_feats = features(model, fault_image)
                    fault_output, _, _ = run_head(
                        model,
                        fault_meta,
                        fault_data,
                        fault_feats.detach(),
                        True,
                        branch_states[protocol_index],
                    )
                fault_outputs[protocol] = fault_output
                fault_logits = fault_output["all_cls_scores"][-1, 0].detach().float().cpu().numpy()
                fault_boxes = physical(fault_output, pc_range)[-1, 0].detach().float().cpu().numpy()
                fault_matches[protocol] = deployed_match_map(
                    fault_logits, fault_boxes, next_targets, next_context
                )
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
                        clean_dataset.data_infos[anchor["index"]],
                        anchor_candidate["prediction"]["box"][:3],
                    ),
                    "camera_quality": np.ones(6, dtype=np.float32),
                }
                assert_prospective_payload(predictor_payload)

                row = {
                    "sample_id": f"{scene}:{instance_token}:{anchor_frame_idx}:{target_frame_idx}",
                    "split": split,
                    "scene_token": scene,
                    "instance_token": instance_token,
                    "anchor_frame_idx": anchor_frame_idx,
                    "target_frame_idx": target_frame_idx,
                    "anchor_sample_token": tokens[anchor_frame_idx],
                    "target_sample_token": tokens[target_frame_idx],
                    "target_clean_query_index": query,
                    "prediction_class": CLASSES[label],
                    "gt_used_as_input": False,
                }
                drops, crosses, fns, valids = [], [], [], []
                for protocol in TRANSFER_PROTOCOLS:
                    fault_logits = (
                        fault_outputs[protocol]["all_cls_scores"][-1, 0]
                        .detach().float().cpu().numpy()
                    )
                    values = same_query_counterfactual(clean_logits, fault_logits, query, label, K)
                    tp_to_fn = int(str(next_target["token"]) not in fault_matches[protocol])
                    row.update({
                        f"{protocol}_clean_score": values["clean_score"],
                        f"{protocol}_fault_score": values["fault_score"],
                        f"{protocol}_clean_flat_rank": values["clean_flat_rank"],
                        f"{protocol}_fault_flat_rank": values["fault_flat_rank"],
                        f"{protocol}_clean_topk": int(values["clean_topk"]),
                        f"{protocol}_fault_topk": int(values["fault_topk"]),
                        f"{protocol}_evidence_drop": values["evidence_drop"],
                        f"{protocol}_cross_topk": int(values["cross_topk"]),
                        f"{protocol}_tp_to_fn": tp_to_fn,
                    })
                    drops.append(float(values["evidence_drop"]))
                    crosses.append(int(values["cross_topk"]))
                    fns.append(tp_to_fn)
                    valids.append(1)

                sample_rows.append(row)
                for key in PREDICTOR_KEYS:
                    current_inputs[key].append(predictor_payload[key])
                drop_arrays.append(drops)
                cross_arrays.append(crosses)
                fn_arrays.append(fns)
                valid_arrays.append(valids)

            clean_state = clean_next_state
            anchor = {
                "frame_idx": target_frame_idx,
                "index": target_index,
                "output": clean_next_output,
                "taps": clean_next_taps,
                "targets": next_targets,
                "candidates": clean_next_candidates,
                "matches": clean_next_matches,
            }
            fault_outputs.clear()

        n = len(sample_rows)
        shaped_inputs = {
            "object_features": np.stack(current_inputs["object_features"]).astype(np.float32)
            if n else np.empty((0, 256), np.float32),
            "temporal_features": np.stack(current_inputs["temporal_features"]).astype(np.float32)
            if n else np.empty((0, 256), np.float32),
            "decision_features": np.stack(current_inputs["decision_features"]).astype(np.float32)
            if n else np.empty((0, 21), np.float32),
            "camera_support": np.stack(current_inputs["camera_support"]).astype(np.float32)
            if n else np.empty((0, 6), np.float32),
            "camera_quality": np.stack(current_inputs["camera_quality"]).astype(np.float32)
            if n else np.empty((0, 6), np.float32),
        }
        main_frame, main_inputs = main_p0_inputs(
            scene,
            engineering=args.engineering_scene,
        )
        assert_exact_sample_alignment(
            main_frame.sample_id.astype(str).tolist(),
            [str(row["sample_id"]) for row in sample_rows],
        )
        assert_predictor_inputs_exact(main_inputs, shaped_inputs)

        labels = {
            "evidence_drop": np.asarray(drop_arrays, np.float32).reshape(n, 2),
            "cross_topk": np.asarray(cross_arrays, np.int8).reshape(n, 2),
            "tp_to_fn": np.asarray(fn_arrays, np.int8).reshape(n, 2),
            "valid_mask": np.asarray(valid_arrays, np.int8).reshape(n, 2),
        }
        prefix = out / scene
        atomic_csv(prefix.with_suffix(".samples.csv"), sample_rows, metadata_fields)
        atomic_csv(
            prefix.with_suffix(".equivalence.csv"),
            equivalence,
            [
                "scene_token", "sample_token", "frame_idx", "check",
                "output_bitwise_equal", "output_max_abs_diff",
                "memory_bitwise_equal", "memory_max_abs_diff",
            ],
        )
        atomic_npz(prefix.with_suffix(".labels.npz"), **labels)

        elapsed = time.time() - started
        summary = {
            "schema_version": SCHEMA,
            "main_scene_manifest_sha256": source["main_scene_manifest_sha256"],
            "protocol_sha256": source["protocol_sha256"],
            "scene_token": scene,
            "split": split,
            "rows": n,
            "drop_mean": {
                p: float(labels["evidence_drop"][:, i].mean()) if n else 0.0
                for i, p in enumerate(TRANSFER_PROTOCOLS)
            },
            "cross_topk_positives": {
                p: int(labels["cross_topk"][:, i].sum())
                for i, p in enumerate(TRANSFER_PROTOCOLS)
            },
            "tp_to_fn_positives": {
                p: int(labels["tp_to_fn"][:, i].sum())
                for i, p in enumerate(TRANSFER_PROTOCOLS)
            },
            "equivalence_pass": all(
                bool(row["output_bitwise_equal"]) and bool(row["memory_bitwise_equal"])
                for row in equivalence
            ),
            "predictor_input_identity_pass": True,
            "peak_cuda_gib": float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)),
            "elapsed_seconds": elapsed,
            "complete": True,
        }
        atomic_json(prefix.with_suffix(".complete.json"), summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

        progress_path = REPORT / "progress_manifest.json"
        if args.engineering_scene:
            progress = json.loads(progress_path.read_text())
            passed = bool(summary["equivalence_pass"] and summary["predictor_input_identity_pass"])
            progress["stages"]["engineering_smoke"] = "PASSED" if passed else "FAILED"
            progress["stages"]["probe_test_extraction"] = (
                "ELIGIBLE" if passed else "LOCKED_ENGINEERING_SMOKE_FAILED"
            )
            progress["status"] = (
                "CROSS_SEVERITY_ENGINEERING_SMOKE_PASSED"
                if passed else "CROSS_SEVERITY_ENGINEERING_SMOKE_FAILED"
            )
            atomic_json(progress_path, progress)
        else:
            update_progress(source)

        if STOP_REQUESTED:
            print("stop requested; current cross-severity scene saved", flush=True)
            break


if __name__ == "__main__":
    main()
