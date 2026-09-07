#!/usr/bin/env python3
"""Resume-safe CARE-3D P2-A0 online query-association extraction.

Probe-train evaluates the frozen 15-configuration grid for train-only selection.
Probe-val is inaccessible until that unique global configuration is frozen.
Probe-test is intentionally absent from this entrypoint.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import defaultdict
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
    clone_counterfactual_states,
    freeze_module,
    states_exact,
)
from analysis.care3d_p2a_association import (  # noqa: E402
    MAX_GEOMETRY_DISTANCE_M,
    P2A_QUERY_COLLISION_POLICY,
    PROTOCOLS,
    AssociationConfig,
    association_cost_components,
    association_grid,
    assert_query_layout,
    assignment_rows,
    baseline_configs,
    filter_p2a_rows,
    hungarian_with_unmatched,
    oracle_diagnostics,
    transform_lidar_centers_between_frames,
    weighted_cost,
)
from scripts.audit_dark_target_recoverability import (  # noqa: E402
    features,
    physical,
    run_head,
    snapshot,
    unpack,
)
from scripts.export_care3d_p1_supervision import main_p0_source  # noqa: E402
from scripts.run_bd_temporal_support_p0 import (  # noqa: E402
    CHECKPOINT,
    CONFIG,
    DATA,
    frame_context,
    protocol_dataset,
)
from scripts.run_prospective_failure_features import (  # noqa: E402
    TAPS,
    captured_head,
    frame_record,
    target_frame,
)
from scripts.run_temporal_representation_p0 import compare_outputs, compare_states  # noqa: E402


REPORT = ROOT / "reports/care3d/p2a_online_query_association"
PROTOCOL_PATHS = {
    "blur_back": ROOT / "protocols/presets/motion_blur_back_10f_s09.json",
    "crash_back": ROOT / "protocols/presets/camera_crash_back_10f.json",
    "dark_back": ROOT / "protocols/presets/dark_back_10f_s09.json",
}
SCHEMA = 1
STOP_REQUESTED = False

VAL_COLUMNS = [
    "sample_id", "split", "scene_token", "instance_token", "protocol", "method",
    "config_id", "geo_weight", "embedding_weight", "class_weight", "max_cost",
    "anchor_frame_idx", "target_frame_idx", "anchor_query_index", "oracle_query_index",
    "anchor_prediction_class", "selected_query", "selected_cost", "exact_match",
    "wrong_match", "unmatched", "oracle_cost", "oracle_rank",
    "correct_vs_best_wrong_margin", "oracle_geometry_eligible",
    "gt_used_as_association_input", "clean_future_used_as_association_input",
    "oracle_query_used_as_association_input",
]
TRAIN_SUMMARY_COLUMNS = [
    "scene_token", "split", "protocol", "config_id", "geo_weight",
    "embedding_weight", "class_weight", "max_cost", "rows", "exact_n", "wrong_n",
    "unmatched_n", "accepted_n", "accepted_cost_sum",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engineering-scene", action="store_true")
    parser.add_argument("--split", choices=("probe_train", "probe_val"))
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if int(args.engineering_scene) + int(args.split is not None) != 1:
        parser.error("choose exactly one of --engineering-scene or --split")
    if args.engineering_scene and args.max_scenes is not None:
        parser.error("--max-scenes is not valid for engineering smoke")
    return args


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_frame(path: Path, frame: pd.DataFrame, columns) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if frame.empty:
        frame = pd.DataFrame(columns=list(columns))
    else:
        missing = [name for name in columns if name not in frame.columns]
        if missing:
            raise RuntimeError(f"P2-A output missing columns {missing}")
        frame = frame.loc[:, list(columns)]
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def load_validation() -> dict:
    path = REPORT / "source_validation.json"
    if not path.exists():
        raise RuntimeError("run scripts/prepare_care3d_p2a.py first")
    value = json.loads(path.read_text())
    if value.get("status") != "VALIDATED_BEFORE_P2A_FORWARD":
        raise RuntimeError("P2-A source validation is not frozen")
    if value.get("probe_test_read") is not False or value.get("probe_test_locked") is not True:
        raise RuntimeError("P2-A probe-test lock changed")
    return value


def selected_scenes(args) -> pd.DataFrame:
    if args.engineering_scene:
        frame = pd.read_csv(REPORT / "engineering_scene_manifest.csv")
        if len(frame) != 1:
            raise RuntimeError("expected exactly one P2-A engineering scene")
        return frame
    frame = pd.read_csv(REPORT / "frozen_scene_manifest.csv")
    frame = frame[frame.split.astype(str) == str(args.split)].reset_index(drop=True)
    expected = 419 if args.split == "probe_train" else 133
    if len(frame) != expected:
        raise RuntimeError(f"P2-A {args.split} scene count changed: {len(frame)}")
    if args.max_scenes is not None:
        frame = frame.iloc[: int(args.max_scenes)].reset_index(drop=True)
    return frame


def output_dir(args) -> Path:
    if args.engineering_scene:
        return REPORT / "engineering_smoke"
    return REPORT / "incremental" / str(args.split)


def marker_valid(path: Path, validation: dict, expected_split: str) -> bool:
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    return bool(
        value.get("complete")
        and value.get("schema_version") == SCHEMA
        and value.get("scene_manifest_sha256") == validation["scene_manifest_sha256"]
        and value.get("split") == expected_split
        and value.get("query_collision_policy") == P2A_QUERY_COLLISION_POLICY
        and value.get("probe_test_read") is False
    )


def require_smoke(validation: dict) -> None:
    engineering = pd.read_csv(REPORT / "engineering_scene_manifest.csv")
    scene = str(engineering.iloc[0].scene_token)
    marker = REPORT / "engineering_smoke" / f"{scene}.complete.json"
    if not marker_valid(marker, validation, "engineering_smoke"):
        raise RuntimeError("P2-A engineering smoke has not completed")
    value = json.loads(marker.read_text())
    required = (
        "equivalence_pass",
        "branch_state_pass",
        "query_layout_pass",
        "association_input_contract_pass",
        "hungarian_contract_pass",
    )
    if not all(bool(value.get(key)) for key in required):
        raise RuntimeError(f"P2-A engineering smoke invariants failed: {value}")


def load_selection() -> dict:
    path = REPORT / "selection.json"
    if not path.exists():
        raise RuntimeError("probe-val remains locked until P2-A train selection is frozen")
    value = json.loads(path.read_text())
    if value.get("status") != "P2A_GLOBAL_ASSOCIATION_CONFIG_FROZEN":
        raise RuntimeError("P2-A selected configuration is not frozen")
    selected = value.get("selected", {})
    ids = {config.config_id for config in association_grid()}
    if selected.get("config_id") not in ids:
        raise RuntimeError("P2-A selected configuration is outside the preregistered grid")
    if value.get("probe_test_read") is not False:
        raise RuntimeError("P2-A selection read probe-test")
    return value


def selected_config(selection: dict) -> AssociationConfig:
    value = selection["selected"]
    return AssociationConfig(
        float(value["geo_weight"]),
        float(value["embedding_weight"]),
        float(value["class_weight"]),
        float(value["max_cost"]),
    )


def method_configs(args, selection):
    if args.engineering_scene:
        return (("smoke_reference", AssociationConfig(0.5, 0.3, 0.2, 0.45)),)
    if args.split == "probe_train":
        return tuple((config.config_id, config) for config in association_grid())
    full = selected_config(selection)
    return (("selected_full", full),) + baseline_configs(full.max_cost)


def update_progress(validation: dict) -> None:
    manifest = pd.read_csv(REPORT / "frozen_scene_manifest.csv")
    progress_path = REPORT / "progress_manifest.json"
    progress = json.loads(progress_path.read_text())
    if progress.get("probe_test_read") is not False:
        raise RuntimeError("P2-A progress indicates probe-test leakage")
    coverage = {}
    for split, expected in (("probe_train", 419), ("probe_val", 133)):
        wanted = set(manifest[manifest.split.astype(str) == split].scene_token.astype(str))
        observed = set()
        eligible_rows = 0
        excluded_rows = 0
        directory = REPORT / "incremental" / split
        for marker in directory.glob("*.complete.json") if directory.exists() else []:
            if not marker_valid(marker, validation, split):
                continue
            value = json.loads(marker.read_text())
            scene = str(value["scene_token"])
            if scene in wanted:
                observed.add(scene)
                eligible_rows += int(value.get("eligible_rows", 0))
                excluded_rows += int(value.get("p2a_collision_excluded_rows", 0))
        coverage[split] = {
            "completed_scenes": len(observed),
            "expected_scenes": expected,
            "eligible_rows": eligible_rows,
            "p2a_collision_excluded_rows": excluded_rows,
        }
    progress["stages"]["probe_train_extraction"] = coverage["probe_train"]
    progress["stages"]["probe_val_extraction"] = coverage["probe_val"]
    if coverage["probe_val"]["completed_scenes"] == 133:
        progress["status"] = "P2A_VAL_EXTRACTION_COMPLETE_ANALYSIS_ELIGIBLE"
        progress["stages"]["probe_val_analysis"] = "ELIGIBLE"
    elif coverage["probe_train"]["completed_scenes"] == 419:
        if (REPORT / "selection.json").exists():
            progress["status"] = "P2A_CONFIG_FROZEN_VAL_EXTRACTION_ELIGIBLE"
            progress["stages"]["config_selection"] = "COMPLETE"
            progress["stages"]["probe_val_extraction"] = coverage["probe_val"]
        else:
            progress["status"] = "P2A_TRAIN_EXTRACTION_COMPLETE_SELECTION_ELIGIBLE"
            progress["stages"]["config_selection"] = "ELIGIBLE"
    else:
        progress["status"] = "P2A_TRAIN_EXTRACTION_RUNNING"
    atomic_json(progress_path, progress)


def _scene_train_accumulator():
    return defaultdict(lambda: {
        "rows": 0,
        "exact_n": 0,
        "wrong_n": 0,
        "unmatched_n": 0,
        "accepted_n": 0,
        "accepted_cost_sum": 0.0,
    })


def _accumulate_train(store, protocol: str, config: AssociationConfig, outcomes) -> None:
    key = (protocol, config.config_id)
    row = store[key]
    row["rows"] += int(len(outcomes["exact_match"]))
    row["exact_n"] += int(np.asarray(outcomes["exact_match"], int).sum())
    row["wrong_n"] += int(np.asarray(outcomes["wrong_match"], int).sum())
    row["unmatched_n"] += int(np.asarray(outcomes["unmatched"], int).sum())
    selected_cost = np.asarray(outcomes["selected_cost"], float)
    accepted = np.isfinite(selected_cost)
    row["accepted_n"] += int(accepted.sum())
    row["accepted_cost_sum"] += float(np.nansum(selected_cost))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    validation = load_validation()
    selection = None
    if not args.engineering_scene:
        require_smoke(validation)
        if args.split == "probe_val":
            selection = load_selection()
            progress = json.loads((REPORT / "progress_manifest.json").read_text())
            train_stage = progress.get("stages", {}).get("probe_train_extraction", {})
            if int(train_stage.get("completed_scenes", -1)) != 419:
                raise RuntimeError("probe-val locked until all 419 probe-train scenes complete")

    rows = selected_scenes(args)
    out = output_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    expected_split = "engineering_smoke" if args.engineering_scene else str(args.split)
    pending = []
    for row in rows.itertuples(index=False):
        marker = out / f"{row.scene_token}.complete.json"
        if not marker_valid(marker, validation, expected_split):
            pending.append(row)
    if not pending:
        print("no pending CARE-3D P2-A scenes")
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
            raise RuntimeError(f"P2-A paired dataset mismatch: {protocol}")

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(CHECKPOINT), map_location="cpu")
    model = freeze_module(model.to(device))
    head = model.pts_bbox_head
    head.reset_memory()
    initial = snapshot(head)
    pc_range = head.pc_range.detach()
    assert_query_layout(int(head.num_query), int(head.num_propagated), int(head.num_query + head.num_propagated))
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(DATA), verbose=False)
    methods = method_configs(args, selection)

    for scene_row in pending:
        started = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        scene = str(scene_row.scene_token)
        split = expected_split
        tokens = json.loads(scene_row.sample_tokens_0_12)
        p1_frame, p1_arrays = main_p0_source(scene, args.engineering_scene)
        main_frame, main_arrays, p2a_audit = filter_p2a_rows(p1_frame, p1_arrays)
        eligible_rows = len(main_frame)

        clean_state = initial
        anchor = None
        equivalence_pass = True
        branch_state_pass = True
        query_layout_pass = True
        association_input_contract_pass = True
        hungarian_contract_pass = True
        val_rows = []
        train_store = _scene_train_accumulator()

        # Warm clean frames 0..2; frame 2 becomes the first current track anchor.
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
                if args.engineering_scene:
                    plain_output, plain_state, _ = run_head(
                        model, meta, data, feats, True, pre_state
                    )
                    out_equal, _ = compare_outputs(output, plain_output)
                    state_equal, _ = compare_states(clean_state, plain_state)
                    equivalence_pass &= bool(out_equal and state_equal)
            targets = target_frame(nusc, tokens[frame_idx])
            context = frame_context(clean_dataset.data_infos[index], clean_dataset)
            candidates, matches = frame_record(
                output, taps, pre_state, data, targets, context, pc_range, int(head.num_query)
            )
            anchor = {
                "frame_idx": frame_idx,
                "index": index,
                "data": data,
                "output": output,
                "taps": taps,
                "targets": targets,
                "candidates": candidates,
                "matches": matches,
                "context": context,
            }

        if anchor is None:
            raise RuntimeError("P2-A clean anchor missing")

        for target_frame_idx in range(3, 13):
            if anchor["frame_idx"] != target_frame_idx - 1:
                raise RuntimeError("P2-A clean anchor progression changed")
            row_indices = np.flatnonzero(
                main_frame.target_frame_idx.to_numpy(dtype=int) == target_frame_idx
            )
            target_index = token_index[tokens[target_frame_idx]]
            branch_states = clone_counterfactual_states(clean_state, 1 + len(PROTOCOLS))
            same_state = all(states_exact(branch_states[0], state) for state in branch_states[1:])
            branch_state_pass &= bool(same_state)
            if not same_state:
                raise RuntimeError("P2-A counterfactual branches do not share H_t")

            # Run the clean continuation only to advance the future current-track
            # state.  Nothing from this output is passed to association below.
            clean_meta, clean_image, clean_data = unpack(clean_dataset[target_index], device)
            with torch.no_grad():
                _, _, clean_feats = features(model, clean_image)
                clean_output, clean_next_state, clean_taps = captured_head(
                    model,
                    clean_meta,
                    clean_data,
                    clean_feats.detach(),
                    True,
                    branch_states[0],
                )
            next_targets = target_frame(nusc, tokens[target_frame_idx])
            next_context = frame_context(clean_dataset.data_infos[target_index], clean_dataset)
            clean_next_candidates, clean_next_matches = frame_record(
                clean_output,
                clean_taps,
                branch_states[0],
                clean_data,
                next_targets,
                next_context,
                pc_range,
                int(head.num_query),
            )

            if row_indices.size:
                frame_rows = main_frame.iloc[row_indices].reset_index(drop=True)
                anchor_features = []
                anchor_centers = []
                anchor_classes = []
                anchor_queries = []
                oracle_queries = frame_rows.target_clean_query_index.to_numpy(dtype=np.int64)
                instances = frame_rows.instance_token.astype(str).tolist()
                for local, instance_token in enumerate(instances):
                    if instance_token not in anchor["candidates"]:
                        raise RuntimeError(
                            f"P2-A current track missing from reconstructed anchor: {scene} {instance_token}"
                        )
                    candidate = anchor["candidates"][instance_token]
                    query = int(candidate["prediction"]["query"])
                    expected_query = int(frame_rows.iloc[local].anchor_query_index)
                    if query != expected_query:
                        raise RuntimeError(
                            f"P2-A anchor query identity changed: {query} != {expected_query}"
                        )
                    anchor_queries.append(query)
                    anchor_features.append(anchor["taps"][TAPS[2]][query].detach().float())
                    anchor_centers.append(np.asarray(candidate["prediction"]["box"][:3], np.float32))
                    anchor_classes.append(int(candidate["prediction"]["label"]))
                if len(anchor_queries) != len(set(anchor_queries)):
                    raise RuntimeError("P2-A shared anchor query survived eligibility filtering")
                if len(oracle_queries) != len(set(oracle_queries.tolist())):
                    raise RuntimeError("P2-A shared oracle query survived eligibility filtering")
                centers_target = transform_lidar_centers_between_frames(
                    np.stack(anchor_centers), anchor["context"], next_context
                )
                anchor_feature_tensor = torch.stack(anchor_features, dim=0).to(device=device)
                anchor_center_tensor = torch.as_tensor(
                    centers_target, device=device, dtype=torch.float32
                )
                anchor_class_tensor = torch.as_tensor(
                    anchor_classes, device=device, dtype=torch.long
                )
            else:
                frame_rows = None
                oracle_queries = np.empty((0,), dtype=np.int64)
                anchor_queries = []
                instances = []
                anchor_feature_tensor = torch.empty((0, 256), device=device)
                anchor_center_tensor = torch.empty((0, 3), device=device)
                anchor_class_tensor = torch.empty((0,), device=device, dtype=torch.long)

            for protocol_index, protocol in enumerate(PROTOCOLS, start=1):
                fault_meta, fault_image, fault_data = unpack(
                    fault_datasets[protocol][target_index], device
                )
                with torch.no_grad():
                    _, _, fault_feats = features(model, fault_image)
                    fault_output, _, fault_taps = captured_head(
                        model,
                        fault_meta,
                        fault_data,
                        fault_feats.detach(),
                        True,
                        branch_states[protocol_index],
                    )
                fault_queries = fault_taps[TAPS[2]].detach().float()
                fault_logits = fault_output["all_cls_scores"][-1, 0].detach().float()
                fault_boxes = physical(fault_output, pc_range)[-1, 0].detach().float()
                assert_query_layout(
                    int(head.num_query), int(head.num_propagated), int(fault_queries.shape[0])
                )
                query_layout_pass &= bool(fault_queries.shape == (900, 256))

                if row_indices.size:
                    components = association_cost_components(
                        anchor_feature_tensor,
                        anchor_center_tensor,
                        anchor_class_tensor,
                        fault_queries,
                        fault_logits,
                        fault_boxes[:, :3],
                        max_geometry_distance_m=MAX_GEOMETRY_DISTANCE_M,
                    )
                    association_input_contract_pass &= True
                    for method_name, config in methods:
                        cost = weighted_cost(components, config)
                        assignment = hungarian_with_unmatched(cost, config.max_cost)
                        diagnostics = oracle_diagnostics(cost, oracle_queries)
                        outcomes = assignment_rows(assignment, oracle_queries, diagnostics)
                        real = outcomes["selected_query"][outcomes["selected_query"] >= 0]
                        unique_real = len(real) == len(set(real.tolist()))
                        exhaustive = np.all(
                            outcomes["exact_match"].astype(int)
                            + outcomes["wrong_match"].astype(int)
                            + outcomes["unmatched"].astype(int)
                            == 1
                        )
                        hungarian_contract_pass &= bool(unique_real and exhaustive)

                        if args.split == "probe_train":
                            _accumulate_train(train_store, protocol, config, outcomes)
                        else:
                            for local in range(len(oracle_queries)):
                                val_rows.append({
                                    "sample_id": str(frame_rows.iloc[local].sample_id),
                                    "split": split,
                                    "scene_token": scene,
                                    "instance_token": instances[local],
                                    "protocol": protocol,
                                    "method": method_name,
                                    "config_id": config.config_id,
                                    "geo_weight": config.geo_weight,
                                    "embedding_weight": config.embedding_weight,
                                    "class_weight": config.class_weight,
                                    "max_cost": config.max_cost,
                                    "anchor_frame_idx": int(frame_rows.iloc[local].anchor_frame_idx),
                                    "target_frame_idx": target_frame_idx,
                                    "anchor_query_index": int(anchor_queries[local]),
                                    "oracle_query_index": int(oracle_queries[local]),
                                    "anchor_prediction_class": int(anchor_classes[local]),
                                    "selected_query": int(outcomes["selected_query"][local]),
                                    "selected_cost": float(outcomes["selected_cost"][local]),
                                    "exact_match": int(outcomes["exact_match"][local]),
                                    "wrong_match": int(outcomes["wrong_match"][local]),
                                    "unmatched": int(outcomes["unmatched"][local]),
                                    "oracle_cost": float(outcomes["oracle_cost"][local]),
                                    "oracle_rank": int(outcomes["oracle_rank"][local]),
                                    "correct_vs_best_wrong_margin": float(
                                        outcomes["correct_vs_best_wrong_margin"][local]
                                    ),
                                    "oracle_geometry_eligible": int(
                                        outcomes["oracle_geometry_eligible"][local]
                                    ),
                                    "gt_used_as_association_input": False,
                                    "clean_future_used_as_association_input": False,
                                    "oracle_query_used_as_association_input": False,
                                })
                del fault_output, fault_taps, fault_feats, fault_image, fault_data

            clean_state = clean_next_state
            anchor = {
                "frame_idx": target_frame_idx,
                "index": target_index,
                "data": clean_data,
                "output": clean_output,
                "taps": clean_taps,
                "targets": next_targets,
                "candidates": clean_next_candidates,
                "matches": clean_next_matches,
                "context": next_context,
            }
            del clean_image, clean_data, clean_feats

        prefix = out / scene
        if args.split == "probe_train":
            summary_rows = []
            by_id = {config.config_id: config for config in association_grid()}
            for (protocol, config_id), counts in sorted(train_store.items()):
                config = by_id[config_id]
                summary_rows.append({
                    "scene_token": scene,
                    "split": split,
                    "protocol": protocol,
                    **config.as_dict(),
                    **counts,
                })
            summary_frame = pd.DataFrame(summary_rows)
            atomic_frame(prefix.with_suffix(".train_summary.csv"), summary_frame, TRAIN_SUMMARY_COLUMNS)
            method_rows = int(sum(int(row["rows"]) for row in summary_rows))
        else:
            value_frame = pd.DataFrame(val_rows)
            atomic_frame(prefix.with_suffix(".rows.csv"), value_frame, VAL_COLUMNS)
            method_rows = int(len(value_frame))

        summary = {
            "schema_version": SCHEMA,
            "scene_manifest_sha256": validation["scene_manifest_sha256"],
            "scene_token": scene,
            "split": split,
            "p1_eligible_rows": int(p2a_audit["p1_rows_total"]),
            "eligible_rows": int(eligible_rows),
            "p2a_collision_excluded_rows": int(p2a_audit["total_excluded_rows"]),
            "anchor_query_collision_excluded_rows": int(
                p2a_audit["anchor_query_collision_excluded_rows"]
            ),
            "anchor_query_collision_groups": int(p2a_audit["anchor_query_collision_groups"]),
            "target_query_collision_excluded_rows": int(
                p2a_audit["target_query_collision_excluded_rows"]
            ),
            "target_query_collision_groups": int(p2a_audit["target_query_collision_groups"]),
            "query_collision_policy": P2A_QUERY_COLLISION_POLICY,
            "method_rows": method_rows,
            "methods": [name for name, _ in methods],
            "equivalence_pass": bool(equivalence_pass),
            "branch_state_pass": bool(branch_state_pass),
            "query_layout_pass": bool(query_layout_pass),
            "association_input_contract_pass": bool(association_input_contract_pass),
            "hungarian_contract_pass": bool(hungarian_contract_pass),
            "gt_used_as_association_input": False,
            "clean_future_used_as_association_input": False,
            "oracle_query_used_as_association_input": False,
            "probe_test_read": False,
            "peak_cuda_gib": float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)),
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
                "branch_state_pass",
                "query_layout_pass",
                "association_input_contract_pass",
                "hungarian_contract_pass",
            ))
            progress["stages"]["engineering_smoke"] = "PASSED" if passed else "FAILED"
            progress["stages"]["probe_train_extraction"] = (
                "ELIGIBLE" if passed else "LOCKED_ENGINEERING_SMOKE_FAILED"
            )
            progress["status"] = (
                "P2A_ENGINEERING_SMOKE_PASSED"
                if passed else "P2A_ENGINEERING_SMOKE_FAILED"
            )
            progress["probe_test_read"] = False
            atomic_json(progress_path, progress)
        else:
            update_progress(validation)

        if STOP_REQUESTED:
            print("stop requested; current P2-A scene saved", flush=True)
            break


if __name__ == "__main__":
    main()
