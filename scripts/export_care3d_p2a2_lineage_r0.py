#!/usr/bin/env python3
"""Extract CARE-3D P2-A2-R0 explicit one-step memory-lineage assignments."""

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
from analysis.care3d_p2a2_lineage import (  # noqa: E402
    BASELINE_METHODS,
    FROZEN_P2A0_CONFIG,
    PRIMARY_METHOD,
    evaluate_assignments,
    lineage_first_assign,
    query_origin,
    verify_post_update_memory,
)
from analysis.care3d_p2a_association import (  # noqa: E402
    MAX_GEOMETRY_DISTANCE_M,
    PROTOCOLS,
    association_cost_components,
    assert_query_layout,
    filter_p2a_rows,
    transform_lidar_centers_between_frames,
    weighted_cost,
)
from analysis.care3d_p2a_execution import build_shared_protocol_dataset, shard_scene_frame  # noqa: E402
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


REPORT = ROOT / "reports/care3d/p2a2_memory_lineage_r0"
PROTOCOL_PATHS = {
    "blur_back": ROOT / "protocols/presets/motion_blur_back_10f_s09.json",
    "crash_back": ROOT / "protocols/presets/camera_crash_back_10f.json",
    "dark_back": ROOT / "protocols/presets/dark_back_10f_s09.json",
}
SCHEMA = 1
STOP_REQUESTED = False
ROW_COLUMNS = [
    "scene_token", "instance_token", "anchor_frame_idx", "target_frame_idx",
    "protocol", "anchor_query_index", "anchor_query_origin",
    "anchor_topk_member", "lineage_position", "lineage_child_query",
    "lineage_available", "oracle_query_index", "oracle_query_origin",
    "lineage_exact", "lineage_wrong", "lineage_unmatched",
    "p2a0_selected_query", "p2a0_exact", "p2a0_wrong", "p2a0_unmatched",
    "hybrid_selected_query", "hybrid_source", "hybrid_exact", "hybrid_wrong",
    "hybrid_unmatched", "p2a0_selected_cost", "fallback_selected_cost",
    "gt_used_as_association_input", "clean_future_used_as_association_input",
    "oracle_query_used_as_association_input",
]


def parse_lineage_split(value: str) -> str:
    if value != "probe_train":
        raise argparse.ArgumentTypeError("P2-A2-R0 exposes only --split probe_train")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engineering-scene", action="store_true")
    parser.add_argument("--split", type=parse_lineage_split)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--defer-progress", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if int(args.engineering_scene) + int(args.split is not None) != 1:
        parser.error("choose exactly one of --engineering-scene or --split probe_train")
    if args.engineering_scene and (
        args.max_scenes is not None or args.num_shards != 1 or args.shard_index != 0
    ):
        parser.error("engineering smoke cannot be truncated or sharded")
    if args.max_scenes is not None and args.max_scenes < 0:
        parser.error("--max-scenes must be non-negative")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("shards require 0 <= shard-index < num-shards")
    return args


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if frame.empty:
        frame = pd.DataFrame(columns=ROW_COLUMNS)
    missing = [column for column in ROW_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"P2-A2-R0 output missing columns: {missing}")
    frame.loc[:, ROW_COLUMNS].to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def load_validation() -> dict:
    path = REPORT / "source_validation.json"
    if not path.exists():
        raise RuntimeError("run scripts/prepare_care3d_p2a2_lineage_r0.py first")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("probe_val_read") is not False or value.get("probe_test_read") is not False:
        raise RuntimeError("P2-A2-R0 source discipline changed")
    if value.get("selected_config", {}).get("config_id") != FROZEN_P2A0_CONFIG.config_id:
        raise RuntimeError("P2-A2-R0 frozen P2-A0 config changed")
    return value


def selected_scenes(args) -> pd.DataFrame:
    filename = "engineering_scene_manifest.csv" if args.engineering_scene else "probe_train_manifest.csv"
    frame = pd.read_csv(REPORT / filename)
    expected = 1 if args.engineering_scene else 419
    if len(frame) != expected:
        raise RuntimeError(f"P2-A2-R0 scene count changed: {len(frame)} != {expected}")
    if not args.engineering_scene:
        if set(frame.split.astype(str)) != {"probe_train"}:
            raise RuntimeError("P2-A2-R0 manifest contains a non-train split")
        frame = shard_scene_frame(
            frame,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            max_scenes=args.max_scenes,
        )
    return frame.reset_index(drop=True)


def require_engineering_smoke() -> None:
    path = REPORT / "engineering_smoke.json"
    if not path.exists():
        raise RuntimeError("P2-A2-R0 probe-train is locked pending engineering smoke")
    value = json.loads(path.read_text(encoding="utf-8"))
    required = (
        value.get("status") == "P2A2_R0_LINEAGE_SMOKE_PASSED",
        value.get("memory_lineage_torch_equal") is True,
        float(value.get("memory_lineage_max_abs_diff", float("inf"))) == 0.0,
        value.get("num_query") == 644,
        value.get("num_propagated") == 256,
        value.get("topk_proposals") == 256,
        value.get("query_count") == 900,
        value.get("probe_val_read") is False,
        value.get("probe_test_read") is False,
    )
    if not all(required):
        raise RuntimeError("P2-A2-R0 probe-train is locked by failed smoke invariants")


def marker_valid(path: Path, validation: dict, split: str) -> bool:
    if not path.exists():
        return False
    value = json.loads(path.read_text(encoding="utf-8"))
    return bool(
        value.get("complete")
        and value.get("schema_version") == SCHEMA
        and value.get("split") == split
        and value.get("p2a0_selection_sha256") == validation["p2a0_selection_sha256"]
        and value.get("probe_val_read") is False
        and value.get("probe_test_read") is False
    )


def output_directory(engineering: bool) -> Path:
    return REPORT / ("engineering_smoke" if engineering else "incremental/probe_train")


def update_progress(validation: dict) -> None:
    manifest = pd.read_csv(REPORT / "probe_train_manifest.csv")
    wanted = set(manifest.scene_token.astype(str))
    observed = set()
    rows = 0
    directory = REPORT / "incremental/probe_train"
    for marker_path in directory.glob("*.complete.json") if directory.exists() else []:
        if not marker_valid(marker_path, validation, "probe_train"):
            continue
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        scene = str(marker["scene_token"])
        if scene in wanted:
            observed.add(scene)
            rows += int(marker.get("rows", 0))
    atomic_json(REPORT / "progress_manifest.json", {
        "schema_version": SCHEMA,
        "status": (
            "P2A2_R0_PROBE_TRAIN_EXTRACTION_COMPLETE_ANALYSIS_ELIGIBLE"
            if len(observed) == 419 else "P2A2_R0_PROBE_TRAIN_EXTRACTION_RUNNING"
        ),
        "completed_scenes": len(observed),
        "expected_scenes": 419,
        "rows": rows,
        "development_pre_gate": True,
        "probe_val_read": False,
        "probe_test_read": False,
    })


def _anchor_record(
    output,
    state,
    taps,
    candidates,
    context,
    frame_idx: int,
    topk_proposals: int,
) -> tuple[dict, dict]:
    memory_check = verify_post_update_memory(
        taps[TAPS[2]],
        output["all_cls_scores"][-1],
        state["memory_embedding"],
        topk_proposals=topk_proposals,
    )
    return ({
        "frame_idx": int(frame_idx),
        "output": output,
        "taps": taps,
        "candidates": candidates,
        "context": context,
        "topk_indexes": memory_check["topk_indexes"],
    }, memory_check)


def main() -> None:
    global STOP_REQUESTED
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    validation = load_validation()
    if not args.engineering_scene:
        require_engineering_smoke()
    rows = selected_scenes(args)
    out = output_directory(args.engineering_scene)
    split = "engineering_smoke" if args.engineering_scene else "probe_train"
    pending = [
        row for row in rows.itertuples(index=False)
        if not marker_valid(out / f"{row.scene_token}.complete.json", validation, split)
    ]
    if not pending:
        print("no pending CARE-3D P2-A2-R0 scenes")
        if not args.engineering_scene and not args.defer_progress:
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
        protocol: build_shared_protocol_dataset(clean_dataset, cfg, path)
        for protocol, path in PROTOCOL_PATHS.items()
    }
    token_index = {
        str(info["token"]): index for index, info in enumerate(clean_dataset.data_infos)
    }
    for protocol, dataset in fault_datasets.items():
        if dataset.data_infos is not clean_dataset.data_infos:
            raise RuntimeError(f"P2-A2-R0 {protocol} did not share immutable data_infos")

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(CHECKPOINT), map_location="cpu")
    model = freeze_module(model.to(device))
    head = model.pts_bbox_head
    assert_query_layout(
        int(head.num_query),
        int(head.num_propagated),
        int(head.num_query + head.num_propagated),
    )
    if int(head.topk_proposals) != 256:
        raise RuntimeError("StreamPETR topk_proposals changed")
    head.reset_memory()
    initial = snapshot(head)
    pc_range = head.pc_range.detach()
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(DATA), verbose=False)

    for scene_row in pending:
        started = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        scene = str(scene_row.scene_token)
        tokens = json.loads(scene_row.sample_tokens_0_12)
        p1_frame, p1_arrays = main_p0_source(scene, args.engineering_scene)
        main_frame, _, p2a_audit = filter_p2a_rows(p1_frame, p1_arrays)
        clean_state = initial
        anchor = None
        output_rows = []
        memory_equal = True
        memory_max_abs_diff = 0.0
        branch_state_pass = True
        capture_equivalence_pass = True

        for frame_idx in range(3):
            index = token_index[tokens[frame_idx]]
            meta, image, data = unpack(clean_dataset[index], device)
            with torch.no_grad():
                _, _, feats = features(model, image)
                feats = feats.detach()
                pre_state = clean_state
                if frame_idx < 2:
                    _, clean_state, _ = run_head(
                        model, meta, data, feats, frame_idx > 0, pre_state
                    )
                    del image, data, feats
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
                    capture_equivalence_pass &= bool(out_equal and state_equal)
            targets = target_frame(nusc, tokens[frame_idx])
            context = frame_context(clean_dataset.data_infos[index], clean_dataset)
            candidates, _ = frame_record(
                output, taps, pre_state, data, targets, context, pc_range, int(head.num_query)
            )
            anchor, check = _anchor_record(
                output, clean_state, taps, candidates, context, frame_idx,
                int(head.topk_proposals),
            )
            memory_equal &= bool(check["torch_equal"] or check["max_abs_diff"] == 0.0)
            memory_max_abs_diff = max(memory_max_abs_diff, float(check["max_abs_diff"]))
            del image, data, feats

        if anchor is None:
            raise RuntimeError("P2-A2-R0 clean anchor missing")

        for target_frame_idx in range(3, 13):
            if anchor["frame_idx"] != target_frame_idx - 1:
                raise RuntimeError("P2-A2-R0 clean anchor progression changed")
            row_indices = np.flatnonzero(
                main_frame.target_frame_idx.to_numpy(dtype=int) == target_frame_idx
            )
            target_index = token_index[tokens[target_frame_idx]]
            next_context = frame_context(clean_dataset.data_infos[target_index], clean_dataset)
            branch_states = clone_counterfactual_states(clean_state, 1 + len(PROTOCOLS))
            same_state = all(states_exact(branch_states[0], state) for state in branch_states[1:])
            branch_state_pass &= bool(same_state)
            if not same_state:
                raise RuntimeError("P2-A2-R0 counterfactual branches do not share H_t")

            if row_indices.size:
                frame_rows = main_frame.iloc[row_indices].reset_index(drop=True)
                anchor_features = []
                anchor_centers = []
                anchor_classes = []
                anchor_queries = []
                instances = frame_rows.instance_token.astype(str).tolist()
                for local, instance_token in enumerate(instances):
                    candidate = anchor["candidates"].get(instance_token)
                    if candidate is None:
                        raise RuntimeError(f"clean online anchor missing: {scene} {instance_token}")
                    query = int(candidate["prediction"]["query"])
                    if query != int(frame_rows.iloc[local].anchor_query_index):
                        raise RuntimeError("frozen anchor query identity changed")
                    anchor_queries.append(query)
                    anchor_features.append(anchor["taps"][TAPS[2]][query].detach().float())
                    anchor_centers.append(np.asarray(candidate["prediction"]["box"][:3], np.float32))
                    anchor_classes.append(int(candidate["prediction"]["label"]))
                if len(anchor_queries) != len(set(anchor_queries)):
                    raise RuntimeError("shared anchor query survived frozen collision policy")
                centers_target = transform_lidar_centers_between_frames(
                    np.stack(anchor_centers), anchor["context"], next_context
                )
                anchor_feature_tensor = torch.stack(anchor_features).to(device)
                anchor_center_tensor = torch.as_tensor(centers_target, device=device)
                anchor_class_tensor = torch.as_tensor(anchor_classes, device=device, dtype=torch.long)
            else:
                frame_rows = None
                anchor_queries = []
                anchor_classes = []
                instances = []
                anchor_feature_tensor = torch.empty((0, 256), device=device)
                anchor_center_tensor = torch.empty((0, 3), device=device)
                anchor_class_tensor = torch.empty((0,), device=device, dtype=torch.long)

            # Fault associations are completed before the clean t+1 forward.
            # Thus clean future outputs cannot influence any selected query.
            for protocol_index, protocol in enumerate(PROTOCOLS, start=1):
                fault_meta, fault_image, fault_data = unpack(
                    fault_datasets[protocol][target_index], device
                )
                with torch.no_grad():
                    _, _, fault_feats = features(model, fault_image)
                    fault_output, _, fault_taps = captured_head(
                        model, fault_meta, fault_data, fault_feats.detach(), True,
                        branch_states[protocol_index],
                    )
                fault_queries = fault_taps[TAPS[2]].detach().float()
                fault_logits = fault_output["all_cls_scores"][-1, 0].detach().float()
                fault_boxes = physical(fault_output, pc_range)[-1, 0].detach().float()
                assert_query_layout(
                    int(head.num_query), int(head.num_propagated), int(fault_queries.shape[0])
                )
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
                    frozen_cost = weighted_cost(components, FROZEN_P2A0_CONFIG)
                    assignments = lineage_first_assign(
                        anchor_queries, anchor["topk_indexes"], frozen_cost
                    )
                    # Oracle is introduced only after the online function returned.
                    oracle_queries = frame_rows.target_clean_query_index.to_numpy(np.int64)
                    outcomes = evaluate_assignments(assignments, oracle_queries)
                    for local in range(len(frame_rows)):
                        output_rows.append({
                            "scene_token": scene,
                            "instance_token": instances[local],
                            "anchor_frame_idx": int(frame_rows.iloc[local].anchor_frame_idx),
                            "target_frame_idx": int(target_frame_idx),
                            "protocol": protocol,
                            "anchor_query_index": int(anchor_queries[local]),
                            "anchor_query_origin": query_origin(anchor_queries[local]),
                            "anchor_topk_member": bool(assignments["anchor_topk_member"][local]),
                            "lineage_position": int(assignments["lineage_position"][local]),
                            "lineage_child_query": int(assignments["lineage_child_query"][local]),
                            "lineage_available": bool(assignments["lineage_available"][local]),
                            "oracle_query_index": int(oracle_queries[local]),
                            "oracle_query_origin": query_origin(oracle_queries[local]),
                            "lineage_exact": int(outcomes["lineage_exact"][local]),
                            "lineage_wrong": int(outcomes["lineage_wrong"][local]),
                            "lineage_unmatched": int(outcomes["lineage_unmatched"][local]),
                            "p2a0_selected_query": int(assignments["p2a0_selected_query"][local]),
                            "p2a0_exact": int(outcomes["p2a0_exact"][local]),
                            "p2a0_wrong": int(outcomes["p2a0_wrong"][local]),
                            "p2a0_unmatched": int(outcomes["p2a0_unmatched"][local]),
                            "hybrid_selected_query": int(assignments["hybrid_selected_query"][local]),
                            "hybrid_source": str(assignments["hybrid_source"][local]),
                            "hybrid_exact": int(outcomes["hybrid_exact"][local]),
                            "hybrid_wrong": int(outcomes["hybrid_wrong"][local]),
                            "hybrid_unmatched": int(outcomes["hybrid_unmatched"][local]),
                            "p2a0_selected_cost": float(assignments["p2a0_selected_cost"][local]),
                            "fallback_selected_cost": float(assignments["fallback_selected_cost"][local]),
                            "gt_used_as_association_input": False,
                            "clean_future_used_as_association_input": False,
                            "oracle_query_used_as_association_input": False,
                        })
                del fault_output, fault_taps, fault_feats, fault_image, fault_data

            if target_frame_idx < 12:
                clean_meta, clean_image, clean_data = unpack(clean_dataset[target_index], device)
                with torch.no_grad():
                    _, _, clean_feats = features(model, clean_image)
                    clean_output, clean_next_state, clean_taps = captured_head(
                        model, clean_meta, clean_data, clean_feats.detach(), True,
                        branch_states[0],
                    )
                next_targets = target_frame(nusc, tokens[target_frame_idx])
                next_candidates, _ = frame_record(
                    clean_output, clean_taps, branch_states[0], clean_data, next_targets,
                    next_context, pc_range, int(head.num_query),
                )
                anchor, check = _anchor_record(
                    clean_output, clean_next_state, clean_taps, next_candidates,
                    next_context, target_frame_idx, int(head.topk_proposals),
                )
                memory_equal &= bool(check["torch_equal"] or check["max_abs_diff"] == 0.0)
                memory_max_abs_diff = max(memory_max_abs_diff, float(check["max_abs_diff"]))
                clean_state = clean_next_state
                del clean_image, clean_data, clean_feats

        frame = pd.DataFrame(output_rows)
        prefix = out / scene
        atomic_frame(prefix.with_suffix(".rows.csv"), frame)
        sources = set(frame.hybrid_source.astype(str)) if len(frame) else set()
        if not sources <= {"lineage", "p2a0_fallback", "unmatched"}:
            raise RuntimeError(f"unexpected hybrid source(s): {sources}")
        summary = {
            "schema_version": SCHEMA,
            "p2a0_selection_sha256": validation["p2a0_selection_sha256"],
            "scene_token": scene,
            "split": split,
            "rows": int(len(frame)),
            "eligible_objects": int(len(main_frame)),
            "p2a_collision_excluded_rows": int(p2a_audit["total_excluded_rows"]),
            "primary_method": PRIMARY_METHOD,
            "baselines": list(BASELINE_METHODS),
            "frozen_config": FROZEN_P2A0_CONFIG.as_dict(),
            "num_query": int(head.num_query),
            "num_propagated": int(head.num_propagated),
            "topk_proposals": int(head.topk_proposals),
            "query_count": int(head.num_query + head.num_propagated),
            "memory_lineage_torch_equal": bool(memory_equal),
            "memory_lineage_max_abs_diff": float(memory_max_abs_diff),
            "capture_equivalence_pass": bool(capture_equivalence_pass),
            "branch_state_pass": bool(branch_state_pass),
            "gt_used_as_input": False,
            "clean_future_used_as_input": False,
            "oracle_query_used_as_input": False,
            "probe_val_read": False,
            "probe_test_read": False,
            "peak_cuda_gib": float(torch.cuda.max_memory_allocated(device) / 1024 ** 3),
            "elapsed_seconds": float(time.time() - started),
            "complete": True,
        }
        atomic_json(prefix.with_suffix(".complete.json"), summary)
        if args.engineering_scene:
            smoke_pass = all((
                summary["memory_lineage_torch_equal"],
                summary["memory_lineage_max_abs_diff"] == 0.0,
                summary["capture_equivalence_pass"],
                summary["branch_state_pass"],
                summary["num_query"] == 644,
                summary["num_propagated"] == 256,
                summary["topk_proposals"] == 256,
                summary["query_count"] == 900,
                summary["rows"] > 0,
            ))
            atomic_json(REPORT / "engineering_smoke.json", {
                **summary,
                "status": (
                    "P2A2_R0_LINEAGE_SMOKE_PASSED"
                    if smoke_pass else "P2A2_R0_LINEAGE_SMOKE_FAILED"
                ),
            })
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        if STOP_REQUESTED:
            print("stop requested; current P2-A2-R0 scene saved", flush=True)
            break

    if not args.engineering_scene and not args.defer_progress:
        update_progress(validation)


if __name__ == "__main__":
    main()
