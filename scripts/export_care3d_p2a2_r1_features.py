#!/usr/bin/env python3
"""Export CARE-3D P2-A2-R1-F0 relative candidate evidence."""

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

from analysis.care3d_counterfactual import (  # noqa: E402
    clone_counterfactual_states,
    freeze_module,
    states_exact,
)
from analysis.care3d_p2a2_lineage import (  # noqa: E402
    FROZEN_P2A0_CONFIG,
    lineage_first_assign,
    query_origin,
    recompute_topk_indexes,
)
from analysis.care3d_p2a2_r1_features import (  # noqa: E402
    EVIDENCE_FEATURE_COLUMNS,
    finite_model_matrix,
    offline_disagreement_labels,
    relative_candidate_features,
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
    frame_context,
    protocol_dataset,
)
from scripts.run_prospective_failure_features import (  # noqa: E402
    CLASSES,
    TAPS,
    captured_head,
)


REPORT = ROOT / "reports/care3d/p2a2_r1_relative_evidence"
R0_REPORT = ROOT / "reports/care3d/p2a2_memory_lineage_r0"
P2A0_REPORT = ROOT / "reports/care3d/p2a_online_query_association"
PROTOCOL_PATHS = {
    "blur_back": ROOT / "protocols/presets/motion_blur_back_10f_s09.json",
    "crash_back": ROOT / "protocols/presets/camera_crash_back_10f.json",
    "dark_back": ROOT / "protocols/presets/dark_back_10f_s09.json",
}
SCHEMA = 1
STOP_REQUESTED = False
METADATA_COLUMNS = (
    "scene_token", "instance_token", "anchor_frame_idx", "target_frame_idx",
    "protocol", "anchor_query_index", "anchor_query_origin",
    "anchor_prediction_class", "p2a0_selected_query", "lineage_child_query",
    "lineage_position", "oracle_query_index",
)
LABEL_COLUMNS = ("p2a0_wins", "lineage_wins", "both_wrong", "outcome_class")
AUDIT_COLUMNS = (
    "feature_computation_frozen_before_oracle", "gt_used_as_feature_input",
    "clean_future_used_as_feature_input", "oracle_query_used_as_feature_input",
)
ROW_COLUMNS = METADATA_COLUMNS + EVIDENCE_FEATURE_COLUMNS + LABEL_COLUMNS + AUDIT_COLUMNS


def parse_r1_split(value: str) -> str:
    if value != "probe_train":
        raise argparse.ArgumentTypeError("R1-F0 exposes only --split probe_train")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--engineering-scene", action="store_true")
    parser.add_argument("--split", type=parse_r1_split)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--defer-progress", action="store_true")
    parser.add_argument("--device", default="cuda:0")
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
        raise RuntimeError(f"R1-F0 output missing columns: {missing}")
    frame.loc[:, ROW_COLUMNS].to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def validate_sources() -> dict:
    r0_decision = json.loads((R0_REPORT / "decision.json").read_text())
    r0_progress = json.loads((R0_REPORT / "progress_manifest.json").read_text())
    r0_smoke = json.loads((R0_REPORT / "engineering_smoke.json").read_text())
    p2a0_decision = json.loads((P2A0_REPORT / "decision.json").read_text())
    required = (
        p2a0_decision.get("decision") == "NO_GO_CARE3D_P2A_ONLINE_QUERY_ASSOCIATION",
        r0_decision.get("decision") == "NO_GO_P2A2_R0_MEMORY_LINEAGE",
        r0_progress.get("completed_scenes") == 419,
        r0_progress.get("rows") == 275985,
        r0_smoke.get("status") == "P2A2_R0_LINEAGE_SMOKE_PASSED",
        r0_decision.get("probe_val_read") is False,
        r0_decision.get("probe_test_read") is False,
        r0_progress.get("probe_val_read") is False,
        r0_progress.get("probe_test_read") is False,
    )
    if not all(required):
        raise RuntimeError("R1-F0 frozen P2-A0/P2-A2-R0 prerequisites changed")
    validation = {
        "schema_version": SCHEMA,
        "status": "P2A2_R1_F0_SOURCES_VALIDATED",
        "p2a0_decision": p2a0_decision["decision"],
        "r0_decision": r0_decision["decision"],
        "r0_completed_scenes": 419,
        "r0_protocol_rows": 275985,
        "frozen_config": FROZEN_P2A0_CONFIG.as_dict(),
        "development_feature_sufficiency_audit": True,
        "final_r1_model": False,
        "probe_val_read": False,
        "probe_test_read": False,
    }
    path = REPORT / "source_validation.json"
    if path.exists():
        previous = json.loads(path.read_text())
        for key in ("p2a0_decision", "r0_decision", "r0_completed_scenes", "frozen_config"):
            if previous.get(key) != validation.get(key):
                raise RuntimeError(f"R1-F0 source validation changed: {key}")
    else:
        atomic_json(path, validation)
    return validation


def require_smoke() -> None:
    path = REPORT / "engineering_smoke.json"
    if not path.exists():
        raise RuntimeError("R1-F0 probe-train is locked pending engineering smoke")
    value = json.loads(path.read_text())
    if not all((
        value.get("status") == "P2A2_R1_F0_ENGINEERING_SMOKE_PASSED",
        value.get("r0_assignment_exact") is True,
        value.get("p2a0_frozen_cost_exact") is True,
        value.get("feature_non_mutating") is True,
        value.get("probe_val_read") is False,
        value.get("probe_test_read") is False,
    )):
        raise RuntimeError("R1-F0 probe-train is locked by failed smoke invariants")


def selected_scenes(args) -> pd.DataFrame:
    filename = "engineering_scene_manifest.csv" if args.engineering_scene else "probe_train_manifest.csv"
    frame = pd.read_csv(R0_REPORT / filename)
    expected = 1 if args.engineering_scene else 419
    if len(frame) != expected:
        raise RuntimeError("R1-F0 source scene count changed")
    if not args.engineering_scene:
        if set(frame.split.astype(str)) != {"probe_train"}:
            raise RuntimeError("R1-F0 source manifest contains held-out scenes")
        frame = shard_scene_frame(
            frame,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            max_scenes=args.max_scenes,
        )
    return frame.reset_index(drop=True)


def output_directory(engineering: bool) -> Path:
    return REPORT / ("engineering_smoke" if engineering else "incremental/probe_train")


def marker_valid(path: Path, split: str) -> bool:
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    return bool(
        value.get("complete")
        and value.get("schema_version") == SCHEMA
        and value.get("split") == split
        and value.get("probe_val_read") is False
        and value.get("probe_test_read") is False
        and value.get("r0_assignment_exact") is True
        and value.get("p2a0_frozen_cost_exact") is True
    )


def load_r0_rows(scene: str, engineering: bool) -> pd.DataFrame:
    directory = R0_REPORT / ("engineering_smoke" if engineering else "incremental/probe_train")
    # Preserve the exact binary64 value serialized by the R0 CSV writer.  The
    # default pandas parser can shorten it by ~1e-16 and create a false mismatch.
    frame = pd.read_csv(
        directory / f"{scene}.rows.csv", float_precision="round_trip"
    )
    expected_split = "engineering_smoke" if engineering else "probe_train"
    if set(frame.protocol.astype(str)) != set(PROTOCOLS):
        raise RuntimeError(f"R1-F0 R0 protocol rows changed: {scene}")
    marker = json.loads((directory / f"{scene}.complete.json").read_text())
    if marker.get("split") != expected_split or int(marker.get("rows", -1)) != len(frame):
        raise RuntimeError(f"R1-F0 invalid R0 source marker: {scene}")
    return frame


def update_progress() -> None:
    manifest = pd.read_csv(R0_REPORT / "probe_train_manifest.csv")
    wanted = set(manifest.scene_token.astype(str))
    observed = set()
    rows = 0
    directory = REPORT / "incremental/probe_train"
    for path in directory.glob("*.complete.json") if directory.exists() else []:
        if not marker_valid(path, "probe_train"):
            continue
        value = json.loads(path.read_text())
        if str(value["scene_token"]) in wanted:
            observed.add(str(value["scene_token"]))
            rows += int(value.get("disagreement_rows", 0))
    atomic_json(REPORT / "progress_manifest.json", {
        "schema_version": SCHEMA,
        "status": (
            "P2A2_R1_F0_EXTRACTION_COMPLETE_ANALYSIS_ELIGIBLE"
            if len(observed) == 419 else "P2A2_R1_F0_EXTRACTION_RUNNING"
        ),
        "completed_scenes": len(observed),
        "expected_scenes": 419,
        "disagreement_rows": rows,
        "probe_val_read": False,
        "probe_test_read": False,
    })


def _r0_group(r0_rows: pd.DataFrame, protocol: str, target_frame_idx: int) -> pd.DataFrame:
    return r0_rows[
        (r0_rows.protocol.astype(str) == protocol)
        & (r0_rows.target_frame_idx.astype(int) == int(target_frame_idx))
    ].reset_index(drop=True)


def main() -> None:
    global STOP_REQUESTED
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    validate_sources()
    if not args.engineering_scene:
        require_smoke()
    scenes = selected_scenes(args)
    split = "engineering_smoke" if args.engineering_scene else "probe_train"
    out = output_directory(args.engineering_scene)
    pending = [
        row for row in scenes.itertuples(index=False)
        if not marker_valid(out / f"{row.scene_token}.complete.json", split)
    ]
    if not pending:
        print("no pending CARE-3D P2-A2-R1-F0 scenes")
        if not args.engineering_scene and not args.defer_progress:
            update_progress()
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
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(CHECKPOINT), map_location="cpu")
    model = freeze_module(model.to(device))
    head = model.pts_bbox_head
    assert_query_layout(
        int(head.num_query), int(head.num_propagated),
        int(head.num_query + head.num_propagated),
    )
    if int(head.topk_proposals) != 256:
        raise RuntimeError("R1-F0 topk_proposals changed")
    head.reset_memory()
    initial = snapshot(head)
    pc_range = head.pc_range.detach()
    for scene_row in pending:
        started = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        scene = str(scene_row.scene_token)
        tokens = json.loads(scene_row.sample_tokens_0_12)
        p1_frame, p1_arrays = main_p0_source(scene, args.engineering_scene)
        main_frame, _, p2a_audit = filter_p2a_rows(p1_frame, p1_arrays)
        if "gt_used_as_input" not in main_frame or main_frame.gt_used_as_input.astype(bool).any():
            raise RuntimeError("R1-F0 frozen online-anchor metadata indicates GT input")
        r0_rows = load_r0_rows(scene, args.engineering_scene)
        clean_state = initial
        anchor = None
        output_rows = []
        r0_assignment_exact = True
        p2a0_frozen_cost_exact = True
        feature_non_mutating = True
        branch_state_pass = True

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
            context = frame_context(clean_dataset.data_infos[index], clean_dataset)
            anchor = {
                "frame_idx": frame_idx,
                "output": output,
                "taps": taps,
                "context": context,
                "topk_indexes": recompute_topk_indexes(
                    output["all_cls_scores"][-1], int(head.topk_proposals)
                ),
            }
            del image, data, feats

        if anchor is None:
            raise RuntimeError("R1-F0 clean anchor missing")

        for target_frame_idx in range(3, 13):
            if anchor["frame_idx"] != target_frame_idx - 1:
                raise RuntimeError("R1-F0 clean anchor progression changed")
            row_indices = np.flatnonzero(
                main_frame.target_frame_idx.to_numpy(dtype=int) == target_frame_idx
            )
            target_index = token_index[tokens[target_frame_idx]]
            next_context = frame_context(clean_dataset.data_infos[target_index], clean_dataset)
            branch_states = clone_counterfactual_states(clean_state, 1 + len(PROTOCOLS))
            same_state = all(states_exact(branch_states[0], state) for state in branch_states[1:])
            branch_state_pass &= bool(same_state)
            if not same_state:
                raise RuntimeError("R1-F0 counterfactual branches do not share H_t")

            if row_indices.size:
                frame_rows = main_frame.iloc[row_indices].reset_index(drop=True)
                instances = frame_rows.instance_token.astype(str).tolist()
                anchor_features = []
                anchor_centers = []
                anchor_classes = []
                anchor_queries = []
                anchor_boxes = physical(anchor["output"], pc_range)[-1, 0].detach().float()
                for local, instance_token in enumerate(instances):
                    query = int(frame_rows.iloc[local].anchor_query_index)
                    anchor_queries.append(query)
                    anchor_features.append(anchor["taps"][TAPS[2]][query].detach().float())
                    anchor_centers.append(
                        anchor_boxes[query, :3].detach().cpu().numpy().astype(np.float32)
                    )
                    anchor_classes.append(
                        CLASSES.index(str(frame_rows.iloc[local].prediction_class))
                    )
                centers_target = transform_lidar_centers_between_frames(
                    np.stack(anchor_centers), anchor["context"], next_context
                )
                anchor_feature_tensor = torch.stack(anchor_features).to(device)
                anchor_center_tensor = torch.as_tensor(centers_target, device=device)
                anchor_class_tensor = torch.as_tensor(anchor_classes, device=device, dtype=torch.long)
            else:
                frame_rows = None
                instances = []
                anchor_queries = []
                anchor_classes = []
                anchor_feature_tensor = torch.empty((0, 256), device=device)
                anchor_center_tensor = torch.empty((0, 3), device=device)
                anchor_class_tensor = torch.empty((0,), device=device, dtype=torch.long)

            # Complete fault feature computation before the clean t+1 forward.
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
                    expected = _r0_group(r0_rows, protocol, target_frame_idx)
                    if len(expected) != len(frame_rows):
                        raise RuntimeError("R1-F0/R0 frame row count changed")
                    if expected.instance_token.astype(str).tolist() != instances:
                        raise RuntimeError("R1-F0/R0 frame row order changed")
                    assignment_equal = all((
                        np.array_equal(
                            assignments["p2a0_selected_query"],
                            expected.p2a0_selected_query.to_numpy(np.int64),
                        ),
                        np.array_equal(
                            assignments["lineage_child_query"],
                            expected.lineage_child_query.to_numpy(np.int64),
                        ),
                        np.array_equal(
                            assignments["hybrid_selected_query"],
                            expected.hybrid_selected_query.to_numpy(np.int64),
                        ),
                        np.array_equal(
                            assignments["hybrid_source"].astype(str),
                            expected.hybrid_source.astype(str).to_numpy(),
                        ),
                    ))
                    r0_assignment_exact &= bool(assignment_equal)
                    if not assignment_equal:
                        raise RuntimeError("R1-F0 recomputation changed an R0 assignment")
                    stored_cost = expected.p2a0_selected_cost.to_numpy(np.float64)
                    recomputed_cost = assignments["p2a0_selected_cost"]
                    cost_equal = np.array_equal(recomputed_cost, stored_cost, equal_nan=True)
                    p2a0_frozen_cost_exact &= bool(cost_equal)
                    if not cost_equal:
                        finite = np.isfinite(recomputed_cost) & np.isfinite(stored_cost)
                        max_abs_diff = float(
                            np.max(np.abs(recomputed_cost[finite] - stored_cost[finite]))
                        ) if finite.any() else float("nan")
                        raise RuntimeError(
                            "R1-F0 recomputation changed frozen P2-A0 selected cost: "
                            f"max_abs_diff={max_abs_diff:.17g}, "
                            f"recomputed={recomputed_cost[finite][:3].tolist()}, "
                            f"stored={stored_cost[finite][:3].tolist()}"
                        )

                    cost_before = frozen_cost.detach().clone()
                    component_before = {
                        key: value.detach().clone() for key, value in components.items()
                    }
                    evidence = relative_candidate_features(
                        components,
                        frozen_cost,
                        fault_logits,
                        anchor_class_tensor,
                        anchor_queries,
                        assignments["p2a0_selected_query"],
                        assignments["lineage_child_query"],
                        assignments["lineage_position"],
                        target_frame_idx=target_frame_idx,
                    )
                    unchanged = torch.equal(cost_before, frozen_cost) and all(
                        torch.equal(component_before[key], components[key])
                        for key in component_before
                    )
                    feature_non_mutating &= bool(unchanged)
                    if not unchanged:
                        raise RuntimeError("R1-F0 feature computation mutated frozen costs")

                    source_rows = evidence["source_row_index"]
                    # Oracle is first read here, after relative_candidate_features returned.
                    oracle = expected.oracle_query_index.to_numpy(np.int64)[source_rows]
                    labels = offline_disagreement_labels(
                        evidence["p2a0_selected_query"],
                        evidence["lineage_child_query"],
                        oracle,
                    )
                    for feature_index, source_row in enumerate(source_rows.tolist()):
                        row = {
                            "scene_token": scene,
                            "instance_token": instances[source_row],
                            "anchor_frame_idx": int(frame_rows.iloc[source_row].anchor_frame_idx),
                            "target_frame_idx": int(target_frame_idx),
                            "protocol": protocol,
                            "anchor_query_index": int(anchor_queries[source_row]),
                            "anchor_query_origin": query_origin(anchor_queries[source_row]),
                            "anchor_prediction_class": int(anchor_classes[source_row]),
                            "p2a0_selected_query": int(evidence["p2a0_selected_query"][feature_index]),
                            "lineage_child_query": int(evidence["lineage_child_query"][feature_index]),
                            "lineage_position": int(assignments["lineage_position"][source_row]),
                            "oracle_query_index": int(oracle[feature_index]),
                            "feature_computation_frozen_before_oracle": True,
                            "gt_used_as_feature_input": False,
                            "clean_future_used_as_feature_input": False,
                            "oracle_query_used_as_feature_input": False,
                        }
                        row.update({
                            column: evidence[column][feature_index]
                            for column in EVIDENCE_FEATURE_COLUMNS
                        })
                        row.update({column: labels[column][feature_index] for column in LABEL_COLUMNS})
                        output_rows.append(row)
                del fault_output, fault_taps, fault_feats, fault_image, fault_data

            if target_frame_idx < 12:
                clean_meta, clean_image, clean_data = unpack(clean_dataset[target_index], device)
                with torch.no_grad():
                    _, _, clean_feats = features(model, clean_image)
                    clean_output, clean_next_state, clean_taps = captured_head(
                        model, clean_meta, clean_data, clean_feats.detach(), True,
                        branch_states[0],
                    )
                anchor = {
                    "frame_idx": target_frame_idx,
                    "output": clean_output,
                    "taps": clean_taps,
                    "context": next_context,
                    "topk_indexes": recompute_topk_indexes(
                        clean_output["all_cls_scores"][-1], int(head.topk_proposals)
                    ),
                }
                clean_state = clean_next_state
                del clean_image, clean_data, clean_feats

        frame = pd.DataFrame(output_rows)
        if len(frame):
            if (frame.p2a0_selected_query == frame.lineage_child_query).any():
                raise RuntimeError("R1-F0 exported a non-disagreement row")
            label_sum = frame[list(LABEL_COLUMNS[:3])].astype(int).sum(axis=1)
            if not (label_sum == 1).all():
                raise RuntimeError("R1-F0 exported invalid offline labels")
            feature_values = frame.loc[:, EVIDENCE_FEATURE_COLUMNS].to_numpy(np.float64)
            nonfinite_feature_values = int((~np.isfinite(feature_values)).sum())
            model_matrix_finite = bool(np.isfinite(finite_model_matrix(frame)).all())
        else:
            nonfinite_feature_values = 0
            model_matrix_finite = True
        prefix = out / scene
        atomic_frame(prefix.with_suffix(".rows.csv"), frame)
        summary = {
            "schema_version": SCHEMA,
            "scene_token": scene,
            "split": split,
            "eligible_objects": int(len(main_frame)),
            "protocol_rows": int(len(r0_rows)),
            "disagreement_rows": int(len(frame)),
            "nonfinite_feature_values": nonfinite_feature_values,
            "fixed_model_matrix_finite": model_matrix_finite,
            "p2a_collision_excluded_rows": int(p2a_audit["total_excluded_rows"]),
            "r0_assignment_exact": bool(r0_assignment_exact),
            "p2a0_selected_query_exact": bool(r0_assignment_exact),
            "lineage_child_query_exact": bool(r0_assignment_exact),
            "p2a0_frozen_cost_exact": bool(p2a0_frozen_cost_exact),
            "feature_non_mutating": bool(feature_non_mutating),
            "branch_state_pass": bool(branch_state_pass),
            "feature_computation_frozen_before_oracle": True,
            "gt_used_as_feature_input": False,
            "clean_future_used_as_feature_input": False,
            "oracle_query_used_as_feature_input": False,
            "probe_val_read": False,
            "probe_test_read": False,
            "elapsed_seconds": float(time.time() - started),
            "peak_cuda_gib": float(torch.cuda.max_memory_allocated(device) / 1024 ** 3),
            "complete": True,
        }
        atomic_json(prefix.with_suffix(".complete.json"), summary)
        if args.engineering_scene:
            passed = all((
                summary["r0_assignment_exact"],
                summary["p2a0_frozen_cost_exact"],
                summary["feature_non_mutating"],
                summary["fixed_model_matrix_finite"],
                summary["branch_state_pass"],
                summary["disagreement_rows"] > 0,
            ))
            atomic_json(REPORT / "engineering_smoke.json", {
                **summary,
                "status": (
                    "P2A2_R1_F0_ENGINEERING_SMOKE_PASSED"
                    if passed else "P2A2_R1_F0_ENGINEERING_SMOKE_FAILED"
                ),
            })
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        if STOP_REQUESTED:
            print("stop requested; current R1-F0 scene saved", flush=True)
            break

    if not args.engineering_scene and not args.defer_progress:
        update_progress()


if __name__ == "__main__":
    main()
