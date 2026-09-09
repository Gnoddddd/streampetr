#!/usr/bin/env python3
"""Audit and resume-safely export frozen CARE-3D R1 confirmatory rows.

The exporter consumes per-scene schema-2 online evidence produced for the
audited official-val cohort. It never opens original probe-val/probe-test row
caches and never fits or searches a model or threshold.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import pandas as pd

from analysis.care3d_p2a2_r1_confirmation import (
    COHORT_CLEAN_STATUS,
    EXPECTED_OFFICIAL_VAL_SCENES,
    FROZEN_ARTIFACT_SHA256,
    FROZEN_P2A0,
    KEY_COLUMNS,
    SOURCE_HEAD,
    attach_offline_outcomes,
    default_discipline,
    heldout_lineage_audit,
    load_frozen_arbiter,
    require_clean_cohort,
    run_frozen_runtime,
    sha256_file,
    source_smoke,
    validate_full_population_keys,
)
from analysis.care3d_p2a2_r1_features import MODEL_FEATURE_COLUMNS
from analysis.care3d_p2a_association import PROTOCOLS


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/care3d/p2a2_r1_confirmation_c0"
FROZEN_ARTIFACT = ROOT / "reports/care3d/p2a2_r1_identity_arbiter/frozen_arbiter.json"
CONFIRMATORY_CONFIG = (
    ROOT / "configs/full_nuscenes/stream_petr_r50_90e_mechanism_val.py"
)
CONFIRMATORY_ANN_FILE = ROOT / "data/nuscenes/nuscenes2d_temporal_infos_val.pkl"
INCREMENTAL = REPORT / "incremental/official_val"
EVIDENCE = REPORT / "online_evidence/official_val"
SCHEMA_VERSION = 1
PROTOCOL_PATHS = {
    "blur_back": ROOT / "protocols/presets/motion_blur_back_10f_s09.json",
    "crash_back": ROOT / "protocols/presets/camera_crash_back_10f.json",
    "dark_back": ROOT / "protocols/presets/dark_back_10f_s09.json",
}
MINIMUM_LINEAGE_SOURCES = (
    "p2_r1_probe_train_419",
    "original_probe_val_133",
)
MINIMUM_LINEAGE_COUNTS = {
    "p2_r1_probe_train_419": 419,
    "original_probe_val_133": 133,
}
SOURCE_AUDIT_FLAGS = {
    "gt_used_as_association_input": False,
    "clean_future_used_as_association_input": False,
    "oracle_query_used_as_association_input": False,
    "feature_computation_frozen_before_oracle": True,
    "gt_used_as_feature_input": False,
    "clean_future_used_as_feature_input": False,
    "oracle_query_used_as_feature_input": False,
}
P2A0_CONFIG_COLUMNS = {
    "geo_weight": FROZEN_P2A0["geo_weight"],
    "embedding_weight": FROZEN_P2A0["embedding_weight"],
    "class_weight": FROZEN_P2A0["class_weight"],
    "max_cost": FROZEN_P2A0["max_cost"],
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def parse_named_manifest(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("lineage manifest must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("lineage manifest requires non-empty NAME and PATH")
    if "probe_test" in name.lower():
        raise argparse.ArgumentTypeError("probe_test remains locked")
    return name, Path(raw_path)


def reject_locked_data_path(path: Path, *, identity_manifest: bool = False) -> None:
    lowered = str(path).lower().replace("-", "_")
    if "probe_test" in lowered:
        raise RuntimeError("probe_test remains locked")
    if "probe_val" in lowered and not identity_manifest:
        raise RuntimeError("raw probe_val remains locked")
    if identity_manifest and "probe_val" in lowered and "manifest" not in path.name.lower():
        raise RuntimeError("probe_val access is limited to scene-identity manifests")


def read_identity_manifest(
    path: Path, name: str, *, require_split: bool = False
) -> pd.DataFrame:
    reject_locked_data_path(path, identity_manifest=True)
    header = pd.read_csv(path, nrows=0)
    allowed = {"scene_token", "split"}
    forbidden = set(header.columns) - allowed
    required = {"scene_token", "split"} if require_split else {"scene_token"}
    if not required <= set(header.columns) or forbidden:
        raise RuntimeError(
            f"{name} must contain scene identity/split provenance only; "
            f"extra={sorted(forbidden)}"
        )
    columns = [column for column in ("scene_token", "split") if column in header.columns]
    return pd.read_csv(path, usecols=columns)


def official_val_scene_tokens_from_metadata(data_root: Path) -> tuple[str, ...]:
    """Resolve the official 150-scene val split from metadata only."""
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    official_names = tuple(create_splits_scenes()["val"])
    if len(official_names) != EXPECTED_OFFICIAL_VAL_SCENES:
        raise RuntimeError("official nuScenes val scene-name count changed")
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(data_root), verbose=False)
    token_by_name = {str(scene["name"]): str(scene["token"]) for scene in nusc.scene}
    missing = sorted(set(official_names) - set(token_by_name))
    if missing:
        raise RuntimeError(f"official nuScenes val scene metadata is incomplete: {missing}")
    tokens = tuple(token_by_name[name] for name in official_names)
    if len(set(tokens)) != EXPECTED_OFFICIAL_VAL_SCENES:
        raise RuntimeError("official nuScenes val scene-token identity is not unique")
    return tokens


def validate_confirmatory_test_ann_file(config) -> str:
    """Hard-fail unless the parsed test dataset is the frozen val pickle."""
    try:
        observed = Path(str(config.data.test.ann_file)).resolve()
    except (AttributeError, TypeError) as error:
        raise RuntimeError("confirmatory config lacks data.test.ann_file") from error
    expected = CONFIRMATORY_ANN_FILE.resolve()
    if observed.name == "nuscenes2d_temporal_infos_train.pkl":
        raise RuntimeError("confirmatory extraction refuses the nuScenes train annotation pickle")
    if observed != expected:
        raise RuntimeError(
            f"confirmatory data.test.ann_file must be {expected}, observed={observed}"
        )
    return str(observed)


def prepare_cohort(
    official_val_manifest_path: Path,
    lineage_manifest_paths: Mapping[str, Path],
    required_sources: tuple[str, ...],
    *,
    report: Path = REPORT,
    official_val_scene_tokens: tuple[str, ...] | None = None,
    data_root: Path = ROOT / "data/nuscenes",
) -> tuple[pd.DataFrame, dict]:
    if not set(MINIMUM_LINEAGE_SOURCES) <= set(required_sources):
        raise RuntimeError(
            "lineage registry must include 419 probe_train and original 133 probe_val"
        )
    official = read_identity_manifest(
        official_val_manifest_path, "official-val manifest", require_split=True
    )
    metadata_tokens = (
        official_val_scene_tokens
        if official_val_scene_tokens is not None
        else official_val_scene_tokens_from_metadata(data_root)
    )
    sources = {
        name: read_identity_manifest(path, f"lineage manifest {name}")
        for name, path in lineage_manifest_paths.items()
    }
    for name, expected in MINIMUM_LINEAGE_COUNTS.items():
        if name in sources and len(sources[name]) != expected:
            raise RuntimeError(
                f"lineage source {name} must contain exactly {expected} scenes"
            )
    cohort, audit = heldout_lineage_audit(
        official,
        metadata_tokens,
        sources,
        required_sources=required_sources,
        official_val_manifest_sha256=sha256_file(official_val_manifest_path),
    )
    atomic_json(report / "heldout_lineage_audit.json", audit)
    if audit.get("status") != COHORT_CLEAN_STATUS:
        raise RuntimeError(str(audit.get("status")))
    atomic_csv(report / "confirmatory_manifest.csv", cohort)
    atomic_json(report / "progress_manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "status": "CONFIRMATORY_COHORT_READY_EXTRACTION_PENDING",
        "source_head": SOURCE_HEAD,
        "expected_scenes": len(cohort),
        "completed_scenes": 0,
        "protocols": list(PROTOCOLS),
        "frozen_artifact_sha256": FROZEN_ARTIFACT_SHA256,
        "probe_val_read": False,
        "probe_test_read": False,
    })
    return cohort, audit


def _load_cohort(report: Path) -> tuple[pd.DataFrame, dict]:
    manifest_path = report / "confirmatory_manifest.csv"
    audit_path = report / "heldout_lineage_audit.json"
    if not manifest_path.is_file() or not audit_path.is_file():
        raise RuntimeError("run the read-only heldout lineage audit first")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require_clean_cohort(audit)
    manifest = pd.read_csv(manifest_path)
    if set(manifest.columns) != {"scene_token", "split"}:
        raise RuntimeError("confirmatory manifest contains non-provenance columns")
    if set(manifest.split.astype(str)) != {"official_val"}:
        raise RuntimeError("confirmatory manifest is not official_val only")
    if len(manifest) != int(audit.get("confirmatory_scenes", -1)):
        raise RuntimeError("confirmatory manifest/audit count mismatch")
    if manifest.scene_token.astype(str).duplicated().any():
        raise RuntimeError("confirmatory manifest contains duplicate scenes")
    return manifest, audit


def marker_valid(path: Path, scene: str, source_sha256: str) -> bool:
    if not path.is_file():
        return False
    value = json.loads(path.read_text(encoding="utf-8"))
    discipline = default_discipline()
    sha_fields_valid = all(
        isinstance(value.get(field), str)
        and len(value[field]) == 64
        and all(character in "0123456789abcdef" for character in value[field])
        for field in ("source_rows_sha256", "rows_sha256")
    )
    return bool(
        value.get("complete") is True
        and value.get("schema_version") == SCHEMA_VERSION
        and value.get("split") == "official_val"
        and value.get("scene_token") == scene
        and value.get("source_rows_sha256") == source_sha256
        and value.get("frozen_artifact_sha256") == FROZEN_ARTIFACT_SHA256
        and value.get("tau_preference") == 0.52
        and value.get("tau_defer") == 0.88
        and sha_fields_valid
        and all(value.get(name) is expected for name, expected in discipline.items())
    )


def _validate_scene_input(frame: pd.DataFrame, scene: str) -> None:
    required = set(KEY_COLUMNS) | {
        "p2a0_selected_query",
        "lineage_child_query",
        "oracle_query_index",
        *MODEL_FEATURE_COLUMNS,
        *SOURCE_AUDIT_FLAGS,
        *P2A0_CONFIG_COLUMNS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"confirmatory scene input lacks columns: {sorted(missing)}")
    if len(frame) == 0:
        return
    if set(frame.scene_token.astype(str)) != {scene}:
        raise RuntimeError("confirmatory scene input identity changed")
    validate_full_population_keys(frame)
    if set(frame.protocol.astype(str)) != set(PROTOCOLS):
        raise RuntimeError("confirmatory scene does not contain all fixed protocols")
    identity = list(KEY_COLUMNS[:-1])
    expected_keys = None
    for protocol in PROTOCOLS:
        keys = set(map(tuple, frame.loc[
            frame.protocol.astype(str) == protocol, identity
        ].to_numpy()))
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise RuntimeError("protocols do not share the same eligible-object population")
    for flag, expected in SOURCE_AUDIT_FLAGS.items():
        values = frame[flag].astype(bool)
        if not (values == expected).all():
            raise RuntimeError(f"confirmatory source audit flag changed: {flag}")
    for column, expected in P2A0_CONFIG_COLUMNS.items():
        values = frame[column].to_numpy(dtype=float)
        if not (values == float(expected)).all():
            raise RuntimeError(f"frozen P2-A0 configuration changed: {column}")


def export_scene(
    scene: str,
    source_path: Path,
    *,
    artifact_path: Path = FROZEN_ARTIFACT,
    output_dir: Path = INCREMENTAL,
) -> bool:
    """Export one scene atomically; return False for a verified resume no-op."""
    reject_locked_data_path(source_path)
    source_sha = sha256_file(source_path)
    marker_path = output_dir / f"{scene}.complete.json"
    rows_path = output_dir / f"{scene}.rows.csv"
    if marker_valid(marker_path, scene, source_sha) and rows_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("rows_sha256") == sha256_file(rows_path):
            return False

    frozen = load_frozen_arbiter(artifact_path)
    artifact_before = sha256_file(artifact_path)
    frame = pd.read_csv(source_path)
    _validate_scene_input(frame, scene)
    if len(frame):
        decisions = run_frozen_runtime(
            frame.loc[:, MODEL_FEATURE_COLUMNS].copy(),
            frame.p2a0_selected_query,
            frame.lineage_child_query,
            frozen,
        )
        outcomes = attach_offline_outcomes(
            decisions, frame.p2a0_selected_query, frame.oracle_query_index
        )
        output = frame.copy()
        for column in outcomes.columns:
            output[column] = outcomes[column].to_numpy()
    else:
        output = frame.copy()
        for column in (
            "p_lineage", "p_both_wrong", "arbiter_selected_query",
            "arbiter_decision", "p2a0_exact", "p2a0_wrong",
            "p2a0_unmatched", "arbiter_exact", "arbiter_wrong",
            "arbiter_unmatched",
        ):
            dtype = "object" if column == "arbiter_decision" else "float64"
            output[column] = pd.Series(dtype=dtype)
    if sha256_file(artifact_path) != artifact_before:
        raise RuntimeError("frozen artifact was modified during confirmatory export")
    atomic_csv(rows_path, output)
    agreement = output.p2a0_selected_query == output.lineage_child_query
    marker = {
        "complete": True,
        "schema_version": SCHEMA_VERSION,
        "split": "official_val",
        "scene_token": scene,
        "rows": len(output),
        "agreement_rows": int(agreement.sum()),
        "disagreement_rows": int((~agreement).sum()),
        "source_rows_sha256": source_sha,
        "rows_sha256": sha256_file(rows_path),
        "frozen_artifact_sha256": frozen.artifact_sha256,
        "tau_preference": frozen.runtime.tau_preference,
        "tau_defer": frozen.runtime.tau_defer,
        "frozen_model_used": True,
        "frozen_thresholds_used": True,
        "model_refit": False,
        "threshold_search": False,
        "feature_search": False,
        "model_search": False,
        "calibration_search": False,
        "protocol_used_as_feature": False,
        "protocol_specific_threshold": False,
        "agreement_modified_count": int(
            output.loc[agreement, "arbiter_selected_query"].ne(
                output.loc[agreement, "p2a0_selected_query"]
            ).sum()
        ),
        "probe_val_read": False,
        "probe_test_read": False,
    }
    atomic_json(marker_path, marker)
    return True


def _scene_sample_tokens(nusc, scene_token: str, token_index: Mapping[str, int]) -> list[str]:
    scene = nusc.get("scene", str(scene_token))
    token = str(scene["first_sample_token"])
    tokens = []
    while token and len(tokens) < 13:
        tokens.append(token)
        token = str(nusc.get("sample", token).get("next", ""))
    if len(tokens) != 13:
        raise RuntimeError(f"official-val scene has fewer than 13 frames: {scene_token}")
    missing = [token for token in tokens if token not in token_index]
    if missing:
        raise RuntimeError(f"official-val scene is outside frozen detector infos: {scene_token}")
    return tokens


def _build_eligible_population(
    scene: str,
    tokens: list[str],
    *,
    model,
    clean_dataset,
    token_index: Mapping[str, int],
    nusc,
    device,
    initial_state,
    pc_range,
) -> pd.DataFrame:
    """Freeze the P1/P2-A0 eligible population in a clean-only pass."""
    import torch

    from analysis.care3d_p1 import filter_aligned_rows
    from analysis.care3d_p2a_association import filter_p2a_rows
    from scripts.audit_dark_target_recoverability import features, run_head, unpack
    from scripts.run_bd_temporal_support_p0 import frame_context
    from scripts.run_prospective_failure_features import (
        TAPS,
        captured_head,
        frame_record,
        target_frame,
    )

    head = model.pts_bbox_head
    clean_state = initial_state
    anchor = None
    for frame_idx in range(3):
        index = token_index[tokens[frame_idx]]
        meta, image, data = unpack(clean_dataset[index], device)
        with torch.no_grad():
            _, _, frame_features = features(model, image)
            pre_state = clean_state
            if frame_idx < 2:
                _, clean_state, _ = run_head(
                    model, meta, data, frame_features.detach(), frame_idx > 0, pre_state
                )
                continue
            output, clean_state, taps = captured_head(
                model, meta, data, frame_features.detach(), True, pre_state
            )
        targets = target_frame(nusc, tokens[frame_idx])
        context = frame_context(clean_dataset.data_infos[index], clean_dataset)
        candidates, _matches = frame_record(
            output, taps, pre_state, data, targets, context, pc_range, int(head.num_query)
        )
        anchor = {"frame_idx": frame_idx, "candidates": candidates}
    if anchor is None:
        raise RuntimeError("official-val eligibility pass lacks frame-2 anchor")

    rows = []
    for target_frame_idx in range(3, 13):
        if anchor["frame_idx"] != target_frame_idx - 1:
            raise RuntimeError("official-val clean eligibility progression changed")
        index = token_index[tokens[target_frame_idx]]
        meta, image, data = unpack(clean_dataset[index], device)
        with torch.no_grad():
            _, _, frame_features = features(model, image)
            pre_state = clean_state
            output, clean_state, taps = captured_head(
                model, meta, data, frame_features.detach(), True, pre_state
            )
        targets = target_frame(nusc, tokens[target_frame_idx])
        context = frame_context(clean_dataset.data_infos[index], clean_dataset)
        next_candidates, _matches = frame_record(
            output, taps, pre_state, data, targets, context, pc_range, int(head.num_query)
        )
        next_by_instance = {str(target["instance_token"]): target for target in targets}
        raw = []
        for instance_token, anchor_candidate in anchor["candidates"].items():
            instance_token = str(instance_token)
            if instance_token not in next_by_instance or instance_token not in next_candidates:
                continue
            clean_candidate = next_candidates[instance_token]
            raw.append({
                "scene_token": scene,
                "instance_token": instance_token,
                "anchor_frame_idx": target_frame_idx - 1,
                "target_frame_idx": target_frame_idx,
                "anchor_query_index": int(anchor_candidate["prediction"]["query"]),
                "target_clean_query_index": int(clean_candidate["prediction"]["query"]),
                "anchor_prediction_class": int(anchor_candidate["prediction"]["label"]),
            })
        frame = pd.DataFrame(raw)
        if len(frame):
            frame, _, _ = filter_aligned_rows(frame, {})
            frame, _, _ = filter_p2a_rows(frame, {})
            rows.append(frame)
        anchor = {"frame_idx": target_frame_idx, "candidates": next_candidates}
    columns = (
        "scene_token", "instance_token", "anchor_frame_idx", "target_frame_idx",
        "anchor_query_index", "target_clean_query_index", "anchor_prediction_class",
    )
    if not rows:
        return pd.DataFrame(columns=columns)
    output = pd.concat(rows, ignore_index=True)
    if output.duplicated(["instance_token", "target_frame_idx"]).any():
        raise RuntimeError("official-val eligible population contains duplicate tracks")
    return output.loc[:, columns]


def _extract_online_evidence(
    scene: str,
    tokens: list[str],
    eligible: pd.DataFrame,
    *,
    model,
    clean_dataset,
    fault_datasets: Mapping[str, object],
    token_index: Mapping[str, int],
    device,
    initial_state,
    pc_range,
) -> pd.DataFrame:
    """Replay online A/L evidence; clean t+1 runs only after all fault heads."""
    import torch

    from analysis.care3d_counterfactual import clone_counterfactual_states, states_exact
    from analysis.care3d_p2a2_lineage import (
        FROZEN_P2A0_CONFIG,
        lineage_first_assign,
        recompute_topk_indexes,
    )
    from analysis.care3d_p2a2_r1_features import relative_candidate_features
    from analysis.care3d_p2a_association import (
        MAX_GEOMETRY_DISTANCE_M,
        association_cost_components,
        transform_lidar_centers_between_frames,
        weighted_cost,
    )
    from scripts.audit_dark_target_recoverability import features, physical, run_head, unpack
    from scripts.run_bd_temporal_support_p0 import frame_context
    from scripts.run_prospective_failure_features import TAPS, captured_head

    head = model.pts_bbox_head
    clean_state = initial_state
    anchor = None
    for frame_idx in range(3):
        index = token_index[tokens[frame_idx]]
        meta, image, data = unpack(clean_dataset[index], device)
        with torch.no_grad():
            _, _, frame_features = features(model, image)
            pre_state = clean_state
            if frame_idx < 2:
                _, clean_state, _ = run_head(
                    model, meta, data, frame_features.detach(), frame_idx > 0, pre_state
                )
                continue
            output, clean_state, taps = captured_head(
                model, meta, data, frame_features.detach(), True, pre_state
            )
        anchor = {
            "frame_idx": frame_idx,
            "output": output,
            "taps": taps,
            "context": frame_context(clean_dataset.data_infos[index], clean_dataset),
            "topk_indexes": recompute_topk_indexes(
                output["all_cls_scores"][-1], int(head.topk_proposals)
            ),
        }
    if anchor is None:
        raise RuntimeError("official-val evidence pass lacks frame-2 anchor")

    output_rows = []
    for target_frame_idx in range(3, 13):
        if anchor["frame_idx"] != target_frame_idx - 1:
            raise RuntimeError("official-val evidence progression changed")
        frame_rows = eligible.loc[
            eligible.target_frame_idx.astype(int) == target_frame_idx
        ].reset_index(drop=True)
        target_index = token_index[tokens[target_frame_idx]]
        next_context = frame_context(clean_dataset.data_infos[target_index], clean_dataset)
        branch_states = clone_counterfactual_states(clean_state, 1 + len(PROTOCOLS))
        if not all(states_exact(branch_states[0], state) for state in branch_states[1:]):
            raise RuntimeError("official-val counterfactual branches do not share H_t")

        if len(frame_rows):
            anchor_queries = frame_rows.anchor_query_index.to_numpy(np.int64)
            anchor_boxes = physical(anchor["output"], pc_range)[-1, 0].detach().float()
            query_tensor = torch.as_tensor(anchor_queries, device=device, dtype=torch.long)
            anchor_features = anchor["taps"][TAPS[2]][query_tensor].detach().float()
            anchor_centers = anchor_boxes[query_tensor, :3].detach().cpu().numpy()
            centers_target = transform_lidar_centers_between_frames(
                anchor_centers, anchor["context"], next_context
            )
            anchor_center_tensor = torch.as_tensor(
                centers_target, device=device, dtype=torch.float32
            )
            anchor_class_tensor = torch.as_tensor(
                frame_rows.anchor_prediction_class.to_numpy(np.int64),
                device=device,
                dtype=torch.long,
            )
        else:
            anchor_queries = np.empty((0,), dtype=np.int64)
            anchor_features = torch.empty((0, 256), device=device)
            anchor_center_tensor = torch.empty((0, 3), device=device)
            anchor_class_tensor = torch.empty((0,), device=device, dtype=torch.long)

        for protocol_index, protocol in enumerate(PROTOCOLS, start=1):
            meta, image, data = unpack(fault_datasets[protocol][target_index], device)
            with torch.no_grad():
                _, _, frame_features = features(model, image)
                fault_output, _, fault_taps = captured_head(
                    model, meta, data, frame_features.detach(), True,
                    branch_states[protocol_index],
                )
            if not len(frame_rows):
                continue
            fault_queries = fault_taps[TAPS[2]].detach().float()
            fault_logits = fault_output["all_cls_scores"][-1, 0].detach().float()
            fault_boxes = physical(fault_output, pc_range)[-1, 0].detach().float()
            components = association_cost_components(
                anchor_features,
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
            features_by_row = {
                index: {column: float("nan") for column in MODEL_FEATURE_COLUMNS}
                for index in range(len(frame_rows))
            }
            for evidence_index, source_row in enumerate(evidence["source_row_index"]):
                features_by_row[int(source_row)] = {
                    column: evidence[column][evidence_index]
                    for column in MODEL_FEATURE_COLUMNS
                }
            # Oracle identity is attached only after A/L evidence is frozen.
            oracle = frame_rows.target_clean_query_index.to_numpy(np.int64)
            for row_index, row in enumerate(frame_rows.itertuples(index=False)):
                value = {
                    "scene_token": scene,
                    "instance_token": str(row.instance_token),
                    "anchor_frame_idx": int(row.anchor_frame_idx),
                    "target_frame_idx": int(row.target_frame_idx),
                    "protocol": protocol,
                    "p2a0_selected_query": int(assignments["p2a0_selected_query"][row_index]),
                    "lineage_child_query": int(assignments["lineage_child_query"][row_index]),
                    "oracle_query_index": int(oracle[row_index]),
                    **SOURCE_AUDIT_FLAGS,
                    **P2A0_CONFIG_COLUMNS,
                }
                value.update(features_by_row[row_index])
                output_rows.append(value)

        # Advance the clean state only after every fault-protocol forward.
        meta, image, data = unpack(clean_dataset[target_index], device)
        with torch.no_grad():
            _, _, frame_features = features(model, image)
            clean_output, clean_state, clean_taps = captured_head(
                model, meta, data, frame_features.detach(), True, branch_states[0]
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
    if not output_rows:
        columns = [
            *KEY_COLUMNS, "p2a0_selected_query", "lineage_child_query",
            "oracle_query_index", *MODEL_FEATURE_COLUMNS, *SOURCE_AUDIT_FLAGS,
            *P2A0_CONFIG_COLUMNS,
        ]
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(output_rows)


def extract_official_val(
    *,
    artifact_path: Path = FROZEN_ARTIFACT,
    device_name: str = "cuda:0",
    num_shards: int = 1,
    shard_index: int = 0,
    max_scenes: int | None = None,
) -> dict:
    """Formal entry point, locked behind a completed identity-only audit."""
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("shards require 0 <= shard-index < num-shards")
    load_frozen_arbiter(artifact_path)
    manifest, _audit = _load_cohort(REPORT)
    selected = manifest.iloc[shard_index::num_shards].reset_index(drop=True)
    if max_scenes is not None:
        selected = selected.iloc[:max_scenes].reset_index(drop=True)
    staging = REPORT / "online_evidence/official_val"
    needs_forward = []
    for scene in selected.scene_token.astype(str):
        source_path = staging / f"{scene}.features.csv"
        if source_path.is_file():
            export_scene(scene, source_path, artifact_path=artifact_path)
        else:
            needs_forward.append(scene)
    if not needs_forward:
        return update_progress()
    selected = selected.loc[
        selected.scene_token.astype(str).isin(needs_forward)
    ].reset_index(drop=True)

    # Framework imports are delayed so identity audit and source smoke cannot
    # accidentally initialize or read the official validation detector data.
    stream = ROOT / "repos/StreamPETR"
    if str(stream) not in sys.path:
        sys.path.insert(0, str(stream))
    import torch
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmdet3d.models import build_model
    from nuscenes.nuscenes import NuScenes

    from analysis.care3d_counterfactual import freeze_module
    from analysis.care3d_p2a_association import assert_query_layout
    from analysis.care3d_p2a_execution import build_shared_protocol_dataset
    from scripts.audit_dark_target_recoverability import snapshot
    from scripts.run_bd_temporal_support_p0 import (
        CHECKPOINT,
        DATA,
        protocol_dataset,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for formal official-val extraction")
    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(device_name)
    cfg = Config.fromfile(str(CONFIRMATORY_CONFIG))
    validate_confirmatory_test_ann_file(cfg)
    import_modules_from_strings(**cfg.custom_imports)
    cfg.model.pretrained = None
    clean_dataset = protocol_dataset(cfg, None)
    fault_datasets = {
        protocol: build_shared_protocol_dataset(clean_dataset, cfg, path)
        for protocol, path in PROTOCOL_PATHS.items()
    }
    token_index = {
        str(info["token"]): index
        for index, info in enumerate(clean_dataset.data_infos)
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
        raise RuntimeError("frozen StreamPETR topk_proposals changed")
    head.reset_memory()
    initial_state = snapshot(head)
    pc_range = head.pc_range.detach()
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(DATA), verbose=False)
    for scene in selected.scene_token.astype(str):
        source_path = staging / f"{scene}.features.csv"
        tokens = _scene_sample_tokens(nusc, scene, token_index)
        eligible = _build_eligible_population(
            scene,
            tokens,
            model=model,
            clean_dataset=clean_dataset,
            token_index=token_index,
            nusc=nusc,
            device=device,
            initial_state=initial_state,
            pc_range=pc_range,
        )
        evidence = _extract_online_evidence(
            scene,
            tokens,
            eligible,
            model=model,
            clean_dataset=clean_dataset,
            fault_datasets=fault_datasets,
            token_index=token_index,
            device=device,
            initial_state=initial_state,
            pc_range=pc_range,
        )
        atomic_csv(source_path, evidence)
        export_scene(scene, source_path, artifact_path=artifact_path)
    return update_progress()


def update_progress(
    *, report: Path = REPORT, output_dir: Path = INCREMENTAL
) -> dict:
    manifest, _audit = _load_cohort(report)
    completed = 0
    rows = 0
    for scene in manifest.scene_token.astype(str):
        marker_path = output_dir / f"{scene}.complete.json"
        if not marker_path.is_file():
            continue
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        rows_path = output_dir / f"{scene}.rows.csv"
        if not rows_path.is_file() or marker.get("rows_sha256") != sha256_file(rows_path):
            continue
        source_sha = str(marker.get("source_rows_sha256"))
        if not marker_valid(marker_path, scene, source_sha):
            continue
        completed += 1
        rows += int(marker.get("rows", 0))
    status = (
        "CONFIRMATORY_EXTRACTION_COMPLETE_ANALYSIS_ELIGIBLE"
        if completed == len(manifest)
        else "CONFIRMATORY_EXTRACTION_RUNNING"
    )
    progress = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "source_head": SOURCE_HEAD,
        "expected_scenes": len(manifest),
        "completed_scenes": completed,
        "rows": rows,
        "protocols": list(PROTOCOLS),
        "frozen_artifact_sha256": FROZEN_ARTIFACT_SHA256,
        "probe_val_read": False,
        "probe_test_read": False,
    }
    atomic_json(report / "progress_manifest.json", progress)
    return progress


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--source-smoke", action="store_true")
    actions.add_argument("--audit-only", action="store_true")
    actions.add_argument("--export", action="store_true")
    actions.add_argument("--extract-official-val", action="store_true")
    parser.add_argument("--official-val-manifest", type=Path)
    parser.add_argument(
        "--lineage-manifest", action="append", type=parse_named_manifest, default=[]
    )
    parser.add_argument("--required-source", action="append", default=[])
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--artifact", type=Path, default=FROZEN_ARTIFACT)
    args = parser.parse_args(argv)
    if args.max_scenes is not None and args.max_scenes < 0:
        parser.error("--max-scenes must be non-negative")
    if args.audit_only and args.official_val_manifest is None:
        parser.error("--audit-only requires --official-val-manifest")
    if args.audit_only and not args.lineage_manifest:
        parser.error("--audit-only requires all --lineage-manifest inputs")
    if args.export and args.input_dir is None:
        parser.error("--export requires --input-dir")
    if args.export:
        try:
            reject_locked_data_path(args.input_dir)
        except RuntimeError as error:
            parser.error(str(error))
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("shards require 0 <= shard-index < num-shards")
    if not args.extract_official_val and (
        args.num_shards != 1 or args.shard_index != 0
    ):
        parser.error("sharding is valid only for --extract-official-val")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    frozen = load_frozen_arbiter(args.artifact)
    if args.source_smoke:
        print(json.dumps(source_smoke(frozen), indent=2, sort_keys=True))
        return
    if args.audit_only:
        named = dict(args.lineage_manifest)
        if len(named) != len(args.lineage_manifest):
            raise RuntimeError("duplicate lineage manifest name")
        required = tuple(args.required_source or named.keys())
        _cohort, audit = prepare_cohort(
            args.official_val_manifest, named, required
        )
        print(json.dumps(audit, indent=2, sort_keys=True))
        return

    if args.extract_official_val:
        progress = extract_official_val(
            artifact_path=args.artifact,
            device_name=args.device,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            max_scenes=args.max_scenes,
        )
        print(json.dumps(progress, indent=2, sort_keys=True))
        return

    reject_locked_data_path(args.input_dir)
    manifest, _audit = _load_cohort(REPORT)
    if args.max_scenes is not None:
        manifest = manifest.iloc[:args.max_scenes]
    for scene in manifest.scene_token.astype(str):
        path = args.input_dir / f"{scene}.features.csv"
        if not path.is_file():
            raise RuntimeError(f"missing official-val scene evidence: {scene}")
        export_scene(scene, path, artifact_path=args.artifact)
    print(json.dumps(update_progress(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
