#!/usr/bin/env python3
"""Incremental P0 representation drift capture on the frozen train population."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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
from mmdet3d.datasets import build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from nuscenes.nuscenes import NuScenes  # noqa: E402

from analysis.fault_boundary_root_cause import candidate_pool_statistics  # noqa: E402
from analysis.temporal_representation_localization import (  # noqa: E402
    geometry_candidates,
    geometry_match,
    matched_representation_metrics,
)
from scripts.audit_dark_target_recoverability import (  # noqa: E402
    features,
    local_gt,
    physical,
    run_head,
    snapshot,
    unpack,
)


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
TAPS = (
    "temporal_alignment_query_state",
    "decoder_layer5_temporal_self_attn_output",
    "final_decoder_pre_cls_query",
)
PAIRS = {"AC": ("A", "C"), "BD": ("B", "D")}
SCHEMA = 1
STOP_REQUESTED = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=tuple(PROTOCOLS), required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise RuntimeError(f"empty scene checkpoint: {path}")
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


def captured_head(model, meta, data, feats, prev_exists, state):
    """Run the unchanged head while passively capturing all preregistered taps."""

    head = model.pts_bbox_head
    if "temporal_alignment" in head.__dict__:
        raise RuntimeError("unexpected instance-level temporal_alignment override")
    captures = {key: [] for key in TAPS}
    original_temporal_alignment = head.temporal_alignment

    def temporal_alignment_wrapper(query_pos, tgt, reference_points):
        result = original_temporal_alignment(query_pos, tgt, reference_points)
        captures["temporal_alignment_query_state"].append(
            torch.cat([result[0], result[1]], dim=-1)
        )
        return result

    head.temporal_alignment = temporal_alignment_wrapper

    def temporal_attention_hook(_module, _arguments, output):
        captures["decoder_layer5_temporal_self_attn_output"].append(output)

    def classifier_pre_hook(_module, arguments):
        captures["final_decoder_pre_cls_query"].append(arguments[0])

    attention_handle = head.transformer.decoder.layers[5].attentions[0].register_forward_hook(
        temporal_attention_hook
    )
    classifier_handle = head.cls_branches[-1].register_forward_pre_hook(classifier_pre_hook)
    try:
        output, next_state, _ = run_head(model, meta, data, feats, prev_exists, state)
    finally:
        attention_handle.remove()
        classifier_handle.remove()
        del head.temporal_alignment
    if len(captures["temporal_alignment_query_state"]) != 1:
        raise RuntimeError("temporal-alignment tap capture count mismatch")
    if len(captures["decoder_layer5_temporal_self_attn_output"]) != 1:
        raise RuntimeError("final temporal-attention tap capture count mismatch")
    if len(captures["final_decoder_pre_cls_query"]) != len(head.cls_branches):
        raise RuntimeError("shared classification tap capture count mismatch")
    temporal = captures["temporal_alignment_query_state"][0]
    attention = captures["decoder_layer5_temporal_self_attn_output"][0]
    final_query = captures["final_decoder_pre_cls_query"][-1]
    if attention.ndim != 3 or attention.shape[1] != temporal.shape[0]:
        raise RuntimeError(f"unexpected temporal-attention shape: {attention.shape}")
    attention = attention.transpose(0, 1).contiguous()
    result = {
        "temporal_alignment_query_state": temporal.detach().float().cpu().numpy(),
        "decoder_layer5_temporal_self_attn_output": attention.detach().float().cpu().numpy(),
        "final_decoder_pre_cls_query": final_query.detach().float().cpu().numpy(),
    }
    expected = {
        "temporal_alignment_query_state": (1, 900, 512),
        "decoder_layer5_temporal_self_attn_output": (1, 900, 256),
        "final_decoder_pre_cls_query": (1, 900, 256),
    }
    observed = {key: value.shape for key, value in result.items()}
    if observed != expected:
        raise RuntimeError(f"tap shapes differ from preregistration: {observed}")
    return output, next_state, {key: value[0] for key, value in result.items()}


def compare_outputs(left, right) -> tuple[bool, float]:
    differences = []
    exact = True
    for key in ("all_cls_scores", "all_bbox_preds"):
        a, b = left[key], right[key]
        exact &= torch.equal(a, b)
        differences.append(float((a.detach().float() - b.detach().float()).abs().max().item()))
    return bool(exact), max(differences, default=0.0)


def compare_states(left: dict, right: dict) -> tuple[bool, float]:
    exact = True
    differences = []
    for key in left:
        a, b = left[key], right[key]
        if a is None and b is None:
            differences.append(0.0)
        elif a is None or b is None:
            return False, math.inf
        else:
            exact &= torch.equal(a, b)
            differences.append(float((a.detach().float() - b.detach().float()).abs().max().item()))
    return bool(exact), max(differences, default=0.0)


def validate_source_replay(row: dict, condition: str, logits, boxes) -> None:
    check = candidate_pool_statistics(
        logits, boxes, row["gt_center"], int(row["gt_label"]), 100, 2.0
    )
    expected_candidate = bool(row[f"{condition}_candidate"])
    if check["candidate_available"] != expected_candidate:
        raise RuntimeError(f"{row['unit_id']}/{condition}: candidate replay mismatch")
    if expected_candidate:
        if int(check["best_query"]) != int(row[f"{condition}_qplus"]):
            raise RuntimeError(f"{row['unit_id']}/{condition}: q+ replay mismatch")
        if abs(float(check["s_pos"]) - float(row[f"{condition}_s_pos"])) > 2e-6:
            raise RuntimeError(f"{row['unit_id']}/{condition}: S_pos replay mismatch")


def update_status() -> None:
    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    if str(progress.get("status", "")).startswith("FINAL_"):
        return
    population = pd.read_csv(REPORT / "population.csv", usecols=["protocol", "scene_token"])
    lines = [
        "# PARTIAL STATUS", "", "`P0_RUNNING`", "",
        "Partial P0 coverage cannot produce a tap decision.", "",
        "| protocol | scenes | per-GT/tap/pair rows |", "|---|---:|---:|",
    ]
    all_complete = True
    p0 = {}
    for protocol in PROTOCOLS:
        expected = int(population[population.protocol == protocol].scene_token.nunique())
        directory = REPORT / "incremental/P0" / protocol
        metas = [json.loads(path.read_text()) for path in directory.glob("*.complete.json")]
        tokens = sorted(meta["scene_token"] for meta in metas if meta.get("complete"))
        rows = sum(int(meta["drift_rows"]) for meta in metas if meta.get("complete"))
        p0[protocol] = {"completed_scenes": tokens, "expected_scenes": expected, "rows": rows}
        lines.append(f"| {protocol} | {len(tokens)}/{expected} | {rows} |")
        all_complete &= len(tokens) == expected
    progress["stages"]["P0"] = p0
    progress["status"] = "P0_COMPLETE_ANALYSIS_PENDING" if all_complete else "P0_RUNNING"
    atomic_json(REPORT / "progress_manifest.json", progress)
    lines += ["", "Resume:", "", "```bash"]
    lines += [f"python scripts/run_temporal_representation_p0.py --protocol {p}" for p in PROTOCOLS]
    lines += ["python scripts/analyze_temporal_representation_p0.py", "```", ""]
    temporary = REPORT / "PARTIAL_STATUS.md.tmp"
    temporary.write_text("\n".join(lines))
    os.replace(temporary, REPORT / "PARTIAL_STATUS.md")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    validation = json.loads((REPORT / "source_validation.json").read_text())
    if validation.get("status") != "VALIDATED_BEFORE_FORWARD":
        raise RuntimeError("pre-forward source validation missing")
    if sha256(REPORT / "population.csv") != validation["population_sha256"]:
        raise RuntimeError("population hash mismatch")
    if sha256(REPORT / "tap_manifest.csv") != validation["tap_manifest_sha256"]:
        raise RuntimeError("tap manifest hash mismatch")
    population = pd.read_csv(REPORT / "population.csv")
    population = population[population.protocol == args.protocol]
    selected_scenes = pd.read_csv(SOURCE / "scene_list.csv").scene_token.tolist()
    output_dir = REPORT / "incremental/P0" / args.protocol
    pending = []
    for scene in selected_scenes:
        meta_path = output_dir / f"{scene}.complete.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if (
                meta.get("complete") is True
                and meta.get("schema_version") == SCHEMA
                and meta.get("population_sha256") == validation["population_sha256"]
                and meta.get("tap_manifest_sha256") == validation["tap_manifest_sha256"]
            ):
                continue
        pending.append(scene)
    if args.max_scenes is not None:
        pending = pending[:args.max_scenes]
    if not pending:
        print(f"no pending P0 scenes for {args.protocol}")
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
        by_sample = {
            token: group.to_dict("records")
            for token, group in scene_population.groupby("sample_token")
        }
        clean_state, fault_state = initial, initial
        rows = []
        equivalence_rows = []
        equivalence_checked = False
        for frame_idx, token in enumerate(scene_tokens[scene]):
            index = token_to_index[token]
            clean_meta, clean_image, clean_data = unpack(clean_dataset[index], device)
            fault_meta, fault_image, fault_data = unpack(fault_dataset[index], device)
            with torch.no_grad():
                _, _, clean_feats = features(model, clean_image)
                _, _, fault_feats = features(model, fault_image)
                clean_feats = clean_feats.detach()
                fault_feats = fault_feats.detach()
                clean_pre, fault_pre = clean_state, fault_state
            frame_population = by_sample.get(token, [])
            if not frame_population:
                with torch.no_grad():
                    _, clean_state, _ = run_head(
                        model, clean_meta, clean_data, clean_feats, frame_idx > 0, clean_pre
                    )
                    _, fault_state, _ = run_head(
                        model, fault_meta, fault_data, fault_feats, frame_idx > 0, fault_pre
                    )
                continue
            with torch.no_grad():
                if not equivalence_checked:
                    canonical_a, canonical_state, _ = run_head(
                        model, clean_meta, clean_data, clean_feats, frame_idx > 0, clean_pre
                    )
                    output_a, clean_state, taps_a = captured_head(
                        model, clean_meta, clean_data, clean_feats, frame_idx > 0, clean_pre
                    )
                    output_exact, output_diff = compare_outputs(canonical_a, output_a)
                    state_exact, state_diff = compare_states(canonical_state, clean_state)
                    if not output_exact or not state_exact:
                        raise RuntimeError("passive tap capture changed B0 output/state")
                    equivalence_rows.append({
                        "protocol": args.protocol,
                        "scene_token": scene,
                        "sample_token": token,
                        "frame_idx": frame_idx,
                        "passive_capture_output_bitwise_equal": output_exact,
                        "passive_capture_output_max_abs_diff": output_diff,
                        "passive_capture_memory_bitwise_equal": state_exact,
                        "passive_capture_memory_max_abs_diff": state_diff,
                    })
                    equivalence_checked = True
                else:
                    output_a, clean_state, taps_a = captured_head(
                        model, clean_meta, clean_data, clean_feats, frame_idx > 0, clean_pre
                    )
                output_b, _, taps_b = captured_head(
                    model, fault_meta, fault_data, fault_feats, frame_idx > 0, clean_pre
                )
                output_c, _, taps_c = captured_head(
                    model, clean_meta, clean_data, clean_feats, frame_idx > 0, fault_pre
                )
                output_d, fault_state, taps_d = captured_head(
                    model, fault_meta, fault_data, fault_feats, frame_idx > 0, fault_pre
                )

            targets = {target["token"]: target for target in local_gt(nusc, token)}
            outputs = {"A": output_a, "B": output_b, "C": output_c, "D": output_d}
            taps = {"A": taps_a, "B": taps_b, "C": taps_c, "D": taps_d}
            logits = {
                condition: output["all_cls_scores"][-1, 0].detach().float().cpu().numpy()
                for condition, output in outputs.items()
            }
            boxes = {
                condition: physical(output, pc_range)[-1, 0].detach().float().cpu().numpy()
                for condition, output in outputs.items()
            }
            for frozen in frame_population:
                target = targets[frozen["gt_token"]]
                enriched = dict(frozen)
                enriched["gt_center"] = target["center"]
                enriched["gt_label"] = target["label"]
                for condition in ("A", "B", "C", "D"):
                    validate_source_replay(enriched, condition, logits[condition], boxes[condition])
                candidates = {
                    condition: geometry_candidates(boxes[condition], target["center"], 2.0)
                    for condition in ("A", "B", "C", "D")
                }
                for pair_name, (source_condition, target_condition) in PAIRS.items():
                    pairs = geometry_match(
                        boxes[source_condition], boxes[target_condition],
                        candidates[source_condition], candidates[target_condition],
                    )
                    geometry_distance = [value[2] for value in pairs]
                    for tap in TAPS:
                        metrics = matched_representation_metrics(
                            taps[source_condition][tap], taps[target_condition][tap], pairs
                        )
                        rows.append({
                            "unit_id": frozen["unit_id"],
                            "protocol": args.protocol,
                            "scene_token": scene,
                            "sample_token": token,
                            "frame_idx": frame_idx,
                            "gt_token": frozen["gt_token"],
                            "instance_token": frozen["instance_token"],
                            "gt_class": frozen["gt_class"],
                            "distance_m": frozen["distance_m"],
                            "visibility_token": frozen["visibility_token"],
                            "population": "lost" if bool(frozen["lost"]) else "retained",
                            "pair": pair_name,
                            "source_condition": source_condition,
                            "target_condition": target_condition,
                            "tap_id": tap,
                            "source_candidate_count": len(candidates[source_condition]),
                            "target_candidate_count": len(candidates[target_condition]),
                            "matched_pair_count": metrics["matched_pair_count"],
                            "matched_source_queries": json.dumps([value[0] for value in pairs]),
                            "matched_target_queries": json.dumps([value[1] for value in pairs]),
                            "median_geometry_match_distance": (
                                float(np.median(geometry_distance)) if geometry_distance else math.nan
                            ),
                            "cosine_distance": metrics["cosine_distance"],
                            "normalized_l2": metrics["normalized_l2"],
                            "pair_cosine_distances": json.dumps(metrics["pair_cosine_distances"]),
                            "pair_normalized_l2": json.dumps(metrics["pair_normalized_l2"]),
                        })
            del output_a, output_b, output_c, output_d, taps_a, taps_b, taps_c, taps_d
            torch.cuda.empty_cache()

        expected_rows = len(scene_population) * len(PAIRS) * len(TAPS)
        if len(rows) != expected_rows:
            raise RuntimeError(f"P0 scene row mismatch: {len(rows)} != {expected_rows}")
        atomic_csv(output_dir / f"{scene}.csv", rows)
        atomic_csv(output_dir / f"{scene}.equivalence.csv", equivalence_rows)
        atomic_json(output_dir / f"{scene}.complete.json", {
            "schema_version": SCHEMA,
            "population_sha256": validation["population_sha256"],
            "tap_manifest_sha256": validation["tap_manifest_sha256"],
            "protocol": args.protocol,
            "scene_token": scene,
            "population_rows": len(scene_population),
            "drift_rows": len(rows),
            "passive_equivalence_rows": len(equivalence_rows),
            "complete": True,
        })
        update_status()
        print(
            f"completed P0 {args.protocol}/{scene} population={len(scene_population)} "
            f"drift_rows={len(rows)}",
            flush=True,
        )
        if STOP_REQUESTED:
            print("stop requested; current P0 scene saved", flush=True)
            break


if __name__ == "__main__":
    main()

