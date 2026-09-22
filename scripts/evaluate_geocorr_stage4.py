#!/usr/bin/env python3
"""Formal Baseline-vs-GeoCorr evaluation on the OccNuScenes val subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

import torch

ROOT = Path(__file__).resolve().parents[1]
STREAM_ROOT = ROOT / "repos/StreamPETR"
for import_path in (ROOT, STREAM_ROOT, STREAM_ROOT / "mmdetection3d"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from configs.geocorr_recovery.stage1_infrastructure import candidate_offsets  # noqa: E402
from datasets.paired_occ_nuscenes import read_manifest  # noqa: E402
from evaluation.geocorr_stage4_evaluator import (  # noqa: E402
    NuScenesSubsetMetric,
    aggregate_fault_summary,
    average_diagnostics,
    checkpoint_payload,
    condition_records,
    current_tokens,
    format_subset_results,
    load_progress,
    normalize_prediction,
    recovery_diagnostics,
    safe_condition_name,
    save_progress,
    unique_records_by_current_token,
    write_json,
)
from evaluation.streampetr_runtime import (  # noqa: E402
    build_streampetr_dataset,
    build_streampetr_model_runtime,
    prepare_streampetr_model_batch,
)
from models.adapters import from_streampetr_result  # noqa: E402
from models.geocorr_recovery import (  # noqa: E402
    GeometryCandidateSampler,
    sample_candidate_features,
    sample_history_candidate_features,
    select_top_predictions,
)
from scripts.audit_geocorr_clean_correspondence import _image_shapes  # noqa: E402
from scripts.smoke_geocorr_stage3b_real import (  # noqa: E402
    _dataset_token_index,
    _prepare_batch,
    _resolved_paths,
)
from training.geocorr_stage3c_trainer import (  # noqa: E402
    GeoCorrStage3CModel,
    Stage3CConfig,
    frozen_detector_parameter_checksum,
)

DEFAULT_CONFIG = "configs/full_nuscenes/stream_petr_r50_90e_clean_val.py"
DEFAULT_CHECKPOINT = "checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth"


def _parse_top_k(value: str) -> Union[int, str]:
    if value == "all":
        return value
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("top-k must be positive or 'all'")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--nuscenes-root", required=True)
    parser.add_argument("--dirty-root", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--detector-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--geocorr-checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--top-k", type=_parse_top_k, default=25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("baseline", "geocorr", "both"), default="both")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--evaluate-clean", action="store_true")
    args = parser.parse_args()
    if args.max_pairs is not None and args.max_pairs <= 0:
        parser.error("--max-pairs must be positive")
    if args.log_interval <= 0:
        parser.error("--log-interval must be positive")
    if args.mode in ("geocorr", "both") and not args.geocorr_checkpoint:
        parser.error("--geocorr-checkpoint is required for GeoCorr evaluation")
    return args


def _resolve(path: str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage3_config(payload: Mapping[str, Any]) -> Stage3CConfig:
    cfg = payload["config"]
    return Stage3CConfig(
        feature_dim=int(cfg["feature_dim"]),
        candidate_offsets=tuple(tuple(float(x) for x in row) for row in cfg["candidate_offsets"]),
        top_k=cfg["top_k"],
        temperature=float(cfg["temperature"]),
        lambda_corr=float(cfg["lambda_corr"]),
        lambda_rec=float(cfg["lambda_rec"]),
    )


def _load_geocorr(path: Path, detector_checksum: str, device: torch.device, top_k):
    payload = checkpoint_payload(path, detector_checksum)
    cfg = _stage3_config(payload)
    if cfg.top_k != top_k:
        raise RuntimeError("GeoCorr checkpoint top-k differs from requested top-k")
    model = GeoCorrStage3CModel(cfg).to(device)
    model.load_state_dict(payload["geocorr"], strict=True)
    model.eval()
    return model, payload


def _direct_from_fpn(runtime, packed: Mapping[str, Any], fpn: torch.Tensor):
    detector = runtime.detector
    detector.eval()
    detector.pts_bbox_head.reset_memory()
    detector.prev_scene_token = None
    data = {key: value for key, value in packed.items() if key != "img_metas"}
    data["img_feats"] = fpn
    with torch.no_grad():
        result = detector.simple_test_pts(packed["img_metas"], **data)
    if len(result) != 1:
        raise RuntimeError("B=1 downstream inference expected one result")
    return normalize_prediction(result[0])


def _geocorr_prediction(
    record,
    dataset,
    token_index,
    runtime,
    model,
    previous_data,
    current_data,
    current_fpn,
    device,
    top_k,
):
    prev_token = str(record["previous_sample_token"])
    cur_token = str(record["current_sample_token"])
    prev_index, cur_index = token_index[prev_token], token_index[cur_token]
    prev_info, cur_info = dataset.data_infos[prev_index], dataset.data_infos[cur_index]

    prev_result, prev_fpn = runtime.forward(previous_data, reset_memory=True)
    prediction = select_top_predictions(
        from_streampetr_result(prev_result[0], float(prev_info["timestamp"]) / 1e6),
        top_k,
    )
    if prediction.score.numel() == 0:
        raise RuntimeError("previous detector produced no selectable predictions")

    previous = prepare_streampetr_model_batch(previous_data, device)
    current = prepare_streampetr_model_batch(current_data, device)
    current_from_previous = current["ego_pose_inv"][0].float() @ previous["ego_pose"][0].float()
    dt = float(record["dt"])
    observed_dt = float(cur_info["timestamp"] - prev_info["timestamp"]) / 1e6
    if dt <= 0 or abs(dt - observed_dt) > 1e-9:
        raise ValueError("manifest and annotation dt do not match")

    geometry = GeometryCandidateSampler(candidate_offsets).forward_with_history(
        prediction,
        current_from_previous,
        dt,
        current["lidar2img"][0].float(),
        _image_shapes(current["img_metas"][0], "img_shape"),
        tuple(current_fpn.shape[-2:]),
        previous["lidar2img"][0].float(),
        _image_shapes(previous["img_metas"][0], "img_shape"),
        tuple(prev_fpn.shape[-2:]),
        current_padded_image_shapes=_image_shapes(current["img_metas"][0], "pad_shape"),
        previous_padded_image_shapes=_image_shapes(previous["img_metas"][0], "pad_shape"),
    )
    current_grid = geometry["current_grid_coords"].unsqueeze(0).to(device)
    current_valid = geometry["current_valid_mask"].unsqueeze(0).to(device)
    history_grid = geometry["previous_grid_coords"].unsqueeze(0).to(device)
    history_valid = geometry["previous_valid_mask"].unsqueeze(0).to(device)
    dirty_tokens = sample_candidate_features(current_fpn, current_grid, current_valid)
    history_tokens = sample_history_candidate_features(
        prev_fpn.unsqueeze(1), history_grid, history_valid
    )
    center = model.correlation.center_candidate_index
    center_coords = geometry["current_feature_coords"][:, center].unsqueeze(0).to(device)
    with torch.no_grad():
        recovery = model.inference_forward(
            dirty_tokens,
            history_tokens,
            current_valid,
            history_valid,
            current_fpn,
            center_coords,
            current_valid[:, :, center],
        )
    return (
        _direct_from_fpn(runtime, current, recovery.recovered_current_fpn),
        recovery_diagnostics(recovery),
    )


def _progress(condition: str, done: int, total: int, started: float) -> None:
    elapsed = max(time.time() - started, 1e-9)
    rate = done / elapsed
    print(json.dumps({
        "event": "evaluation_progress",
        "condition": condition,
        "done": done,
        "target": total,
        "percent": 100.0 * done / max(total, 1),
        "rate_pairs_per_second": rate,
        "elapsed_seconds": elapsed,
        "eta_seconds": (total - done) / rate if rate > 0 else 0.0,
    }, sort_keys=True), flush=True)


def _run_records(
    label,
    records,
    dataset,
    token_index,
    runtime,
    geocorr,
    nuscenes_root,
    image_root,
    clean_current,
    mode,
    device,
    top_k,
    output_dir,
    resume,
    log_interval,
):
    tokens = current_tokens(records)
    progress_path = output_dir / "progress.pth"
    baseline_results, geocorr_results, diagnostics = [], [], []
    start_index = 0
    if resume:
        payload = load_progress(progress_path, label, tokens)
        if payload is not None:
            start_index = int(payload["next_index"])
            baseline_results = list(payload.get("baseline_results", []))
            geocorr_results = list(payload.get("geocorr_results", []))
            diagnostics = list(payload.get("diagnostics", []))

    started = time.time()
    for i in range(start_index, len(records)):
        record = records[i]
        prev_idx = token_index[str(record["previous_sample_token"])]
        cur_idx = token_index[str(record["current_sample_token"])]
        prev_paths = _resolved_paths(nuscenes_root, record["previous_clean_paths"])
        current_paths = _resolved_paths(
            nuscenes_root if clean_current else image_root,
            record["current_clean_paths"] if clean_current else record["current_dirty_paths"],
        )
        previous_data = _prepare_batch(dataset, prev_idx, prev_paths, record["previous_clean_paths"])
        current_data = _prepare_batch(dataset, cur_idx, current_paths, record["current_clean_paths"])
        current_result, current_fpn = runtime.forward(current_data, reset_memory=True)

        if mode in ("baseline", "both"):
            baseline_results.append(normalize_prediction(current_result[0]))
        if mode in ("geocorr", "both"):
            result, diag = _geocorr_prediction(
                record, dataset, token_index, runtime, geocorr,
                previous_data, current_data, current_fpn, device, top_k,
            )
            geocorr_results.append(result)
            diagnostics.append(diag)

        done = i + 1
        save_progress(progress_path, {
            "condition": label,
            "tokens": tokens,
            "next_index": done,
            "baseline_results": baseline_results,
            "geocorr_results": geocorr_results,
            "diagnostics": diagnostics,
        })
        if done % log_interval == 0 or done == len(records):
            _progress(label, done, len(records), started)

    expected = len(tokens)
    if mode in ("baseline", "both") and len(baseline_results) != expected:
        raise RuntimeError("baseline prediction count differs from token count")
    if mode in ("geocorr", "both") and len(geocorr_results) != expected:
        raise RuntimeError("GeoCorr prediction count differs from token count")
    return tokens, baseline_results, geocorr_results, diagnostics


def _compact(metrics):
    result = {"mean_ap": float(metrics["mean_ap"]), "nd_score": float(metrics["nd_score"])}
    for name in ("trans_err", "scale_err", "orient_err", "vel_err", "attr_err"):
        if name in metrics.get("tp_errors", {}):
            result[name] = float(metrics["tp_errors"][name])
    return result


def _score(dataset, metric, tokens, results, output_dir):
    path = format_subset_results(dataset, tokens, results, output_dir)
    return _compact(metric.evaluate(path, tokens, output_dir))


def run(args: argparse.Namespace) -> None:
    manifest = _resolve(args.manifest)
    nusc = _resolve(args.nuscenes_root)
    dirty = _resolve(args.dirty_root)
    config = _resolve(args.config)
    detector_ckpt = _resolve(args.detector_checkpoint)
    geocorr_ckpt = _resolve(args.geocorr_checkpoint) if args.geocorr_checkpoint else None
    out = _resolve(args.output_dir)
    ann = nusc / "nuscenes2d_temporal_infos_val.pkl"
    for path in [manifest, nusc, dirty, config, detector_ckpt, ann] + ([geocorr_ckpt] if geocorr_ckpt else []):
        if not path.exists():
            raise FileNotFoundError(str(path))
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)

    grouped = condition_records(read_manifest(manifest), args.conditions, args.max_pairs)
    cfg, dataset = build_streampetr_dataset(config, nusc, ann)
    token_index = _dataset_token_index(dataset)
    runtime = build_streampetr_model_runtime(cfg, detector_ckpt, device)
    detector_checksum = frozen_detector_parameter_checksum(runtime.detector)
    geocorr, geocorr_payload = None, None
    if geocorr_ckpt:
        geocorr, geocorr_payload = _load_geocorr(
            geocorr_ckpt, detector_checksum, device, args.top_k
        )
    metric = NuScenesSubsetMetric(nusc, dataset.version, dataset.eval_detection_configs)

    condition_metrics, recovery_summary = {}, {}
    for condition in sorted(grouped):
        condition_dir = out / "fault" / safe_condition_name(condition)
        tokens, baseline, recovered, diagnostics = _run_records(
            condition, grouped[condition], dataset, token_index, runtime, geocorr,
            nusc, dirty, False, args.mode, device, args.top_k, condition_dir,
            args.resume, args.log_interval,
        )
        condition_metrics[condition] = {}
        if args.mode in ("baseline", "both"):
            condition_metrics[condition]["baseline"] = _score(
                dataset, metric, tokens, baseline, condition_dir / "baseline"
            )
        if args.mode in ("geocorr", "both"):
            condition_metrics[condition]["geocorr"] = _score(
                dataset, metric, tokens, recovered, condition_dir / "geocorr"
            )
            recovery_summary[condition] = average_diagnostics(diagnostics)
        if args.mode == "both":
            b, g = condition_metrics[condition]["baseline"], condition_metrics[condition]["geocorr"]
            condition_metrics[condition]["delta"] = {
                "mean_ap": g["mean_ap"] - b["mean_ap"],
                "nd_score": g["nd_score"] - b["nd_score"],
            }
        write_json(out / "condition_metrics.json", condition_metrics)
        write_json(out / "severity_metrics.json", condition_metrics)
        write_json(out / "recovery_diagnostics.json", recovery_summary)

    clean_metrics = {}
    if args.evaluate_clean:
        clean_records = unique_records_by_current_token(grouped)
        clean_dir = out / "clean"
        tokens, baseline, recovered, diagnostics = _run_records(
            "clean", clean_records, dataset, token_index, runtime, geocorr,
            nusc, nusc, True, args.mode, device, args.top_k, clean_dir,
            args.resume, args.log_interval,
        )
        if args.mode in ("baseline", "both"):
            clean_metrics["baseline"] = _score(dataset, metric, tokens, baseline, clean_dir / "baseline")
        if args.mode in ("geocorr", "both"):
            clean_metrics["geocorr"] = _score(dataset, metric, tokens, recovered, clean_dir / "geocorr")
            clean_metrics["diagnostics"] = average_diagnostics(diagnostics)
        if args.mode == "both":
            clean_metrics["delta"] = {
                "mean_ap": clean_metrics["geocorr"]["mean_ap"] - clean_metrics["baseline"]["mean_ap"],
                "nd_score": clean_metrics["geocorr"]["nd_score"] - clean_metrics["baseline"]["nd_score"],
            }

    summary = {
        "protocol": "OccNuScenes 4-scene official-val subset",
        "subset_scene_count": len({str(r["scene_token"]) for rows in grouped.values() for r in rows}),
        "conditions": sorted(grouped),
        "condition_pair_counts": {name: len(rows) for name, rows in grouped.items()},
        "fault_averages": aggregate_fault_summary(condition_metrics),
        "clean": clean_metrics,
        "detector_checkpoint_sha256": _sha256(detector_ckpt),
        "detector_parameter_checksum": detector_checksum,
        "geocorr_checkpoint_sha256": _sha256(geocorr_ckpt) if geocorr_ckpt else None,
        "geocorr_step": int(geocorr_payload["step"]) if geocorr_payload else None,
        "geocorr_epoch": int(geocorr_payload["epoch"]) if geocorr_payload else None,
        "top_k": args.top_k,
        "mode": args.mode,
    }
    write_json(out / "summary.json", summary)
    print(json.dumps({"event": "stage4_complete", "summary": summary}, sort_keys=True), flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
