#!/usr/bin/env python
"""Run full-clean GeoCorr correspondence evaluation over nuScenes validation."""

import argparse
import gzip
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
STREAM_ROOT = ROOT / "repos/StreamPETR"
for path in (ROOT, STREAM_ROOT, STREAM_ROOT / "mmdetection3d"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.geocorr_recovery.stage1_infrastructure import (  # noqa: E402
    candidate_offsets,
    candidate_radius_m,
)
from evaluation.geocorr_full_evaluator import (  # noqa: E402
    EvaluationStore,
    atomic_write_json,
    build_frame_and_pair_plan,
    cache_rollover,
    compact_group_summary,
    compact_valid_view_summary,
    resume_frame_start,
)
from evaluation.streampetr_runtime import (  # noqa: E402
    build_streampetr_dataset,
    build_streampetr_model_runtime,
)
from models.adapters import from_streampetr_result  # noqa: E402
from models.geocorr_recovery import (  # noqa: E402
    GeometryCandidateSampler,
    sample_candidate_features,
    sample_history_candidate_features,
)
from scripts.audit_geocorr_clean_correspondence import (  # noqa: E402
    _audit_object_group,
    _data_value,
    _image_shapes,
)
from analysis.geocorr_correspondence_audit import (  # noqa: E402
    confidence_group_indices,
    summarize_valid_view_groups,
)


DEFAULT_CONFIG = "configs/full_nuscenes/stream_petr_r50_90e_clean_val.py"
DEFAULT_CHECKPOINT = "checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth"
QUARTILE_NAMES = {"q1_high": "q1", "q2": "q2", "q3": "q3", "q4_low": "q4"}


class RecoverablePairError(Exception):
    """A pair-local data condition that may be logged without stopping the run."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default="outputs/geocorr_full_clean")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--save-per-query", action="store_true")
    parser.add_argument("--data-root", default="data/nuscenes")
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()
    if args.max_pairs < 0:
        parser.error("--max-pairs must be zero or positive")
    if args.progress_interval <= 0:
        parser.error("--progress-interval must be positive")
    return args


def _resolve(path: str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git"] + list(arguments), cwd=str(ROOT), text=True
    ).strip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return "%02d:%02d:%02d" % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)


def _build_manifest(
    args: argparse.Namespace,
    config_path: Path,
    checkpoint_path: Path,
    data_root: Path,
    versions: Mapping[str, str],
) -> Dict[str, object]:
    device = torch.device(args.device)
    gpu_name = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
    return {
        "evaluation_name": "full clean correspondence evaluation",
        "git_branch": _git("branch", "--show-current"),
        "git_head": _git("rev-parse", "HEAD"),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "dataset_root": str(data_root),
        "split": "validation",
        "candidate_radius_m": candidate_radius_m,
        "temperature": args.temperature,
        "start_time": _now(),
        "finish_time": None,
        "cuda_device": args.device,
        "gpu_name": gpu_name,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "mmcv": versions["mmcv"],
        "mmdet": versions["mmdet"],
        "mmdet3d": versions["mmdet3d"],
        "command_line_args": vars(args),
        "invocations": [vars(args)],
        "random_seed": args.seed,
        "gt_used": False,
        "tracking_ids_used": False,
        "training": False,
    }


def _open_manifest(
    args: argparse.Namespace,
    output_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    data_root: Path,
    versions: Mapping[str, str],
) -> Dict[str, object]:
    path = output_dir / "run_manifest.json"
    if path.exists():
        if not args.resume:
            raise FileExistsError("run_manifest.json exists; use --resume or a new output")
        manifest = json.loads(path.read_text())
        expected = {
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "dataset_root": str(data_root),
            "cuda_device": args.device,
            "random_seed": args.seed,
            "temperature": args.temperature,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError("resume mismatch for %s" % key)
        manifest.setdefault("invocations", []).append(vars(args))
        manifest["finish_time"] = None
    else:
        if args.resume and any(output_dir.iterdir()) if output_dir.exists() else False:
            raise FileNotFoundError("cannot resume output without run_manifest.json")
        manifest = _build_manifest(
            args, config_path, checkpoint_path, data_root, versions
        )
    atomic_write_json(path, manifest)
    return manifest


def _collated(dataset, index: int):
    from mmcv.parallel import collate

    return collate([dataset[index]], samples_per_gpu=1)


def _frame_cache(frame, info, data, result, features) -> Dict[str, object]:
    meta = _data_value(data, "img_metas")
    timestamp = float(info["timestamp"]) / 1e6
    return {
        "frame": frame,
        "info": info,
        "meta": meta,
        "features": features,
        "prediction": from_streampetr_result(result[0], timestamp),
        "ego_pose": _data_value(data, "ego_pose").float(),
        "ego_pose_inv": _data_value(data, "ego_pose_inv").float(),
        "lidar2img": _data_value(data, "lidar2img").float(),
        "image_shapes": _image_shapes(meta, "img_shape"),
        "padded_shapes": _image_shapes(meta, "pad_shape"),
    }


def _evaluate_pair(
    previous: Mapping[str, object],
    current: Mapping[str, object],
    temperature: float,
) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    prediction = previous["prediction"]
    if prediction.center_3d.shape[0] == 0:
        raise RecoverablePairError("previous frame has no decoded predictions")
    current_from_previous = current["ego_pose_inv"] @ previous["ego_pose"]
    delta_t = (
        float(current["info"]["timestamp"] - previous["info"]["timestamp"]) / 1e6
    )
    if delta_t <= 0:
        raise RecoverablePairError("pair timestamp delta is not positive")
    sampler = GeometryCandidateSampler(candidate_offsets)
    geometry = sampler.forward_with_history(
        prediction,
        current_from_previous,
        delta_t,
        current["lidar2img"],
        current["image_shapes"],
        tuple(current["features"].shape[-2:]),
        previous["lidar2img"],
        previous["image_shapes"],
        tuple(previous["features"].shape[-2:]),
        current_padded_image_shapes=current["padded_shapes"],
        previous_padded_image_shapes=previous["padded_shapes"],
    )
    device = current["features"].device
    current_grid = geometry["current_grid_coords"].unsqueeze(0).to(device)
    current_valid = geometry["current_valid_mask"].unsqueeze(0).to(device)
    history_grid = geometry["previous_grid_coords"].unsqueeze(0).to(device)
    history_valid = geometry["previous_valid_mask"].unsqueeze(0).to(device)
    current_tokens = sample_candidate_features(
        current["features"], current_grid, current_valid
    )
    history_tokens = sample_history_candidate_features(
        previous["features"].unsqueeze(1), history_grid, history_valid
    )
    scores = prediction.score.detach().flatten()
    if scores.numel() != current_tokens.shape[1]:
        raise RuntimeError("systematic tensor-shape mismatch: score/object alignment")
    indices = confidence_group_indices(scores)
    group_summaries = {}
    all_metrics = None
    for name in ("all", "top25", "top50", "top100"):
        summary, metrics = _audit_object_group(
            indices[name], scores, current_tokens, history_tokens,
            current_valid, history_valid, temperature,
        )
        group_summaries[name] = compact_group_summary(summary)
        if name == "all":
            all_metrics = metrics
    for source_name, output_name in QUARTILE_NAMES.items():
        summary, _ = _audit_object_group(
            indices["quartiles"][source_name], scores, current_tokens,
            history_tokens, current_valid, history_valid, temperature,
        )
        group_summaries[output_name] = compact_group_summary(summary)
    raw_view_groups = summarize_valid_view_groups(all_metrics, candidate_offsets)
    valid_view_groups = {
        "valid_views_1": compact_valid_view_summary(raw_view_groups["1_view"]),
        "valid_views_2": compact_valid_view_summary(raw_view_groups["2_views"]),
        "valid_views_ge3": compact_valid_view_summary(
            raw_view_groups["3_or_more_views"]
        ),
    }
    pair_record = {
        "pair_key": "%s->%s" % (
            previous["frame"].sample_token, current["frame"].sample_token
        ),
        "scene_token": current["frame"].scene_token,
        "previous_sample_token": previous["frame"].sample_token,
        "current_sample_token": current["frame"].sample_token,
        "delta_t": delta_t,
        "num_predictions": int(scores.numel()),
        "num_valid_queries": group_summaries["all"]["num_valid_queries"],
        "num_same_view_queries": group_summaries["all"]["num_same_view_queries"],
        "groups": group_summaries,
        "valid_view_groups": valid_view_groups,
    }
    return pair_record, all_metrics


def _append_per_query(
    output_dir: Path,
    pair_record: Mapping[str, object],
    metrics: Mapping[str, torch.Tensor],
) -> None:
    valid = metrics["query_valid"]
    indices = valid.nonzero(as_tuple=False).cpu().tolist()
    fields = (
        "normalized_full_entropy",
        "normalized_position_entropy",
        "normalized_view_entropy",
        "position_top1_probability",
        "center_candidate_mass",
        "position_top1_minus_top2_margin",
    )
    with gzip.open(str(output_dir / "per_query.jsonl.gz"), "at") as handle:
        for batch, object_index, current_view in indices:
            record = {
                "pair_key": pair_record["pair_key"],
                "object_index": object_index,
                "current_view": current_view,
            }
            for field in fields:
                record[field] = float(
                    metrics[field][batch, object_index, current_view].item()
                )
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def _fatal_pair_exception(error: Exception) -> bool:
    if isinstance(error, RecoverablePairError):
        return False
    message = str(error).lower()
    fatal_terms = (
        "out of memory", "tensor-shape", "shape mismatch", "size mismatch",
        "non-finite", "nan", "inf", "mat1 and mat2",
    )
    return isinstance(error, (AssertionError, MemoryError)) or any(
        term in message for term in fatal_terms
    )


def _gpu_memory(device: torch.device) -> str:
    if device.type != "cuda":
        return "n/a"
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    return "%.1f/%.1f GiB" % (allocated, total)


def _print_progress(
    scene_number: int,
    scene_count: int,
    frame_number: int,
    frame_count: int,
    processed: int,
    total_pairs: int,
    started: float,
    device: torch.device,
) -> None:
    elapsed = time.time() - started
    rate = processed / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total_pairs - processed)
    eta = remaining / rate if rate > 0 else 0.0
    percent = 100.0 * processed / total_pairs if total_pairs else 100.0
    print(
        "[GeoCorr] scene %d/%d | frame %d/%d | pairs %d/%d | "
        "%.1f%% | elapsed %s | %.3f pair/s | ETA %s | VRAM %s"
        % (
            scene_number, scene_count, frame_number, frame_count,
            processed, total_pairs, percent, _elapsed(elapsed), rate,
            _elapsed(eta), _gpu_memory(device),
        ),
        flush=True,
    )


def run(args: argparse.Namespace) -> Dict[str, object]:
    if args.max_pairs < 0 or args.progress_interval <= 0:
        raise ValueError("invalid max-pairs or progress-interval")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)

    from mmcv import __version__ as mmcv_version
    from mmdet import __version__ as mmdet_version
    from mmdet3d import __version__ as mmdet3d_version
    config_path = _resolve(args.config)
    checkpoint_path = _resolve(args.checkpoint)
    data_root = _resolve(args.data_root)
    annotation = data_root / "nuscenes2d_temporal_infos_val.pkl"
    for required in (config_path, checkpoint_path, data_root, annotation):
        if not required.exists():
            raise FileNotFoundError(str(required))

    cfg, dataset = build_streampetr_dataset(config_path, data_root, annotation)
    frames, pairs = build_frame_and_pair_plan(dataset.data_infos)
    target_pairs = pairs[: args.max_pairs] if args.max_pairs else pairs
    target_keys = {pair.key for pair in target_pairs}
    scene_tokens = sorted({frame.scene_token for frame in frames})
    scene_numbers = {token: index + 1 for index, token in enumerate(scene_tokens)}

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    versions = {
        "mmcv": mmcv_version,
        "mmdet": mmdet_version,
        "mmdet3d": mmdet3d_version,
    }
    manifest = _open_manifest(
        args, output_dir, config_path, checkpoint_path, data_root, versions
    )
    manifest.update(
        dataset_frames=len(frames),
        derived_scenes=len(scene_tokens),
        derived_possible_pairs=len(pairs),
    )
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    store = EvaluationStore(output_dir, args.resume)
    # Required files exist even for a zero-pair or error-free run.
    if not store.pair_path.exists():
        with gzip.open(str(store.pair_path), "wb"):
            pass
    store.error_path.touch(exist_ok=True)

    completed_target = len(target_keys & store.processed_keys)
    if completed_target == len(target_pairs):
        summary = store.accumulator.summary()
        atomic_write_json(output_dir / "summary.json", summary)
        last_pair = target_pairs[-1] if target_pairs else None
        store.write_progress({
            "status": "complete", "processed_pairs": completed_target,
            "target_pairs": len(target_pairs),
            "last_sample": (
                last_pair.current.sample_token if last_pair is not None else None
            ),
            "last_scene": (
                last_pair.current.scene_token if last_pair is not None else None
            ),
            "updated_at": _now(),
        })
        manifest["finish_time"] = _now()
        manifest["status"] = "complete"
        manifest.setdefault("invocation_results", []).append({
            "finish_time": manifest["finish_time"],
            "detector_forwards": 0,
            "resume_short_circuit": True,
        })
        atomic_write_json(output_dir / "run_manifest.json", manifest)
        print("[GeoCorr] requested pairs already complete; no detector forward executed")
        return summary

    runtime = build_streampetr_model_runtime(cfg, checkpoint_path, device)
    pair_by_current = {pair.current.sample_token: pair for pair in target_pairs}
    start_index = resume_frame_start(frames, target_pairs, store.processed_keys)
    started = time.time()
    previous_cache: Optional[Mapping[str, object]] = None
    new_attempts = 0
    detector_forwards = 0

    with torch.no_grad():
        for frame_index in range(start_index, len(frames)):
            frame = frames[frame_index]
            if target_keys.issubset(store.processed_keys):
                break
            data = _collated(dataset, frame.dataset_index)
            result, features = runtime.forward(data)
            detector_forwards += 1
            info = dataset.data_infos[frame.dataset_index]
            current_cache = _frame_cache(frame, info, data, result, features)
            pair = pair_by_current.get(frame.sample_token)
            if pair is not None and pair.key not in store.processed_keys:
                if (
                    previous_cache is None
                    or previous_cache["frame"].sample_token
                    != pair.previous.sample_token
                ):
                    raise RuntimeError("systematic tensor-shape mismatch: cache/pair plan")
                try:
                    pair_record, all_metrics = _evaluate_pair(
                        previous_cache, current_cache, args.temperature
                    )
                    appended = store.append_pair(pair_record)
                    if appended and args.save_per_query:
                        _append_per_query(output_dir, pair_record, all_metrics)
                except Exception as error:
                    if _fatal_pair_exception(error):
                        raise
                    store.append_error({
                        "pair_key": pair.key,
                        "scene_token": frame.scene_token,
                        "sample_token": frame.sample_token,
                        "exception": "%s: %s" % (type(error).__name__, error),
                        "traceback": traceback.format_exc(),
                    })
                new_attempts += 1
                completed_target = len(target_keys & store.processed_keys)
                progress = {
                    "status": "running",
                    "processed_pairs": completed_target,
                    "successful_pairs": len(store.accumulator.records),
                    "target_pairs": len(target_pairs),
                    "last_scene": frame.scene_token,
                    "last_sample": frame.sample_token,
                    "updated_at": _now(),
                }
                store.write_progress(progress)
                if new_attempts % args.progress_interval == 0:
                    _print_progress(
                        scene_numbers[frame.scene_token], len(scene_tokens),
                        frame_index + 1, len(frames), completed_target,
                        len(target_pairs), started, device,
                    )
            previous_cache = cache_rollover(previous_cache, current_cache)

    completed_target = len(target_keys & store.processed_keys)
    if completed_target != len(target_pairs):
        raise RuntimeError(
            "evaluation stopped before all target pairs: %d/%d"
            % (completed_target, len(target_pairs))
        )
    summary = store.accumulator.summary()
    atomic_write_json(output_dir / "summary.json", summary)
    store.write_progress({
        "status": "complete",
        "processed_pairs": completed_target,
        "successful_pairs": len(store.accumulator.records),
        "target_pairs": len(target_pairs),
        "last_scene": target_pairs[-1].current.scene_token if target_pairs else None,
        "last_sample": target_pairs[-1].current.sample_token if target_pairs else None,
        "updated_at": _now(),
    })
    manifest["finish_time"] = _now()
    manifest["status"] = "complete"
    manifest["detector_forward_policy"] = "one forward per visited frame"
    manifest["runtime_seconds_this_invocation"] = time.time() - started
    if device.type == "cuda":
        manifest["peak_vram_gib"] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    manifest.setdefault("invocation_results", []).append({
        "finish_time": manifest["finish_time"],
        "runtime_seconds": manifest["runtime_seconds_this_invocation"],
        "detector_forwards": detector_forwards,
        "resume_short_circuit": False,
    })
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    _print_progress(
        scene_numbers[target_pairs[-1].current.scene_token] if target_pairs else 0,
        len(scene_tokens), frames.index(target_pairs[-1].current) + 1 if target_pairs else 0,
        len(frames), completed_target, len(target_pairs), started, device,
    )
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
