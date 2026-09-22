#!/usr/bin/env python3
"""Short real-data forward-only smoke for GeoCorr Stage 3-B plumbing."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Sequence, Union

import torch

ROOT = Path(__file__).resolve().parents[1]
STREAM_ROOT = ROOT / "repos/StreamPETR"
for import_path in (ROOT, STREAM_ROOT, STREAM_ROOT / "mmdetection3d"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from configs.geocorr_recovery.stage1_infrastructure import candidate_offsets  # noqa: E402
from datasets.paired_occ_nuscenes import read_manifest  # noqa: E402
from datasets.temporal_occ_nuscenes import NUSCENES_CAMERA_ORDER  # noqa: E402
from evaluation.geocorr_stage3b_smoke import (  # noqa: E402
    descriptor_l2_mean,
    masked_statistics,
    probability_entropy_mean,
    probability_sum_error,
    select_manifest_pairs,
    tensor_nonfinite_counts,
)
from evaluation.streampetr_runtime import (  # noqa: E402
    StreamPETRRuntime,
    build_streampetr_dataset,
    build_streampetr_model_runtime,
)
from models.adapters import from_streampetr_result  # noqa: E402
from models.geocorr_recovery import (  # noqa: E402
    GeoCorrStage3BRecovery,
    GeometryCandidateSampler,
    ObjectCentricCorrelationField,
    correspondence_distillation,
    sample_candidate_features,
    sample_history_candidate_features,
    select_top_predictions,
)
from scripts.audit_geocorr_clean_correspondence import (  # noqa: E402
    _data_value,
    _image_shapes,
)


DEFAULT_CONFIG = "configs/full_nuscenes/stream_petr_r50_90e_clean_val.py"
DEFAULT_CHECKPOINT = "checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth"
PARALLEL_CAMERA_FIELDS = (
    "img_filename", "img_timestamp", "lidar2img", "intrinsics", "extrinsics"
)


def _parse_top_k(value: str) -> Union[int, str]:
    if value == "all":
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("top-k must be positive or 'all'") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("top-k must be positive or 'all'")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--nuscenes-root", required=True)
    parser.add_argument("--dirty-root", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--top-k", type=_parse_top_k, default=25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--identity-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if args.max_pairs <= 0:
        parser.error("--max-pairs must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.identity_tolerance < 0:
        parser.error("--identity-tolerance must be non-negative")
    return args


def _resolve(path: str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _camera_relative(path: str) -> str:
    parts = PurePosixPath(path.replace("\\", "/")).parts
    if "samples" not in parts:
        raise ValueError("dataset clean path has no samples component: %s" % path)
    index = parts.index("samples")
    return PurePosixPath(*parts[index:]).as_posix()


def _resolved_paths(root: Path, values: Sequence[object]) -> List[str]:
    if len(values) != len(NUSCENES_CAMERA_ORDER):
        raise ValueError("manifest image path list must contain exactly six cameras")
    paths = []
    for value in values:
        path = Path(str(value))
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file():
            raise FileNotFoundError(str(resolved))
        paths.append(str(resolved))
    return paths


def _prepare_batch(
    dataset: Any,
    dataset_index: int,
    image_paths: Sequence[str],
    expected_clean_paths: Sequence[object],
) -> Any:
    """Run the official deterministic test pipeline with manifest image paths."""
    from mmcv.parallel import collate

    info = dataset.data_infos[dataset_index]
    source_names = list(info["cams"].keys())
    if set(source_names) != set(NUSCENES_CAMERA_ORDER):
        raise ValueError("dataset info does not contain the canonical six cameras")
    input_dict = dataset.get_data_info(dataset_index)
    positions = [source_names.index(camera) for camera in NUSCENES_CAMERA_ORDER]
    source_clean = [input_dict["img_filename"][index] for index in positions]
    expected = [str(path) for path in expected_clean_paths]
    observed = [_camera_relative(str(path)) for path in source_clean]
    if observed != expected:
        raise ValueError("manifest clean paths do not match token-indexed dataset info")
    for field in PARALLEL_CAMERA_FIELDS:
        input_dict[field] = [input_dict[field][index] for index in positions]
    input_dict["img_filename"] = list(image_paths)
    dataset.pre_pipeline(input_dict)
    example = dataset.pipeline(input_dict)
    if example is None:
        raise RuntimeError("StreamPETR test pipeline returned no example")
    return collate([example], samples_per_gpu=1)


def _dataset_token_index(dataset: Any) -> Dict[str, int]:
    index: Dict[str, int] = {}
    for position, info in enumerate(dataset.data_infos):
        token = str(info["token"])
        if token in index:
            raise ValueError("duplicate token in StreamPETR annotation: %s" % token)
        index[token] = position
    return index


def _shape(name: str, tensor: torch.Tensor) -> None:
    print("%s %s" % (name, tuple(tensor.shape)), flush=True)


def _validate_forward_shapes(
    selected_count: int,
    channels: int,
    previous_fpn: torch.Tensor,
    current_clean_fpn: torch.Tensor,
    current_dirty_fpn: torch.Tensor,
    history_feature: torch.Tensor,
    q_clean: torch.Tensor,
    q_dirty: torch.Tensor,
    p_teacher: torch.Tensor,
    p_student: torch.Tensor,
    confidence: torch.Tensor,
    delta_q: torch.Tensor,
    q_recovered: torch.Tensor,
    recovered_fpn: torch.Tensor,
) -> None:
    views = len(NUSCENES_CAMERA_ORDER)
    candidates = len(candidate_offsets)
    expected_q = (1, selected_count, views, channels)
    expected_history = (1, selected_count, candidates, 1, views, channels)
    expected_probability = (1, selected_count, views, candidates, 1, views)
    expected_confidence = expected_q[:-1] + (1,)
    checks = {
        "q_clean": tuple(q_clean.shape) == expected_q,
        "q_dirty": tuple(q_dirty.shape) == expected_q,
        "history_feature": tuple(history_feature.shape) == expected_history,
        "p_teacher": tuple(p_teacher.shape) == expected_probability,
        "p_student": tuple(p_student.shape) == expected_probability,
        "confidence": tuple(confidence.shape) == expected_confidence,
        "delta_q": tuple(delta_q.shape) == expected_q,
        "q_recovered": tuple(q_recovered.shape) == expected_q,
        "recovered_fpn": recovered_fpn.shape == current_dirty_fpn.shape,
        "clean_dirty_fpn": current_clean_fpn.shape == current_dirty_fpn.shape,
        "previous_fpn": (
            previous_fpn.ndim == 5
            and previous_fpn.shape[0] == 1
            and previous_fpn.shape[1] == views
            and previous_fpn.shape[2] == channels
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("illegal Stage 3-B tensor shapes: %s" % failed)


def _run_pair(
    pair_index: int,
    record: Mapping[str, object],
    dataset: Any,
    token_index: Mapping[str, int],
    runtime: StreamPETRRuntime,
    nuscenes_root: Path,
    dirty_root: Path,
    top_k: Union[int, str],
    temperature: float,
) -> Dict[str, object]:
    previous_token = str(record["previous_sample_token"])
    current_token = str(record["current_sample_token"])
    if previous_token not in token_index or current_token not in token_index:
        raise KeyError("manifest sample token is absent from StreamPETR annotation")
    previous_index = token_index[previous_token]
    current_index = token_index[current_token]
    previous_info = dataset.data_infos[previous_index]
    current_info = dataset.data_infos[current_index]
    if str(current_info.get("prev", "")) != previous_token:
        raise ValueError("StreamPETR annotation current.prev does not match manifest")
    if (
        str(previous_info.get("scene_token")) != str(record["scene_token"])
        or str(current_info.get("scene_token")) != str(record["scene_token"])
        or int(previous_info["timestamp"]) != int(record["previous_timestamp"])
        or int(current_info["timestamp"]) != int(record["current_timestamp"])
    ):
        raise ValueError("manifest scene/timestamps do not match StreamPETR annotation")

    previous_clean_paths = _resolved_paths(
        nuscenes_root, record["previous_clean_paths"]
    )
    current_clean_paths = _resolved_paths(
        nuscenes_root, record["current_clean_paths"]
    )
    current_dirty_paths = _resolved_paths(
        dirty_root, record["current_dirty_paths"]
    )
    previous_data = _prepare_batch(
        dataset, previous_index, previous_clean_paths, record["previous_clean_paths"]
    )
    current_clean_data = _prepare_batch(
        dataset, current_index, current_clean_paths, record["current_clean_paths"]
    )
    current_dirty_data = _prepare_batch(
        dataset, current_index, current_dirty_paths, record["current_clean_paths"]
    )

    previous_result, previous_fpn = runtime.forward(previous_data, reset_memory=True)
    _, current_clean_fpn = runtime.forward(current_clean_data, reset_memory=True)
    _, current_dirty_fpn = runtime.forward(current_dirty_data, reset_memory=True)
    if not (
        previous_fpn.ndim == current_clean_fpn.ndim == current_dirty_fpn.ndim == 5
        and previous_fpn.shape[1] == len(NUSCENES_CAMERA_ORDER)
        and current_clean_fpn.shape == current_dirty_fpn.shape
    ):
        raise RuntimeError("illegal StreamPETR FPN shapes")

    prediction = from_streampetr_result(
        previous_result[0], float(previous_info["timestamp"]) / 1e6
    )
    prediction_count = int(prediction.score.numel())
    selected_prediction = select_top_predictions(prediction, top_k)
    selected_count = int(selected_prediction.score.numel())
    if selected_count == 0:
        raise RuntimeError("previous detector produced no selectable predictions")

    previous_meta = _data_value(previous_data, "img_metas")
    current_meta = _data_value(current_clean_data, "img_metas")
    current_from_previous = (
        _data_value(current_clean_data, "ego_pose_inv").float()
        @ _data_value(previous_data, "ego_pose").float()
    )
    delta_t = float(record["dt"])
    observed_dt = float(current_info["timestamp"] - previous_info["timestamp"]) / 1e6
    if delta_t <= 0 or abs(delta_t - observed_dt) > 1e-9:
        raise ValueError("manifest and StreamPETR annotation dt do not match")

    sampler = GeometryCandidateSampler(candidate_offsets)
    geometry = sampler.forward_with_history(
        selected_prediction,
        current_from_previous,
        delta_t,
        _data_value(current_clean_data, "lidar2img").float(),
        _image_shapes(current_meta, "img_shape"),
        tuple(current_clean_fpn.shape[-2:]),
        _data_value(previous_data, "lidar2img").float(),
        _image_shapes(previous_meta, "img_shape"),
        tuple(previous_fpn.shape[-2:]),
        current_padded_image_shapes=_image_shapes(current_meta, "pad_shape"),
        previous_padded_image_shapes=_image_shapes(previous_meta, "pad_shape"),
    )
    device = current_dirty_fpn.device
    current_grid = geometry["current_grid_coords"].unsqueeze(0).to(device)
    current_valid = geometry["current_valid_mask"].unsqueeze(0).to(device)
    history_grid = geometry["previous_grid_coords"].unsqueeze(0).to(device)
    history_valid = geometry["previous_valid_mask"].unsqueeze(0).to(device)
    current_clean_tokens = sample_candidate_features(
        current_clean_fpn, current_grid, current_valid
    )
    current_dirty_tokens = sample_candidate_features(
        current_dirty_fpn, current_grid, current_valid
    )
    history_tokens = sample_history_candidate_features(
        previous_fpn.unsqueeze(1), history_grid, history_valid
    )

    channels = int(current_dirty_fpn.shape[2])
    correlation = ObjectCentricCorrelationField(
        candidate_offsets, feature_dim=channels
    ).to(device)
    correlation_output = correlation(
        current_clean_tokens,
        current_dirty_tokens,
        history_tokens,
        current_valid,
        history_valid,
    )
    _, correspondence = correspondence_distillation(
        correlation_output["teacher_logits"],
        correlation_output["student_logits"],
        correlation_output["valid_mask"],
        temperature,
        correlation.center_candidate_index,
    )
    p_teacher = correspondence["teacher_probs"]
    p_student = correspondence["student_probs"]
    q_clean = correlation_output["clean_descriptor"]
    q_dirty = correlation_output["dirty_descriptor"]
    center = correlation.center_candidate_index
    center_coords = geometry["current_feature_coords"][:, center].unsqueeze(0).to(device)
    center_valid = current_valid[:, :, center]

    recovery = GeoCorrStage3BRecovery(
        feature_dim=channels, top_k=top_k, recovery_zero_init=True
    ).to(device)
    output = recovery(
        q_dirty=q_dirty,
        q_clean=q_clean,
        historical_features=correlation_output["history_descriptor"],
        p_student=p_student,
        correspondence_valid_mask=correlation_output["valid_mask"],
        p_teacher=p_teacher,
        previous_predictions=selected_prediction,
        current_clean_fpn=current_clean_fpn,
        current_dirty_fpn=current_dirty_fpn,
        projected_center_coords=center_coords,
        projected_center_valid=center_valid,
    )
    if output.recovered_current_fpn is None:
        raise RuntimeError("Stage 3-B did not return a recovered FPN")
    _validate_forward_shapes(
        selected_count,
        channels,
        previous_fpn,
        current_clean_fpn,
        current_dirty_fpn,
        correlation_output["history_descriptor"],
        q_clean,
        q_dirty,
        p_teacher,
        p_student,
        output.confidence,
        output.delta_q,
        output.q_recovered,
        output.recovered_current_fpn,
    )

    tensors = {
        "previous_fpn": previous_fpn,
        "current_clean_fpn": current_clean_fpn,
        "current_dirty_fpn": current_dirty_fpn,
        "q_clean": q_clean,
        "q_dirty": q_dirty,
        "p_teacher": p_teacher,
        "p_student": p_student,
        "retrieved": output.historical_retrieved_feature,
        "confidence": output.confidence,
        "delta_q": output.delta_q,
        "q_recovered": output.q_recovered,
        "recovered_fpn": output.recovered_current_fpn,
    }
    nan_count, inf_count = tensor_nonfinite_counts(tensors)
    query_valid = output.query_valid
    valid_count = int(query_valid.sum().item())
    total_count = int(query_valid.numel())
    confidence_stats = masked_statistics(output.confidence, query_valid)
    q_identity = float((output.q_recovered - q_dirty).abs().max().item())
    fpn_identity = float(
        (output.recovered_current_fpn - current_dirty_fpn).abs().max().item()
    )
    student_sum_error = probability_sum_error(
        output.normalized_student_probabilities, query_valid
    )
    retrieved_l2 = descriptor_l2_mean(
        output.historical_retrieved_feature, query_valid
    )
    q_dirty_l2 = descriptor_l2_mean(q_dirty, query_valid)
    q_clean_l2 = descriptor_l2_mean(q_clean, query_valid)
    teacher_entropy = probability_entropy_mean(p_teacher, query_valid)
    student_entropy = probability_entropy_mean(p_student, query_valid)

    print("PAIR %d" % pair_index, flush=True)
    print("condition %s" % record["raw_condition"])
    print("severity %s" % record.get("severity"))
    print("scene %s" % record["scene_name"])
    print("previous sample token %s" % previous_token)
    print("current sample token %s" % current_token)
    print("dt %.9f" % delta_t)
    _shape("previous FPN shape", previous_fpn)
    _shape("current clean FPN shape", current_clean_fpn)
    _shape("current dirty FPN shape", current_dirty_fpn)
    print("previous predictions count %d" % prediction_count)
    print("selected Top-K %d" % selected_count)
    _shape("q_clean shape", q_clean)
    _shape("q_dirty shape", q_dirty)
    _shape("historical feature shape", correlation_output["history_descriptor"])
    _shape("P_teacher shape", p_teacher)
    _shape("P_student shape", p_student)
    _shape("retrieved h shape", output.historical_retrieved_feature)
    _shape("confidence shape", output.confidence)
    _shape("delta_q shape", output.delta_q)
    _shape("q_recovered shape", output.q_recovered)
    _shape("recovered FPN shape", output.recovered_current_fpn)
    print("NaN count %d" % nan_count)
    print("Inf count %d" % inf_count)
    print("query_valid %d / %d" % (valid_count, total_count))
    print("query_valid %% %.6f" % (100.0 * valid_count / max(total_count, 1)))
    for name in ("min", "mean", "median", "max"):
        print("confidence %s %.9f" % (name, confidence_stats[name]))
    print("P_student sum error %.9g" % student_sum_error)
    print("teacher probability finite %s" % bool(torch.isfinite(p_teacher).all().item()))
    print("student probability finite %s" % bool(torch.isfinite(p_student).all().item()))
    print("teacher entropy mean %.9f" % teacher_entropy)
    print("student entropy mean %.9f" % student_entropy)
    print("retrieved_feature L2 mean %.9f" % retrieved_l2)
    print("q_dirty L2 mean %.9f" % q_dirty_l2)
    print("q_clean L2 mean %.9f" % q_clean_l2)
    print("max_abs(q_recovered - q_dirty) %.9g" % q_identity)
    print("max_abs(recovered_fpn - current_dirty_fpn) %.9g" % fpn_identity)
    return {
        "nan_count": nan_count,
        "inf_count": inf_count,
        "valid_count": valid_count,
        "query_count": total_count,
        "confidence_values": output.confidence.squeeze(-1).masked_select(query_valid).cpu(),
        "retrieved_norms": torch.linalg.norm(
            output.historical_retrieved_feature, dim=-1
        ).masked_select(query_valid).cpu(),
        "q_identity": q_identity,
        "fpn_identity": fpn_identity,
        "student_sum_error": student_sum_error,
    }


def _print_summary(
    requested: int,
    completed: int,
    top_k: Union[int, str],
    metrics: Sequence[Mapping[str, object]],
    failed: bool,
    identity_tolerance: float,
) -> bool:
    nan_count = sum(int(value["nan_count"]) for value in metrics)
    inf_count = sum(int(value["inf_count"]) for value in metrics)
    valid_count = sum(int(value["valid_count"]) for value in metrics)
    query_count = sum(int(value["query_count"]) for value in metrics)
    confidence_parts = [value["confidence_values"] for value in metrics]
    retrieved_parts = [value["retrieved_norms"] for value in metrics]
    confidence = torch.cat(confidence_parts) if confidence_parts else torch.empty(0)
    retrieved = torch.cat(retrieved_parts) if retrieved_parts else torch.empty(0)
    confidence_stats = (
        {
            "min": float(confidence.min().item()),
            "mean": float(confidence.mean().item()),
            "median": float(confidence.median().item()),
            "max": float(confidence.max().item()),
        }
        if confidence.numel()
        else {name: 0.0 for name in ("min", "mean", "median", "max")}
    )
    q_identity = max((float(value["q_identity"]) for value in metrics), default=0.0)
    fpn_identity = max((float(value["fpn_identity"]) for value in metrics), default=0.0)
    retrieved_l2 = float(retrieved.mean().item()) if retrieved.numel() else 0.0
    gate = bool(
        not failed
        and completed == requested
        and requested > 0
        and nan_count == 0
        and inf_count == 0
        and valid_count > 0
        and all(int(value["valid_count"]) > 0 for value in metrics)
        and q_identity <= identity_tolerance
        and fpn_identity <= identity_tolerance
    )
    print("PAIRS REQUESTED %d" % requested)
    print("PAIRS COMPLETED %d" % completed)
    print("TOP_K %s" % top_k)
    print("NAN_COUNT %d" % nan_count)
    print("INF_COUNT %d" % inf_count)
    print("QUERY_VALID_RATIO %.9f" % (valid_count / max(query_count, 1)))
    print("CONFIDENCE_MIN %.9f" % confidence_stats["min"])
    print("CONFIDENCE_MEAN %.9f" % confidence_stats["mean"])
    print("CONFIDENCE_MEDIAN %.9f" % confidence_stats["median"])
    print("CONFIDENCE_MAX %.9f" % confidence_stats["max"])
    print("RETRIEVED_L2_MEAN %.9f" % retrieved_l2)
    print("Q_IDENTITY_MAX_ABS_DIFF %.9g" % q_identity)
    print("FPN_IDENTITY_MAX_ABS_DIFF %.9g" % fpn_identity)
    print("GATE = %s" % ("PASS" if gate else "FAIL"))
    return gate


def run(args: argparse.Namespace) -> bool:
    manifest_path = _resolve(args.manifest)
    nuscenes_root = _resolve(args.nuscenes_root)
    dirty_root = _resolve(args.dirty_root)
    config_path = _resolve(args.config)
    checkpoint_path = _resolve(args.checkpoint)
    annotation = nuscenes_root / "nuscenes2d_temporal_infos_val.pkl"
    for required in (
        manifest_path, nuscenes_root, dirty_root, config_path, checkpoint_path, annotation
    ):
        if not required.exists():
            raise FileNotFoundError(str(required))
    records = select_manifest_pairs(
        read_manifest(manifest_path), args.conditions, args.max_pairs
    )
    print("CAMERA ORDER %s" % " ".join(NUSCENES_CAMERA_ORDER), flush=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        torch.cuda.set_device(device)
    cfg, dataset = build_streampetr_dataset(config_path, nuscenes_root, annotation)
    token_index = _dataset_token_index(dataset)
    runtime = build_streampetr_model_runtime(cfg, checkpoint_path, device)

    metrics: List[Mapping[str, object]] = []
    failed = False
    with torch.no_grad():
        for pair_index, record in enumerate(records):
            try:
                metrics.append(_run_pair(
                    pair_index,
                    record,
                    dataset,
                    token_index,
                    runtime,
                    nuscenes_root,
                    dirty_root,
                    args.top_k,
                    args.temperature,
                ))
            except Exception:
                failed = True
                print("PAIR FAILURE index=%d" % pair_index, flush=True)
                print("scene=%s" % record.get("scene_name"), flush=True)
                print("condition=%s" % record.get("raw_condition"), flush=True)
                print(
                    "sample_tokens=%s -> %s"
                    % (
                        record.get("previous_sample_token"),
                        record.get("current_sample_token"),
                    ),
                    flush=True,
                )
                traceback.print_exc()
                break
    return _print_summary(
        len(records), len(metrics), args.top_k, metrics, failed, args.identity_tolerance
    )


def main() -> None:
    try:
        passed = run(parse_args())
    except Exception:
        traceback.print_exc()
        print("GATE = FAIL")
        raise SystemExit(1)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
