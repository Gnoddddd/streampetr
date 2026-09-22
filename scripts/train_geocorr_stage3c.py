#!/usr/bin/env python3
"""Train the GeoCorr Stage 3-C sidecar with a frozen StreamPETR detector."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import torch

ROOT = Path(__file__).resolve().parents[1]
STREAM_ROOT = ROOT / "repos/StreamPETR"
for import_path in (ROOT, STREAM_ROOT, STREAM_ROOT / "mmdetection3d"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from configs.geocorr_recovery.stage1_infrastructure import candidate_offsets  # noqa: E402
from datasets.paired_occ_nuscenes import read_manifest  # noqa: E402
from datasets.temporal_occ_nuscenes import NUSCENES_CAMERA_ORDER  # noqa: E402
from evaluation.geocorr_stage3b_smoke import select_manifest_pairs  # noqa: E402
from evaluation.streampetr_runtime import prepare_streampetr_model_batch  # noqa: E402
from models.adapters import from_streampetr_result  # noqa: E402
from models.geocorr_recovery import (  # noqa: E402
    GeometryCandidateSampler,
    sample_candidate_features,
    sample_history_candidate_features,
    select_top_predictions,
)
from scripts.audit_geocorr_clean_correspondence import _image_shapes  # noqa: E402
from training.geocorr_stage3c_trainer import (  # noqa: E402
    GeoCorrStage3CModel,
    JsonlLogger,
    Stage3CConfig,
    ZeroGradientMonitor,
    assert_finite_gradients,
    assert_finite_parameters,
    assert_finite_tensor,
    assert_synchronized_preprocessing,
    build_optimizer,
    compose_losses,
    deterministic_train_pipeline,
    forward_diagnostics,
    freeze_detector,
    frozen_detector_parameter_checksum,
    gpu_memory_megabytes,
    epoch_permutation,
    limit_pairs,
    load_checkpoint,
    module_grad_norm,
    now,
    parameter_report,
    parameter_update_norm,
    reached_max_steps,
    save_checkpoint,
    snapshot_parameters,
    streampetr_detection_losses,
)


DEFAULT_CONFIG = "configs/full_nuscenes/stream_petr_r50_90e_clean_val.py"
DEFAULT_CHECKPOINT = "checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth"
PARALLEL_CAMERA_FIELDS = (
    "img_filename", "img_timestamp", "lidar2img", "intrinsics", "extrinsics"
)
GT_FIELDS = (
    "gt_bboxes_3d", "gt_labels_3d", "gt_bboxes", "gt_labels", "centers2d", "depths"
)


def _parse_top_k(value: str) -> Union[int, str]:
    if value == "all":
        return value
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("top-k must be positive or 'all'")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--nuscenes-root", required=True)
    parser.add_argument("--dirty-root", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--top-k", type=_parse_top_k, default=25)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--lambda-corr", type=float, required=True)
    parser.add_argument("--lambda-rec", type=float, required=True)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    shuffle_group = parser.add_mutually_exclusive_group()
    shuffle_group.add_argument("--shuffle", dest="shuffle", action="store_true")
    shuffle_group.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.set_defaults(shuffle=True)
    parser.add_argument("--resume")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--zero-gradient-patience", type=int, default=10)
    args = parser.parse_args()
    positive = (
        "epochs", "grad_accum", "log_interval", "save_interval",
        "zero_gradient_patience",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error("--%s must be positive" % name.replace("_", "-"))
    for name in ("max_pairs", "max_steps"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error("--%s must be positive" % name.replace("_", "-"))
    if args.temperature <= 0 or args.lr <= 0:
        parser.error("--temperature and --lr must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    return args


def _resolve(path: str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _camera_relative(path: str) -> str:
    parts = PurePosixPath(path.replace("\\", "/")).parts
    if "samples" not in parts:
        raise ValueError("clean path has no samples component: %s" % path)
    return PurePosixPath(*parts[parts.index("samples"):]).as_posix()


def _resolved_paths(root: Path, values: Sequence[object]) -> List[str]:
    if len(values) != len(NUSCENES_CAMERA_ORDER):
        raise ValueError("manifest path list must contain the canonical six cameras")
    result = []
    for value in values:
        path = Path(str(value))
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file():
            raise FileNotFoundError(str(resolved))
        result.append(str(resolved))
    return result


def _build_dataset(config_path: Path, data_root: Path, annotation: Path) -> Tuple[Any, Any]:
    from mmcv import Config
    from mmdet3d.datasets import build_dataset

    importlib.import_module("projects.mmdet3d_plugin")
    cfg = Config.fromfile(str(config_path))
    train = cfg.data.train
    train.data_root = str(data_root) + "/"
    train.ann_file = str(annotation)
    train.test_mode = False
    train.queue_length = 1
    train.seq_mode = True
    train.num_frame_losses = 1
    train.pipeline = deterministic_train_pipeline(train.pipeline)
    return cfg, build_dataset(train)


def _build_detector(cfg: Any, checkpoint: Path, device: torch.device) -> Any:
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet3d.models import build_model

    cfg.model.pretrained = None
    detector = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16") is not None:
        wrap_fp16_model(detector)
    load_checkpoint(detector, str(checkpoint), map_location="cpu")
    detector.to(device).eval()
    freeze_detector(detector)
    return detector


def _dataset_token_index(dataset: Any) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for index, info in enumerate(dataset.data_infos):
        token = str(info["token"])
        if token in result:
            raise ValueError("duplicate token in StreamPETR annotation: %s" % token)
        result[token] = index
    return result


def _prepare_example(
    dataset: Any,
    index: int,
    image_paths: Sequence[str],
    expected_clean_paths: Sequence[object],
) -> Any:
    info = dataset.data_infos[index]
    source_names = list(info["cams"].keys())
    positions = [source_names.index(name) for name in NUSCENES_CAMERA_ORDER]
    input_dict = dataset.get_data_info(index)
    observed = [
        _camera_relative(str(input_dict["img_filename"][position]))
        for position in positions
    ]
    if observed != [str(path) for path in expected_clean_paths]:
        raise ValueError("manifest clean paths do not match token-indexed annotation")
    for field in PARALLEL_CAMERA_FIELDS:
        input_dict[field] = [input_dict[field][position] for position in positions]
    input_dict["img_filename"] = list(image_paths)
    dataset.pre_pipeline(input_dict)
    example = dataset.pipeline(input_dict)
    if example is None:
        raise RuntimeError("deterministic StreamPETR train pipeline returned no example")
    return example


def _materialize(example: Any, device: torch.device) -> Mapping[str, Any]:
    """Apply the same MMCV collate/scatter boundary as the Stage 3-B runtime."""
    from mmcv.parallel import collate, scatter

    batch = collate([example], samples_per_gpu=1)
    if device.type == "cuda":
        return scatter(batch, [device.index or 0])[0]

    # CPU is intended for synthetic/interface debugging only.  Mirror MMCV's
    # one-sample DataContainer unwrapping without invoking its CUDA scatter.
    from mmcv.parallel import DataContainer

    def unwrap(value: Any) -> Any:
        if isinstance(value, DataContainer):
            raw = value.data
            while isinstance(raw, list) and len(raw) == 1:
                raw = raw[0]
            return unwrap(raw)
        if isinstance(value, Mapping):
            return {key: unwrap(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(unwrap(item) for item in value)
        if isinstance(value, list):
            return [unwrap(item) for item in value]
        return value.to(device) if isinstance(value, torch.Tensor) else value

    return unwrap(batch)


def _detached_fpn(detector: Any, batch: Mapping[str, Any]) -> torch.Tensor:
    detector.eval()
    with torch.no_grad():
        feature = detector.extract_img_feat(batch["img"], 1, False)
    return feature.detach()


def _previous_prediction(
    detector: Any, batch: Mapping[str, Any], fpn: torch.Tensor, timestamp: float
) -> Any:
    detector.eval()
    detector.pts_bbox_head.reset_memory()
    detector.prev_scene_token = None
    data = {key: value for key, value in batch.items() if key not in GT_FIELDS and key != "img_metas"}
    data["img_feats"] = fpn
    with torch.no_grad():
        result = detector.simple_test_pts(batch["img_metas"], **data)
    return from_streampetr_result(result[0], timestamp)


def _paired_batches(
    record: Mapping[str, object], dataset: Any, token_index: Mapping[str, int],
    nuscenes_root: Path, dirty_root: Path, device: torch.device,
) -> Tuple[int, int, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    previous_token = str(record["previous_sample_token"])
    current_token = str(record["current_sample_token"])
    previous_index, current_index = token_index[previous_token], token_index[current_token]
    if str(dataset.data_infos[current_index].get("prev", "")) != previous_token:
        raise ValueError("annotation current.prev does not match manifest")
    previous_raw = _materialize(_prepare_example(
        dataset, previous_index,
        _resolved_paths(nuscenes_root, record["previous_clean_paths"]),
        record["previous_clean_paths"],
    ), device)
    clean_raw = _materialize(_prepare_example(
        dataset, current_index,
        _resolved_paths(nuscenes_root, record["current_clean_paths"]),
        record["current_clean_paths"],
    ), device)
    dirty_raw = _materialize(_prepare_example(
        dataset, current_index,
        _resolved_paths(dirty_root, record["current_dirty_paths"]),
        record["current_clean_paths"],
    ), device)
    assert_synchronized_preprocessing(clean_raw, dirty_raw)
    previous = prepare_streampetr_model_batch(previous_raw, device)
    clean = prepare_streampetr_model_batch(clean_raw, device)
    dirty = prepare_streampetr_model_batch(dirty_raw, device)
    return previous_index, current_index, previous, clean, dirty


def _stage3_forward(
    model: GeoCorrStage3CModel,
    detector: Any,
    record: Mapping[str, object],
    previous_info: Mapping[str, Any],
    current_info: Mapping[str, Any],
    previous: Mapping[str, Any],
    clean: Mapping[str, Any],
    dirty: Mapping[str, Any],
    top_k: Union[int, str],
) -> Any:
    previous_fpn = _detached_fpn(detector, previous)
    clean_fpn = _detached_fpn(detector, clean)
    dirty_fpn = _detached_fpn(detector, dirty)
    prediction = select_top_predictions(_previous_prediction(
        detector, previous, previous_fpn, float(previous_info["timestamp"]) / 1e6
    ), top_k)
    if prediction.score.numel() == 0:
        raise RuntimeError("previous detector produced no selectable predictions")
    current_from_previous = (
        clean["ego_pose_inv"][0].float() @ previous["ego_pose"][0].float()
    )
    dt = float(record["dt"])
    observed_dt = float(current_info["timestamp"] - previous_info["timestamp"]) / 1e6
    if dt <= 0 or abs(dt - observed_dt) > 1e-9:
        raise ValueError("manifest and annotation dt do not match")
    geometry = GeometryCandidateSampler(candidate_offsets).forward_with_history(
        prediction,
        current_from_previous,
        dt,
        clean["lidar2img"][0].float(),
        _image_shapes(clean["img_metas"][0], "img_shape"),
        tuple(clean_fpn.shape[-2:]),
        previous["lidar2img"][0].float(),
        _image_shapes(previous["img_metas"][0], "img_shape"),
        tuple(previous_fpn.shape[-2:]),
        current_padded_image_shapes=_image_shapes(clean["img_metas"][0], "pad_shape"),
        previous_padded_image_shapes=_image_shapes(previous["img_metas"][0], "pad_shape"),
    )
    device = dirty_fpn.device
    current_grid = geometry["current_grid_coords"].unsqueeze(0).to(device)
    current_valid = geometry["current_valid_mask"].unsqueeze(0).to(device)
    history_grid = geometry["previous_grid_coords"].unsqueeze(0).to(device)
    history_valid = geometry["previous_valid_mask"].unsqueeze(0).to(device)
    clean_tokens = sample_candidate_features(clean_fpn, current_grid, current_valid)
    dirty_tokens = sample_candidate_features(dirty_fpn, current_grid, current_valid)
    history_tokens = sample_history_candidate_features(
        previous_fpn.unsqueeze(1), history_grid, history_valid
    )
    center = model.correlation.center_candidate_index
    center_coords = geometry["current_feature_coords"][:, center].unsqueeze(0).to(device)
    return model(
        clean_tokens, dirty_tokens, history_tokens, current_valid, history_valid,
        dirty_fpn, center_coords, current_valid[:, :, center], clean_fpn,
    )


def _print_parameter_report(report: Mapping[str, Any]) -> None:
    print("TRAINABLE PARAMETERS")
    for name in report["trainable_parameters"]:
        print(name)
    print("TRAINABLE MODULES")
    for name in report["trainable_modules"]:
        print(name)
    print("FROZEN PARAMETERS")
    for name in report["frozen_parameters"]:
        print(name)
    print("TRAINABLE PARAMETER COUNT %d" % report["trainable_parameter_count"], flush=True)


def run(args: argparse.Namespace) -> None:
    paths = {
        "manifest": _resolve(args.train_manifest),
        "nuscenes": _resolve(args.nuscenes_root),
        "dirty": _resolve(args.dirty_root),
        "config": _resolve(args.config),
        "checkpoint": _resolve(args.checkpoint),
        "output": _resolve(args.output_dir),
    }
    annotation = paths["nuscenes"] / "nuscenes2d_temporal_infos_train.pkl"
    for path in (paths["manifest"], paths["nuscenes"], paths["dirty"], paths["config"], paths["checkpoint"], annotation):
        if not path.exists():
            raise FileNotFoundError(str(path))
    paths["output"].mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but unavailable")
        torch.cuda.set_device(device)
    cfg, dataset = _build_dataset(paths["config"], paths["nuscenes"], annotation)
    detector = _build_detector(cfg, paths["checkpoint"], device)
    manifest_records = read_manifest(paths["manifest"])
    records = limit_pairs(select_manifest_pairs(
        manifest_records, args.conditions, len(manifest_records)
    ), args.max_pairs)
    token_index = _dataset_token_index(dataset)
    model_config = Stage3CConfig(
        feature_dim=256,
        candidate_offsets=tuple(tuple(value) for value in candidate_offsets),
        top_k=args.top_k,
        temperature=args.temperature,
        lambda_corr=args.lambda_corr,
        lambda_rec=args.lambda_rec,
    )
    model = GeoCorrStage3CModel(model_config).to(device).train()
    optimizer = build_optimizer(detector, model, args.lr)
    report = parameter_report(detector, model)
    _print_parameter_report(report)
    detector_checksum = frozen_detector_parameter_checksum(detector)
    step, start_epoch, start_pair = 0, 0, 0
    if args.resume:
        step, start_epoch, start_pair = load_checkpoint(
            _resolve(args.resume), model, optimizer, detector_checksum
        )
    logger = JsonlLogger(paths["output"] / "train.jsonl")
    zero_monitor = ZeroGradientMonitor(args.zero_gradient_patience)
    optimizer.zero_grad()
    stop = False
    accumulated = 0
    resume_epoch, resume_pair = start_epoch, start_pair
    for epoch in range(start_epoch, args.epochs):
        order = epoch_permutation(len(records), args.seed, epoch, args.shuffle)
        first_pair = start_pair if epoch == start_epoch else 0
        for pair_position in range(first_pair, len(order)):
            if reached_max_steps(step, args.max_steps):
                stop = True
                break
            started = now()
            record_index = order[pair_position]
            record = records[record_index]
            previous_index, current_index, previous, clean, dirty = _paired_batches(
                record, dataset, token_index, paths["nuscenes"], paths["dirty"], device
            )
            output = _stage3_forward(
                model, detector, record, dataset.data_infos[previous_index],
                dataset.data_infos[current_index], previous, clean, dirty, args.top_k,
            )
            recovered = output.recovery.recovered_current_fpn
            if recovered is None:
                raise RuntimeError("Stage 3-C produced no recovered FPN")
            assert_finite_tensor("recovered feature", recovered)
            detector.train()
            detector.pts_bbox_head.reset_memory()
            detector.prev_scene_token = None
            official = streampetr_detection_losses(detector, recovered, dirty)
            losses = compose_losses(
                official, output.loss_corr, output.loss_rec,
                args.lambda_corr, args.lambda_rec,
            )
            assert_finite_tensor("loss_total", losses.total)
            (losses.total / args.grad_accum).backward()
            assert_finite_gradients(model, "GeoCorr")
            accumulated += 1
            do_update = accumulated == args.grad_accum or pair_position + 1 == len(order)
            if not do_update:
                continue
            descriptor_before = snapshot_parameters(model.correlation.adapter)
            recovery_before = snapshot_parameters(model.recovery.recovery)
            descriptor_grad = module_grad_norm(model.correlation.adapter)
            recovery_grad = module_grad_norm(model.recovery.recovery)
            zero_monitor.update({
                "descriptor_adapter": descriptor_grad,
                "recovery_mlp": recovery_grad,
            })
            optimizer.step()
            assert_finite_parameters(model, "GeoCorr")
            descriptor_update = parameter_update_norm(
                model.correlation.adapter, descriptor_before
            )
            recovery_update = parameter_update_norm(
                model.recovery.recovery, recovery_before
            )
            optimizer.zero_grad()
            accumulated = 0
            step += 1
            resume_epoch, resume_pair = epoch, pair_position + 1
            if resume_pair == len(order):
                resume_epoch, resume_pair = epoch + 1, 0
            diagnostics = forward_diagnostics(output)
            last_layer = model.recovery.recovery.layers[-1]
            log = {
                "step": step,
                "epoch": epoch,
                "pair_index": pair_position,
                "manifest_index": record_index,
                "shuffle": bool(args.shuffle),
                "shuffle_seed": int(args.seed),
                "sample_token": record["current_sample_token"],
                "loss_total": float(losses.total.detach().item()),
                "loss_det": float(losses.detection.detach().item()),
                "loss_corr": float(losses.correspondence.detach().item()),
                "loss_rec": float(losses.recovery.detach().item()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "grad_norm_descriptor_adapter": descriptor_grad,
                "grad_norm_recovery_mlp": recovery_grad,
                "parameter_update_norm_descriptor_adapter": descriptor_update,
                "parameter_update_norm_recovery_mlp": recovery_update,
                "recovery_output_weight_norm": float(last_layer.weight.detach().float().norm().item()),
                "recovery_output_bias_norm": float(last_layer.bias.detach().float().norm().item()),
                "gpu_memory_mb": gpu_memory_megabytes(device),
                "step_time_seconds": now() - started,
            }
            log.update(diagnostics)
            log["detection_components"] = {
                name: float(value.detach().item())
                for name, value in losses.detection_components.items()
            }
            if step % args.log_interval == 0:
                logger.log(log)
                print(json.dumps(log, sort_keys=True), flush=True)
            if step % args.save_interval == 0:
                save_checkpoint(
                    paths["output"] / ("step_%08d.pth" % step), model, optimizer,
                    step, resume_epoch, model_config, detector_checksum, resume_pair,
                )
        if stop:
            break
        start_pair = 0
    save_checkpoint(
        paths["output"] / "latest.pth", model, optimizer, step,
        resume_epoch, model_config, detector_checksum, resume_pair,
    )
    if frozen_detector_parameter_checksum(detector) != detector_checksum:
        raise RuntimeError("frozen StreamPETR parameters changed during training")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
