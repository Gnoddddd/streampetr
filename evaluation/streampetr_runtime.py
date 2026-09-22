"""Shared, checkpoint-matched StreamPETR runtime used by GeoCorr tools."""

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch
from torch import Tensor


STREAM_PETR_PER_CAMERA_FIELDS = (
    "img", "intrinsics", "extrinsics", "lidar2img", "img_timestamp",
)
STREAM_PETR_PER_BATCH_FIELDS = ("ego_pose", "ego_pose_inv", "timestamp", "prev_exists")
STREAM_PETR_REQUIRED_GEOMETRY_FIELDS = (
    "intrinsics", "extrinsics", "lidar2img", "ego_pose", "ego_pose_inv",
)


def _unwrap_data_container(value: Any) -> Any:
    """Unwrap MMCV-like payload wrappers without touching Tensor ``.data``."""
    while not isinstance(value, (Tensor, np.ndarray, Mapping, list, tuple)) and hasattr(
        value, "data"
    ):
        value = value.data
    return value


def _describe(value: Any) -> str:
    value = _unwrap_data_container(value)
    nested = type(value).__name__
    if isinstance(value, (list, tuple)) and value:
        nested += "[%s]" % type(_unwrap_data_container(value[0])).__name__
    shape = tuple(value.shape) if isinstance(value, (Tensor, np.ndarray)) else None
    device = str(value.device) if isinstance(value, Tensor) else "n/a"
    dtype = str(value.dtype) if isinstance(value, (Tensor, np.ndarray)) else "n/a"
    return "type=%s nested=%s shape=%s device=%s dtype=%s" % (
        type(value).__name__, nested, shape, device, dtype,
    )


def _batch_error(field: str, reason: str, value: Any) -> RuntimeError:
    return RuntimeError(
        "StreamPETR model batch contract failed for %s: %s; %s"
        % (field, reason, _describe(value))
    )


def _tensor_leaf(field: str, value: Any, device: torch.device) -> Tensor:
    value = _unwrap_data_container(value)
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(device)
    raise _batch_error(field, "expected Tensor or ndarray leaf", value)


def _pack_per_camera_tensor(
    field: str, value: Any, device: torch.device, camera_count: int
) -> Tensor:
    """Pack one B=1, V-camera input without tensorizing nested containers."""
    value = _unwrap_data_container(value)
    if isinstance(value, (Tensor, np.ndarray)):
        tensor = _tensor_leaf(field, value, device)
        if tensor.ndim >= 2 and tensor.shape[0] == 1 and tensor.shape[1] == camera_count:
            return tensor
        if tensor.ndim >= 1 and tensor.shape[0] == camera_count:
            return tensor.unsqueeze(0)
        raise _batch_error(
            field, "expected [V,...] or [1,V,...] with V=%d" % camera_count, value
        )
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _pack_per_camera_tensor(field, value[0], device, camera_count)
        if len(value) != camera_count:
            raise _batch_error(
                field, "expected singleton batch wrapper or %d cameras" % camera_count, value
            )
        leaves = [_tensor_leaf("%s[%d]" % (field, index), item, device)
                  for index, item in enumerate(value)]
        first_shape = leaves[0].shape
        if any(item.shape != first_shape for item in leaves[1:]):
            raise _batch_error(field, "camera leaf shape mismatch", value)
        return torch.stack(leaves, dim=0).unsqueeze(0)
    raise _batch_error(field, "unsupported per-camera container", value)


def _pack_per_batch_tensor(field: str, value: Any, device: torch.device) -> Tensor:
    """Pack one B=1 non-camera input while preserving its remaining dimensions."""
    value = _unwrap_data_container(value)
    if isinstance(value, (Tensor, np.ndarray)):
        tensor = _tensor_leaf(field, value, device)
        if tensor.ndim == 0:
            return tensor.reshape(1)
        return tensor if tensor.shape[0] == 1 else tensor.unsqueeze(0)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _pack_per_batch_tensor(field, value[0], device)
    raise _batch_error(field, "expected singleton batch tensor", value)


def _pack_img_metas(value: Any) -> List[Dict[str, Any]]:
    value = _unwrap_data_container(value)
    while isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(
        _unwrap_data_container(value[0]), (list, tuple)
    ):
        value = _unwrap_data_container(value[0])
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, (list, tuple)) and all(isinstance(item, Mapping) for item in value):
        return [dict(item) for item in value]
    raise _batch_error("img_metas", "expected list[dict] or dict", value)


def assert_streampetr_model_batch(
    batch: Mapping[str, Any], device: torch.device, camera_count: int = 6
) -> None:
    """Fail early with model-facing diagnostics before entering StreamPETR."""
    for field in STREAM_PETR_REQUIRED_GEOMETRY_FIELDS:
        if field not in batch:
            raise _batch_error(field, "missing required geometry field", None)
    for field in ("intrinsics", "extrinsics", "lidar2img"):
        value = batch[field]
        if not isinstance(value, Tensor):
            raise _batch_error(field, "must be Tensor after packing", value)
        if value.ndim < 3 or value.shape[0] != 1 or value.shape[1] != camera_count:
            raise _batch_error(
                field, "expected batch/camera dimensions [1,%d,...]" % camera_count, value
            )
        if value.device != device or not torch.is_floating_point(value):
            raise _batch_error(field, "requires floating Tensor on requested device", value)
    for field in ("ego_pose", "ego_pose_inv"):
        value = batch[field]
        if not isinstance(value, Tensor) or value.ndim < 2 or value.shape[0] != 1:
            raise _batch_error(field, "expected [1,...] Tensor", value)
        if value.device != device or not torch.is_floating_point(value):
            raise _batch_error(field, "requires floating Tensor on requested device", value)
    if "img_timestamp" in batch:
        value = batch["img_timestamp"]
        if not isinstance(value, Tensor) or value.ndim < 2 or value.shape[:2] != (
            1, camera_count
        ):
            raise _batch_error("img_timestamp", "expected [1,V,...] Tensor", value)
        if value.device != device:
            raise _batch_error("img_timestamp", "requires requested device", value)
    for field in ("timestamp", "prev_exists"):
        if field in batch:
            value = batch[field]
            if not isinstance(value, Tensor) or value.ndim < 1 or value.shape[0] != 1:
                raise _batch_error(field, "expected [1,...] Tensor", value)
            if value.device != device:
                raise _batch_error(field, "requires requested device", value)
    if "img" not in batch or not isinstance(batch["img"], Tensor):
        raise _batch_error("img", "must be Tensor after packing", batch.get("img"))
    if batch["img"].ndim < 3 or batch["img"].shape[:2] != (1, camera_count):
        raise _batch_error("img", "expected batch/camera dimensions", batch["img"])
    if batch["img"].device != device or not torch.is_floating_point(batch["img"]):
        raise _batch_error("img", "requires floating Tensor on requested device", batch["img"])
    if "img_metas" not in batch or not isinstance(batch["img_metas"], list) or not all(
        isinstance(item, dict) for item in batch["img_metas"]
    ):
        raise _batch_error("img_metas", "must remain list[dict]", batch.get("img_metas"))


def prepare_streampetr_model_batch(
    batch: Mapping[str, Any], device: torch.device, camera_count: int = 6
) -> Dict[str, Any]:
    """Turn one collated pipeline sample into the direct StreamPETR contract.

    The helper mirrors the B=1 stack performed by StreamPETR's sequence
    dataset for collect keys.  It leaves GT objects untouched so a caller can
    pass them to ``forward_pts_train`` with their official types intact.
    """
    result = dict(batch)
    for field in STREAM_PETR_PER_CAMERA_FIELDS:
        if field in result:
            result[field] = _pack_per_camera_tensor(
                field, result[field], device, camera_count
            )
    for field in STREAM_PETR_PER_BATCH_FIELDS:
        if field in result:
            result[field] = _pack_per_batch_tensor(field, result[field], device)
    if "img_metas" in result:
        result["img_metas"] = _pack_img_metas(result["img_metas"])
    assert_streampetr_model_batch(result, device, camera_count)
    return result


@dataclass
class StreamPETRRuntime:
    detector: Any
    parallel: Any
    captured_features: List[Tensor]

    def forward(self, data: Any, reset_memory: bool = False) -> Tuple[Any, Tensor]:
        """Run frozen inference and return the single captured CPFPN tensor."""
        if reset_memory:
            self.detector.pts_bbox_head.reset_memory()
            self.detector.prev_scene_token = None
        self.captured_features.clear()
        result = self.parallel(return_loss=False, rescale=True, **data)
        if len(self.captured_features) != 1:
            raise RuntimeError("one StreamPETR FPN capture expected per forward")
        return result, self.captured_features[0]


def build_streampetr_dataset(
    config_path: Path, data_root: Path, annotation: Path
) -> Tuple[Any, Any]:
    """Build the official test pipeline with explicit nuScenes paths."""
    from mmcv import Config
    from mmdet3d.datasets import build_dataset

    importlib.import_module("projects.mmdet3d_plugin")
    cfg = Config.fromfile(str(config_path))
    cfg.data.test.data_root = str(data_root) + "/"
    cfg.data.test.ann_file = str(annotation)
    cfg.data.test.test_mode = True
    return cfg, build_dataset(cfg.data.test)


def build_streampetr_model_runtime(
    cfg: Any, checkpoint_path: Path, device: torch.device
) -> StreamPETRRuntime:
    """Load the frozen official detector and attach its CPFPN capture hook."""
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet3d.models import build_model

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    detector = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16") is not None:
        wrap_fp16_model(detector)
    load_checkpoint(detector, str(checkpoint_path), map_location="cpu")
    detector.to(device).eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)

    captured: List[Tensor] = []
    original_extract = detector.extract_img_feat

    def capture_features(*feature_args: Any, **feature_kwargs: Any) -> Tensor:
        features = original_extract(*feature_args, **feature_kwargs)
        captured.append(features.detach())
        return features

    detector.extract_img_feat = capture_features
    device_ids = [device.index or 0] if device.type == "cuda" else []
    parallel = MMDataParallel(detector, device_ids=device_ids)
    return StreamPETRRuntime(detector, parallel, captured)


__all__ = [
    "StreamPETRRuntime",
    "assert_streampetr_model_batch",
    "build_streampetr_dataset",
    "build_streampetr_model_runtime",
    "prepare_streampetr_model_batch",
]
