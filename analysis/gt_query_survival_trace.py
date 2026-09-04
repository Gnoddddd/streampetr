"""Opt-in, read-only decoder/query survival trace for frozen B0."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


def _cpu(value: torch.Tensor, dtype=None) -> np.ndarray:
    array = value.detach().cpu().numpy()
    return array.astype(dtype, copy=False) if dtype is not None else array


def _first(value, default: np.ndarray) -> np.ndarray:
    if not torch.is_tensor(value):
        return default.copy()
    tensor = value
    while tensor.ndim > default.ndim:
        tensor = tensor[0]
    return _cpu(tensor, default.dtype)


def _install_trace() -> None:
    trace_root = os.environ.get("GT_QUERY_SURVIVAL_TRACE_DIR")
    if not trace_root:
        return
    from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
    from projects.mmdet3d_plugin.models.dense_heads.streampetr_head import StreamPETRHead

    if getattr(StreamPETRHead, "_gt_query_survival_trace_installed", False):
        return
    destination = Path(trace_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    original_forward = StreamPETRHead.forward

    def traced_forward(self, memory_center, img_metas, topk_indexes=None, **data):
        result = original_forward(self, memory_center, img_metas, topk_indexes, **data)
        features = data["img_feats"]
        batch, cameras, channels, height, width = features.shape
        flat_features = features.permute(0, 1, 3, 4, 2).reshape(
            batch, cameras * height * width, channels
        )
        if topk_indexes is None:
            selected_index = torch.arange(
                cameras * height * width, device=features.device
            ).unsqueeze(0).repeat(batch, 1)
        else:
            selected_index = topk_indexes.reshape(batch, -1).long()
        gather = selected_index.unsqueeze(-1).expand(-1, -1, channels)
        selected_norm = torch.gather(flat_features, 1, gather).float().norm(dim=-1)
        layer_logits = result["all_cls_scores"]
        raw_boxes = result["all_bbox_preds"]
        layers, _, queries = layer_logits.shape[:3]
        layer_boxes = denormalize_bbox(
            raw_boxes.reshape(layers * batch * queries, -1), self.pc_range
        ).reshape(layers, batch, queries, -1)
        for index, meta in enumerate(img_metas):
            sample_token = str(meta.get("sample_idx", meta.get("sample_token", "")))
            if not sample_token:
                raise KeyError("GT-query survival trace requires sample_idx")
            pad_shape = meta.get("pad_shape", meta.get("img_shape"))
            if isinstance(pad_shape, (list, tuple)) and pad_shape and isinstance(
                pad_shape[0], (list, tuple)
            ):
                pad_shape = pad_shape[0]
            image_hw = np.asarray(pad_shape[:2], dtype=np.int32)
            np.savez_compressed(
                destination / f"{sample_token}.npz",
                sample_token=np.asarray(sample_token),
                scene_token=np.asarray(str(meta.get("scene_token", ""))),
                frame_idx=np.asarray(int(meta.get("frame_idx", -1))),
                timestamp=np.asarray(float(data["timestamp"][index].item())),
                feature_hw=np.asarray([height, width], dtype=np.int32),
                image_hw=image_hw,
                selected_token_index=_cpu(selected_index[index], np.int32),
                selected_token_norm=_cpu(selected_norm[index], np.float32),
                layer_logits=_cpu(layer_logits[:, index], np.float16),
                layer_boxes=_cpu(layer_boxes[:, index], np.float32),
                lidar2img=_first(
                    data.get("lidar2img"), np.zeros((6, 4, 4), np.float32)
                ),
                camera_online=_first(
                    data.get("camera_online_mask"), np.ones(6, np.float32)
                ),
                camera_quality=_first(
                    data.get("camera_quality"), np.ones(6, np.float32)
                ),
                corruption_severity=_first(
                    data.get("corruption_severity"), np.zeros(6, np.float32)
                ),
            )
        return result

    StreamPETRHead.forward = traced_forward
    StreamPETRHead._gt_query_survival_trace_installed = True


_install_trace()

