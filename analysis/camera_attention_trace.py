"""Opt-in read-only Camera-to-Query cross-attention trace.

Importing this module is a no-op unless ``CAMERA_ATTENTION_TRACE_DIR`` is set.
It wraps only the project-loaded StreamPETR head, registers temporary forward
hooks on the stock cross-attention modules, saves detached arrays, and returns
the exact original output object.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


def _cpu(value: torch.Tensor, dtype=None) -> np.ndarray:
    array = value.detach().cpu().numpy()
    return array.astype(dtype, copy=False) if dtype is not None else array


def _first(value: object, default: np.ndarray) -> np.ndarray:
    if not torch.is_tensor(value):
        return default.copy()
    tensor = value
    while tensor.ndim > default.ndim:
        tensor = tensor[0]
    return _cpu(tensor, default.dtype)


def _install_trace() -> None:
    trace_root = os.environ.get("CAMERA_ATTENTION_TRACE_DIR")
    if not trace_root:
        return

    from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
    from projects.mmdet3d_plugin.models.dense_heads.streampetr_head import (
        StreamPETRHead,
    )

    if getattr(StreamPETRHead, "_camera_attention_trace_installed", False):
        return
    output = Path(trace_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    original_forward = StreamPETRHead.forward

    def traced_forward(self, memory_center, img_metas, topk_indexes=None, **data):
        image_features = data["img_feats"]
        batch, camera_count, _, height, width = image_features.shape
        tokens_per_camera = int(height * width)
        if topk_indexes is None:
            token_camera = torch.arange(
                camera_count, device=image_features.device
            ).repeat_interleave(tokens_per_camera).unsqueeze(0).repeat(batch, 1)
        else:
            token_camera = topk_indexes.reshape(batch, -1).long()
            token_camera = torch.div(
                token_camera, tokens_per_camera, rounding_mode="floor"
            )

        captured: List[torch.Tensor] = []
        handles = []

        def capture_attention(_module, _inputs, result):
            weights = result[1] if isinstance(result, tuple) else None
            if weights is None:
                raise RuntimeError(
                    "camera reliability trace requires explicit attention weights"
                )
            if weights.ndim == 4:  # un-averaged heads
                weights = weights.mean(1)
            if weights.ndim != 3:
                raise RuntimeError(f"unexpected attention shape {tuple(weights.shape)}")
            if weights.shape[-1] != token_camera.shape[-1]:
                raise RuntimeError("cross-attention/token-source length mismatch")
            mass = weights.new_zeros(
                weights.shape[0], weights.shape[1], camera_count
            )
            source = token_camera[:, None, :].expand(
                weights.shape[0], weights.shape[1], -1
            )
            mass.scatter_add_(2, source, weights)
            captured.append(mass.detach())

        layers = self.transformer.decoder.layers
        for layer in layers:
            handles.append(layer.attentions[1].attn.register_forward_hook(capture_attention))
        try:
            result = original_forward(
                self, memory_center, img_metas, topk_indexes, **data
            )
        finally:
            for handle in handles:
                handle.remove()

        if len(captured) != len(layers):
            raise RuntimeError(
                f"captured {len(captured)} cross-attention layers, expected {len(layers)}"
            )
        layer_mass = torch.stack(captured)  # layer, batch, query, camera
        logits = result["all_cls_scores"][-1]
        raw_boxes = result["all_bbox_preds"][-1]
        max_num = int(self.bbox_coder.max_num)
        post_range = logits.new_tensor(self.bbox_coder.post_center_range)
        token_count = token_camera.shape[-1]

        for batch_index, meta in enumerate(img_metas):
            probabilities = logits[batch_index].sigmoid()
            scores, flat_index = probabilities.reshape(-1).topk(max_num)
            labels = flat_index.remainder(self.num_classes)
            query_index = torch.div(
                flat_index, self.num_classes, rounding_mode="floor"
            )
            boxes = denormalize_bbox(
                raw_boxes[batch_index, query_index], self.pc_range
            )
            valid = (boxes[:, :3] >= post_range[:3]).all(-1)
            valid &= (boxes[:, :3] <= post_range[3:]).all(-1)
            scores, labels = scores[valid], labels[valid]
            query_index, boxes = query_index[valid], boxes[valid]
            selected_mass = layer_mass[:, batch_index, query_index]
            all_mass = layer_mass[:, batch_index]
            selected_sources = token_camera[batch_index]
            token_share = torch.stack([
                (selected_sources == camera).float().sum() / max(token_count, 1)
                for camera in range(camera_count)
            ])

            sample_token = str(meta.get("sample_idx", meta.get("sample_token", "")))
            if not sample_token:
                raise KeyError("camera attention trace requires sample_idx")
            np.savez_compressed(
                output / f"{sample_token}.npz",
                sample_token=np.asarray(sample_token),
                scene_token=np.asarray(str(meta.get("scene_token", ""))),
                frame_idx=np.asarray(int(meta.get("frame_idx", -1))),
                timestamp=np.asarray(float(data["timestamp"][batch_index].item())),
                query_index=_cpu(query_index, np.int16),
                logits=_cpu(logits[batch_index, query_index], np.float16),
                boxes=_cpu(boxes, np.float32),
                scores=_cpu(scores, np.float32),
                labels=_cpu(labels, np.int16),
                deployed_camera_attention=_cpu(selected_mass, np.float32),
                all_query_camera_attention_mean=_cpu(all_mass.mean(1), np.float32),
                camera_token_share=_cpu(token_share, np.float32),
                camera_online=_first(
                    data.get("camera_online_mask"), np.ones(6, np.float32)
                ),
                camera_quality=_first(
                    data.get("camera_quality"), np.ones(6, np.float32)
                ),
                camera_fresh=_first(
                    data.get("camera_fresh_mask"), np.ones(6, np.float32)
                ),
                corruption_severity=_first(
                    data.get("corruption_severity"), np.zeros(6, np.float32)
                ),
            )
        return result

    StreamPETRHead.forward = traced_forward
    StreamPETRHead._camera_attention_trace_installed = True


_install_trace()

