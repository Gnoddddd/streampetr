"""Opt-in, prediction-invariant trace for frozen StreamPETR inference.

Importing this module is a no-op unless ``COUNTERFACTUAL_TRACE_DIR`` is set.
The trace wraps the stock StreamPETR head in project space and saves only
detached inference tensors. It never registers state or changes model outputs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def _cpu(value: torch.Tensor, dtype=None) -> np.ndarray:
    array = value.detach().cpu().numpy()
    return array.astype(dtype, copy=False) if dtype is not None else array


def _first_tensor(data: Dict[str, Any], key: str, default) -> np.ndarray:
    value = data.get(key)
    if not torch.is_tensor(value):
        return np.asarray(default)
    while value.ndim > np.asarray(default).ndim:
        value = value[0]
    return _cpu(value)


def _install_trace() -> None:
    trace_root = os.environ.get("COUNTERFACTUAL_TRACE_DIR")
    if not trace_root:
        return

    from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
    from projects.mmdet3d_plugin.models.dense_heads.streampetr_head import (
        StreamPETRHead,
    )

    if getattr(StreamPETRHead, "_counterfactual_trace_installed", False):
        return

    output = Path(trace_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    trace_layers = os.environ.get("COUNTERFACTUAL_TRACE_LAYERS") == "1"
    original_forward = StreamPETRHead.forward

    def traced_forward(self, memory_center, img_metas, topk_indexes=None, **data):
        captured: Dict[str, torch.Tensor] = {}
        transformer_forward = self.transformer.forward

        def traced_transformer(*args, **kwargs):
            result = transformer_forward(*args, **kwargs)
            captured["query_feature"] = result[0]
            captured["memory_query"] = args[1]
            captured["memory_age"] = self.memory_timestamp
            return result

        self.transformer.forward = traced_transformer
        try:
            result = original_forward(
                self,
                memory_center,
                img_metas,
                topk_indexes,
                **data,
            )
        finally:
            self.transformer.forward = transformer_forward

        logits = result["all_cls_scores"][-1]
        raw_boxes = result["all_bbox_preds"][-1]
        query_feature = captured["query_feature"][-1]
        memory_query = captured["memory_query"]
        memory_age = captured["memory_age"]
        max_num = int(self.bbox_coder.max_num)
        post_range = logits.new_tensor(self.bbox_coder.post_center_range)
        base_queries = int(self.reference_points.weight.shape[0])

        for batch_index, meta in enumerate(img_metas):
            probabilities = logits[batch_index].sigmoid()
            scores, flat_index = probabilities.reshape(-1).topk(max_num)
            labels = flat_index.remainder(self.num_classes)
            query_index = torch.div(
                flat_index,
                self.num_classes,
                rounding_mode="floor",
            )
            boxes = denormalize_bbox(raw_boxes[batch_index, query_index], self.pc_range)
            valid = (boxes[:, :3] >= post_range[:3]).all(-1)
            valid &= (boxes[:, :3] <= post_range[3:]).all(-1)
            scores = scores[valid]
            labels = labels[valid]
            query_index = query_index[valid]
            boxes = boxes[valid]

            selected_query = query_feature[batch_index, query_index]
            selected_memory = memory_query[batch_index, query_index]
            selected_age = scores.new_zeros(scores.shape)
            propagated = query_index >= base_queries
            propagated_index = query_index[propagated] - base_queries
            propagated_index = propagated_index.clamp_max(
                max(int(memory_age.shape[1]) - 1, 0)
            )
            if propagated.any() and memory_age.shape[1] > 0:
                selected_age[propagated] = memory_age[
                    batch_index, propagated_index, 0
                ].to(selected_age)

            sample_token = str(meta.get("sample_idx", meta.get("sample_token", "")))
            if not sample_token:
                raise KeyError("counterfactual trace requires sample_idx")
            destination = output / f"{sample_token}.npz"
            payload = dict(
                sample_token=np.asarray(sample_token),
                scene_token=np.asarray(str(meta.get("scene_token", ""))),
                frame_idx=np.asarray(int(meta.get("frame_idx", -1))),
                timestamp=np.asarray(float(data["timestamp"][batch_index].item())),
                query_index=_cpu(query_index, np.int16),
                query_source=_cpu((query_index >= base_queries).to(torch.uint8)),
                query_feature=_cpu(selected_query, np.float16),
                memory_query=_cpu(selected_memory, np.float16),
                memory_age=_cpu(selected_age, np.float32),
                logits=_cpu(logits[batch_index, query_index], np.float16),
                boxes=_cpu(boxes, np.float32),
                scores=_cpu(scores, np.float32),
                labels=_cpu(labels, np.int16),
                camera_online=_first_tensor(
                    data, "camera_online_mask", np.ones(6, np.float32)
                ).astype(np.float32),
                camera_quality=_first_tensor(
                    data, "camera_quality", np.ones(6, np.float32)
                ).astype(np.float32),
                camera_fresh=_first_tensor(
                    data, "camera_fresh_mask", np.ones(6, np.float32)
                ).astype(np.float32),
                lidar2img=_first_tensor(
                    data, "lidar2img", np.zeros((6, 4, 4), np.float32)
                ).astype(np.float32),
                intrinsics=_first_tensor(
                    data, "intrinsics", np.zeros((6, 4, 4), np.float32)
                ).astype(np.float32),
                extrinsics=_first_tensor(
                    data, "extrinsics", np.zeros((6, 4, 4), np.float32)
                ).astype(np.float32),
            )
            if trace_layers:
                layer_logits = result["all_cls_scores"][:, batch_index]
                layer_raw_boxes = result["all_bbox_preds"][:, batch_index]
                layer_features = captured["query_feature"][:, batch_index]
                layer_count, query_count = layer_logits.shape[:2]
                layer_scores, layer_labels = layer_logits.sigmoid().max(-1)
                layer_boxes = denormalize_bbox(
                    layer_raw_boxes.reshape(-1, layer_raw_boxes.shape[-1]),
                    self.pc_range,
                ).reshape(layer_count, query_count, -1)
                payload.update(
                    decoder_layer=np.repeat(
                        np.arange(layer_count, dtype=np.int16), query_count
                    ),
                    lineage_query_index=np.tile(
                        np.arange(query_count, dtype=np.int16), layer_count
                    ),
                    layer_query_feature=_cpu(
                        layer_features.reshape(
                            layer_count * query_count, -1
                        ),
                        np.float16,
                    ),
                    layer_logits=_cpu(
                        layer_logits.reshape(
                            layer_count * query_count, self.num_classes
                        ),
                        np.float16,
                    ),
                    layer_boxes=_cpu(
                        layer_boxes.reshape(layer_count * query_count, -1),
                        np.float32,
                    ),
                    layer_scores=_cpu(
                        layer_scores.reshape(-1), np.float32
                    ),
                    layer_labels=_cpu(
                        layer_labels.reshape(-1), np.int16
                    ),
                )
            np.savez_compressed(destination, **payload)
        return result

    StreamPETRHead.forward = traced_forward
    StreamPETRHead._counterfactual_trace_installed = True


_install_trace()
