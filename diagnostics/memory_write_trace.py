"""Opt-in, read-only tracing of stock StreamPETR Top-K memory writes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def install_memory_write_trace() -> None:
    output = os.environ.get("S3_R1_MEMORY_TRACE")
    if not output:
        return
    from projects.mmdet3d_plugin.models.dense_heads.streampetr_head import (
        StreamPETRHead,
    )

    if getattr(StreamPETRHead, "_s3_r1_trace_installed", False):
        return
    original_forward = StreamPETRHead.forward
    original_post = StreamPETRHead.post_update_memory
    trace_path = Path(output)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    def traced_forward(self, memory_center, img_metas, topk_indexes=None, **data):
        self._s3_r1_trace_metas = img_metas
        return original_forward(
            self, memory_center, img_metas, topk_indexes, **data
        )

    def traced_post(
        self,
        data,
        rec_ego_pose,
        all_cls_scores,
        all_bbox_preds,
        outs_dec,
        mask_dict,
    ):
        result = original_post(
            self,
            data,
            rec_ego_pose,
            all_cls_scores,
            all_bbox_preds,
            outs_dec,
            mask_dict,
        )
        with torch.no_grad():
            logits = all_cls_scores[-1]
            boxes = all_bbox_preds[-1]
            if mask_dict and mask_dict.get("pad_size", 0) > 0:
                pad = int(mask_dict["pad_size"])
                logits = logits[:, pad:]
                boxes = boxes[:, pad:]
            score, label = logits.sigmoid().max(dim=-1)
            indexes = score.topk(self.topk_proposals, dim=1).indices
            metas = getattr(self, "_s3_r1_trace_metas", [{}] * score.size(0))
            records = []
            for batch_index in range(score.size(0)):
                query_indexes = indexes[batch_index]
                meta = metas[batch_index]
                records.append(
                    {
                        "sample_token": meta.get("sample_idx"),
                        "scene_token": meta.get("scene_token"),
                        "frame_idx": int(meta.get("frame_idx", -1)),
                        "query_index": query_indexes.cpu().tolist(),
                        "score": score[batch_index, query_indexes].float().cpu().tolist(),
                        "class": label[batch_index, query_indexes].cpu().tolist(),
                        "center": boxes[batch_index, query_indexes, :3]
                        .float()
                        .cpu()
                        .tolist(),
                    }
                )
            with trace_path.open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        return result

    StreamPETRHead.forward = traced_forward
    StreamPETRHead.post_update_memory = traced_post
    StreamPETRHead._s3_r1_trace_installed = True

