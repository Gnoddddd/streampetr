#!/usr/bin/env python3
"""Precompute train-only per-GT LiDAR BEV teacher tokens."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Iterable, List

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos/StreamPETR"))
from models.object_evidence.privileged_geometry.cache import save_cache  # noqa: E402


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_teacher_feature(feature, centers, train_cfg):
    if isinstance(feature, (list, tuple)):
        if len(feature) != 1:
            raise RuntimeError("teacher head received multiple feature levels")
        feature = feature[0]
    if feature.ndim != 4:
        raise RuntimeError("teacher shared BEV feature must be [B,C,H,W]")
    _, _, height, width = feature.shape
    point_range = train_cfg["point_cloud_range"]
    voxel = train_cfg["voxel_size"]
    factor = float(train_cfg["out_size_factor"])
    x = (centers[:, 0] - float(point_range[0])) / (float(voxel[0]) * factor)
    y = (centers[:, 1] - float(point_range[1])) / (float(voxel[1]) * factor)
    nx = 2 * x / max(width - 1, 1) - 1 if width > 1 else x * 0
    ny = 2 * y / max(height - 1, 1) - 1 if height > 1 else y * 0
    grid = torch.stack((nx, ny), dim=-1).view(1, -1, 1, 2)
    return F.grid_sample(feature[:1], grid, align_corners=True)[0, :, :, 0].T


def _unwrap(value):
    while hasattr(value, "data") and not torch.is_tensor(value):
        value = value.data
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-config", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--split", required=True, choices=("train",))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmdet3d.datasets import build_dataloader, build_dataset
    from mmdet3d.models import build_model

    config = Config.fromfile(args.teacher_config)
    if config.get("custom_imports"):
        import_modules_from_strings(**config.custom_imports)
    dataset = build_dataset(config.data.train)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False
    )
    model = build_model(
        config.model, train_cfg=config.get("train_cfg"), test_cfg=config.get("test_cfg")
    )
    load_checkpoint(model, args.teacher_checkpoint, map_location="cpu", strict=False)
    model = MMDataParallel(model.to(args.device), device_ids=[int(args.device.split(":")[-1])])
    model.eval()
    head = model.module.pts_bbox_head
    captured: List[Any] = []
    handle = head.register_forward_pre_hook(
        lambda _module, arguments: captured.append(arguments[0])
    )
    records, scenes, feature_dim = [], set(), None
    try:
        for sample_index, data in enumerate(loader):
            captured.clear()
            with torch.no_grad():
                model(return_loss=True, **data)
            if len(captured) != 1:
                raise RuntimeError("expected one LiDAR-head input capture")
            boxes_object = _unwrap(data["gt_bboxes_3d"])
            boxes = boxes_object.tensor if hasattr(boxes_object, "tensor") else boxes_object
            centers = (
                boxes_object.gravity_center if hasattr(boxes_object, "gravity_center") else boxes[:, :3]
            ).to(args.device)
            tokens = sample_teacher_feature(captured[0], centers, head.train_cfg).detach().cpu().half()
            feature_dim = int(tokens.shape[1])
            metadata = _unwrap(data.get("img_metas", {}))
            metadata = metadata if isinstance(metadata, dict) else {}
            sample_token = str(metadata.get("sample_idx", sample_index))
            scene_token = str(metadata.get("scene_token", sample_token))
            scenes.add(scene_token)
            annotation_tokens = metadata.get("annotation_tokens", [])
            instance_tokens = metadata.get("instance_tokens", [])
            point_counts = _unwrap(data.get("num_lidar_pts", torch.zeros(len(tokens))))
            labels = _unwrap(data["gt_labels_3d"])
            for gt_index, token in enumerate(tokens):
                records.append(dict(
                    sample_token=sample_token,
                    annotation_token=str(annotation_tokens[gt_index]) if gt_index < len(annotation_tokens) else f"{sample_token}:{gt_index}",
                    instance_token=str(instance_tokens[gt_index]) if gt_index < len(instance_tokens) else f"{sample_token}:{gt_index}",
                    label=int(labels[gt_index]),
                    num_lidar_pts=int(point_counts[gt_index]),
                    teacher_token=token,
                ))
    finally:
        handle.remove()
    manifest = dict(
        teacher_config_sha256=sha256_file(args.teacher_config),
        teacher_checkpoint_sha256=sha256_file(args.teacher_checkpoint),
        feature_layer="pts_bbox_head.forward_pre_hook",
        feature_dim=int(feature_dim or 0), split="train",
        scene_count=len(scenes), sample_count=len(dataset),
    )
    save_cache(args.output, records, manifest)
    print(f"saved {len(records)} train object tokens to {args.output}")


if __name__ == "__main__":
    main()
