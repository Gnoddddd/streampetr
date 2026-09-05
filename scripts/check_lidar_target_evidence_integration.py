#!/usr/bin/env python3
"""One-batch no-update integration check for the train-only privileged path."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

import evidence3d_plugin  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage4/lidar_privileged_target_evidence_audit.py"
CHECKPOINT = ROOT / "outputs/stage3/observability_distillation/b0/iter_969.pth"
OUTPUT = ROOT / "reports/stage4/lidar_privileged_target_evidence_audit/integration_check.csv"


def main():
    cfg = Config.fromfile(str(CONFIG))
    dataset = build_dataset(cfg.data.train)
    schedule = json.loads((ROOT / "protocols/feq_core/train_episode_seed314159.json").read_text())
    target_index = None
    for index, info in enumerate(dataset.data_infos):
        events = schedule["scenes"].get(info["scene_token"], [])
        for event in events:
            cameras = event.get("failed_cameras", []) + event.get("lost_cameras", [])
            if ("CAM_BACK" in cameras
                    and event["start_frame"] <= info["frame_idx"] <= event["end_frame"]):
                target_index = index
                break
        if target_index is not None:
            break
    if target_index is None:
        raise RuntimeError("no CAM_BACK fault sample in fixed train schedule")
    loader = build_dataloader(dataset, 1, 0, 1, dist=False, shuffle=False,
                              seed=2026, runner_type="IterBasedRunner")
    data = next(value for index, value in enumerate(loader) if index == target_index)
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(CHECKPOINT), map_location="cpu", strict=False)
    wrap_fp16_model(model)
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.train()
    with torch.set_grad_enabled(True):
        losses = model(return_loss=True, **data)
    keys = [name for name in losses if name.endswith("loss_lidar_target_evidence")]
    if len(keys) != 1:
        raise RuntimeError(f"missing privileged loss; keys={sorted(losses)}")
    key = keys[0]
    value = losses[key]
    diagnostics = model.module.pts_bbox_head._last_lidar_target_diagnostics
    rows = [{
        "dataset_index": target_index,
        "sample_token": dataset.data_infos[target_index]["token"],
        "scene_token": dataset.data_infos[target_index]["scene_token"],
        "frame_idx": dataset.data_infos[target_index]["frame_idx"],
        "loss_present": True,
        "loss_finite": bool(torch.isfinite(value)),
        "raw_loss": float(value.detach()),
        "diagnostic_gt": len(diagnostics),
        "selected_gt": sum(bool(item.get("selected")) for item in diagnostics),
        "all_selected_lidar_supported": all(
            item.get("lidar_supported") for item in diagnostics if item.get("selected")
        ),
        "optimizer_created": False,
        "optimizer_step": False,
    }]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    if not rows[0]["loss_finite"]:
        raise RuntimeError(rows)
    print(json.dumps(rows[0], indent=2))


if __name__ == "__main__":
    main()
