#!/usr/bin/env python3
"""Run the preregistered 20-batch, no-update FEQ loss-scale audit."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

import evidence3d_plugin  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage4/feq_f1.py")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.batches != 20:
        raise ValueError("FEQ preregistration requires exactly 20 batches")
    cfg = Config.fromfile(args.config)
    torch.manual_seed(int(cfg.seed)); torch.cuda.manual_seed_all(int(cfg.seed))
    dataset = build_dataset(cfg.data.train)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                              num_gpus=1, dist=False, shuffle=False,
                              seed=int(cfg.seed), runner_type="IterBasedRunner")
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"),
                        test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, cfg.load_from, map_location="cpu", strict=False)
    model = MMDataParallel(model.cuda(), device_ids=[0]); model.train()
    rows = []
    for index, data in enumerate(loader):
        if index >= args.batches: break
        model.module.zero_grad(set_to_none=True)
        with torch.no_grad():
            output = model.train_step(data, optimizer=None)
        logs = output["log_vars"]
        detection = sum(float(value) for key, value in logs.items()
                        if ("loss_cls" in key or "loss_bbox" in key)
                        and "feq" not in key)
        def value_named(suffix):
            matches = [float(value) for key, value in logs.items()
                       if key.endswith(suffix)]
            if len(matches) != 1:
                raise KeyError(f"expected one {suffix}, got keys={sorted(logs)}")
            return matches[0]
        row = {
            "batch": index, "detection_loss": detection,
            "raw_otm": value_named("feq_raw_otm"),
            "raw_rank": value_named("feq_raw_rank"),
            "raw_survival": value_named("feq_raw_survival"),
        }
        if not all(np.isfinite(value) and value >= 0 for key, value in row.items()
                   if key != "batch"):
            raise RuntimeError(f"negative/non-finite calibration loss: {row}")
        rows.append(row)
    if len(rows) != args.batches:
        raise RuntimeError(f"expected 20 batches, got {len(rows)}")
    med = {key: float(np.median([row[key] for row in rows]))
           for key in ("detection_loss", "raw_otm", "raw_rank", "raw_survival")}
    if not all(np.isfinite(value) and value > 0 for value in med.values()):
        raise RuntimeError(f"zero/non-finite median; calibration is invalid: {med}")
    targets = {"raw_otm": .08, "raw_rank": .04, "raw_survival": .04}
    weights = {key: med["detection_loss"] * ratio / med[key]
               for key, ratio in targets.items()}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        fieldnames = list(rows[0]) + ["row_type", "target_ratio", "lambda"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader()
        for row in rows: writer.writerow({**row, "row_type": "batch"})
        for key in targets:
            writer.writerow({"row_type": f"median_{key}", "detection_loss": med["detection_loss"],
                             key: med[key], "target_ratio": targets[key], "lambda": weights[key]})
    print("medians", med); print("lambdas", weights); print("total_target_ratio", sum(targets.values()))


if __name__ == "__main__":
    main()
