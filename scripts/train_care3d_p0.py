#!/usr/bin/env python3
"""Train only the CARE-3D P0 vulnerability predictor on frozen exports."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from analysis.care3d_counterfactual import PROTOCOLS
from models.care3d import CARE3DCore, CounterfactualVulnerabilityLoss


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/care3d/p0_counterfactual_vulnerability"
CONFIG = ROOT / "configs/care3d/p0_counterfactual_vulnerability.py"
SCHEMA = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(42, 2027, 2028), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_split(split: str) -> dict[str, np.ndarray]:
    if split not in {"probe_train", "probe_val"}:
        raise ValueError("trainer may read only probe_train/probe_val")
    manifest = pd.read_csv(REPORT / "frozen_scene_manifest.csv")
    scenes = manifest[manifest.split == split].scene_token.astype(str).tolist()
    arrays = {key: [] for key in (
        "object_features", "temporal_features", "decision_features", "camera_support",
        "camera_quality", "evidence_drop", "cross_topk", "valid_mask",
    )}
    for scene in scenes:
        prefix = REPORT / "incremental/P0" / scene
        marker = prefix.with_suffix(".complete.json")
        packed_path = prefix.with_suffix(".features.npz")
        if not marker.exists() or not packed_path.exists():
            raise RuntimeError(f"formal CARE extraction incomplete: {split}/{scene}")
        meta = json.loads(marker.read_text())
        if not meta.get("complete") or meta.get("schema_version") != SCHEMA:
            raise RuntimeError(f"invalid scene completion marker: {scene}")
        packed = np.load(packed_path)
        n = int(packed["object_features"].shape[0])
        expected = {
            "object_features": (n, 256), "temporal_features": (n, 256),
            "decision_features": (n, 21), "camera_support": (n, 6),
            "camera_quality": (n, 6), "evidence_drop": (n, 3),
            "cross_topk": (n, 3), "valid_mask": (n, 3),
        }
        for key, shape in expected.items():
            value = np.asarray(packed[key])
            if value.shape != shape:
                raise RuntimeError(f"{scene}: {key} shape {value.shape} != {shape}")
            arrays[key].append(value)
    result = {}
    for key, values in arrays.items():
        if not values:
            raise RuntimeError(f"no arrays for {split}/{key}")
        result[key] = np.concatenate(values, axis=0)
    if len(result["object_features"]) == 0:
        raise RuntimeError(f"empty CARE split: {split}")
    return result


class ArrayDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray]):
        self.arrays = arrays
        self.length = len(arrays["object_features"])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {key: torch.from_numpy(np.asarray(value[index])) for key, value in self.arrays.items()}


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device=device, non_blocking=True) for key, value in batch.items()}


def forward_loss(model, criterion, batch):
    output = model(
        object_features=batch["object_features"].float().unsqueeze(1),
        camera_support=batch["camera_support"].float().unsqueeze(1),
        camera_quality=batch["camera_quality"].float().unsqueeze(1),
        temporal_features=batch["temporal_features"].float().unsqueeze(1),
        decision_features=batch["decision_features"].float().unsqueeze(1),
    )
    if model.enable_routing:
        raise RuntimeError("P0 routing must remain disabled")
    losses = criterion(
        output,
        batch["evidence_drop"].float().unsqueeze(1),
        batch["cross_topk"].float().unsqueeze(1),
        batch["valid_mask"].bool().unsqueeze(1),
    )
    return output, losses


def evaluate(model, criterion, loader, device):
    model.eval()
    sums = {"total": 0.0, "regression": 0.0, "crossing": 0.0, "rows": 0}
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device)
            _, losses = forward_loss(model, criterion, batch)
            n = int(batch["object_features"].shape[0])
            sums["total"] += float(losses["loss_care3d"].item()) * n
            sums["regression"] += float(losses["loss_care3d_vulnerability"].item()) * n
            sums["crossing"] += float(losses["loss_care3d_crossing"].item()) * n
            sums["rows"] += n
    return {key: (value / max(sums["rows"], 1) if key != "rows" else value)
            for key, value in sums.items()}


def main() -> None:
    args = parse_args()
    cfg = runpy.run_path(str(CONFIG))
    train_cfg = dict(cfg["training"])
    model_cfg = dict(cfg["model"])
    loss_cfg = dict(cfg["loss"])
    epochs = int(args.epochs or train_cfg["epochs"])
    batch_size = int(args.batch_size or train_cfg["batch_size"])

    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    formal = progress.get("stages", {}).get("formal_extraction")
    if not isinstance(formal, dict):
        raise RuntimeError("formal counterfactual extraction has not started")
    if any(int(v["completed_scenes"]) != int(v["expected_scenes"]) for v in formal.values()):
        raise RuntimeError("all 684 formal scenes must be complete before training")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_arrays = load_split("probe_train")
    val_arrays = load_split("probe_val")
    # Deliberately do not read probe_test in this script.
    train_labels = train_arrays["cross_topk"].astype(np.float64)
    train_valid = train_arrays["valid_mask"].astype(bool)
    positives, negatives, pos_weights = {}, {}, {}
    for index, protocol in enumerate(PROTOCOLS):
        valid = train_valid[:, index]
        pos = int(train_labels[valid, index].sum())
        neg = int(valid.sum()) - pos
        positives[protocol], negatives[protocol] = pos, neg
        pos_weights[protocol] = float(neg / max(pos, 1))

    train_loader = DataLoader(ArrayDataset(train_arrays), batch_size=batch_size, shuffle=True,
                              num_workers=int(train_cfg["num_workers"]), drop_last=False)
    val_loader = DataLoader(ArrayDataset(val_arrays), batch_size=batch_size, shuffle=False,
                            num_workers=int(train_cfg["num_workers"]), drop_last=False)

    model = CARE3DCore(**model_cfg).to(device)
    if model.enable_routing or model.router is not None:
        raise RuntimeError("SparseEvidenceRouter is prohibited in P0")
    criterion = CounterfactualVulnerabilityLoss(**loss_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]),
                                  weight_decay=float(train_cfg["weight_decay"]))

    output_dir = REPORT / "training" / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pth"
    history = []
    best_val = math.inf
    best_epoch = -1
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total, rows = 0.0, 0
        for batch in train_loader:
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            _, losses = forward_loss(model, criterion, batch)
            losses["loss_care3d"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["gradient_clip_norm"]))
            optimizer.step()
            n = int(batch["object_features"].shape[0])
            total += float(losses["loss_care3d"].item()) * n
            rows += n
        val = evaluate(model, criterion, val_loader, device)
        row = {"epoch": epoch, "train_loss": total / max(rows, 1), **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if val["total"] < best_val - 1e-8:
            best_val = float(val["total"])
            best_epoch = epoch
            patience = 0
            torch.save({
                "schema_version": SCHEMA,
                "seed": args.seed,
                "epoch": epoch,
                "model_config": model_cfg,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val,
                "routing_enabled": False,
            }, best_path)
        else:
            patience += 1
            if patience >= int(train_cfg["patience"]):
                break

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    manifest = {
        "schema_version": SCHEMA,
        "status": "TRAINING_COMPLETE_TEST_UNSEEN",
        "seed": args.seed,
        "train_rows": int(len(train_arrays["object_features"])),
        "val_rows": int(len(val_arrays["object_features"])),
        "probe_test_read": False,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "boundary_train_positives": positives,
        "boundary_train_negatives": negatives,
        "boundary_train_pos_weight_reference": pos_weights,
        "boundary_loss_weighting": "unweighted_existing_CounterfactualVulnerabilityLoss",
        "detector_parameters_in_optimizer": 0,
        "routing_enabled": False,
    }
    atomic_json(output_dir / "training_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
