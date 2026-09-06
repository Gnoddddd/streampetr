#!/usr/bin/env python3
"""Train CARE-3D P1 sparse score routers without reading probe-test."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STREAM = ROOT / "repos/StreamPETR"
sys.dont_write_bytecode = True
sys.path.insert(0, str(STREAM))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmcv.utils import import_modules_from_strings  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler  # noqa: E402

from analysis.care3d_counterfactual import freeze_module  # noqa: E402
from analysis.care3d_p1 import PROTOCOLS, SOURCE_NAMES  # noqa: E402
from models.care3d import CARE3DCore  # noqa: E402
from models.care3d_p1 import CARE3DP1ScoreRouter, p1_score_routing_loss  # noqa: E402

REPORT = ROOT / "reports/care3d/p1_sparse_evidence_router"
P0 = ROOT / "reports/care3d/p0_counterfactual_vulnerability"
P1_CONFIG = ROOT / "configs/care3d/p1_sparse_evidence_router.py"
SEEDS = (42, 2027, 2028)
SCHEMA = 1
PREDICTOR_KEYS = (
    "object_features", "temporal_features", "decision_features",
    "camera_support", "camera_quality",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_history(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(rows[0]) if rows else ["epoch"]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def require_ready() -> tuple[dict, dict]:
    validation_path = REPORT / "source_validation.json"
    progress_path = REPORT / "progress_manifest.json"
    smoke_path = REPORT / "engineering_smoke_gate.json"
    if not validation_path.exists() or not progress_path.exists() or not smoke_path.exists():
        raise RuntimeError("run P1 prepare + engineering smoke gate first")
    validation = json.loads(validation_path.read_text())
    progress = json.loads(progress_path.read_text())
    smoke = json.loads(smoke_path.read_text())
    if validation.get("status") != "VALIDATED_BEFORE_P1_FORWARD" or not smoke.get("passed"):
        raise RuntimeError("P1 frozen validation/engineering smoke did not pass")
    if progress.get("status") not in (
        "P1_SUPERVISION_COMPLETE_TRAINING_ELIGIBLE",
        "P1_TRAINING_RUNNING",
        "P1_TRAINING_COMPLETE_TEST_EVALUATION_ELIGIBLE",
    ):
        raise RuntimeError(f"P1 training is not eligible: {progress.get('status')}")
    supervision = progress.get("stages", {}).get("supervision_extraction", {})
    for split, expected in (("probe_train", 419), ("probe_val", 133)):
        if int(supervision.get(split, {}).get("completed_scenes", -1)) != expected:
            raise RuntimeError(f"P1 {split} supervision incomplete")
    test_dir = REPORT / "evaluation/probe_test"
    if test_dir.exists() and any(test_dir.glob("*.complete.json")):
        raise RuntimeError("probe-test evaluation exists before all P1 checkpoints freeze")
    return validation, progress


def load_split(split: str) -> dict[str, np.ndarray]:
    if split not in ("probe_train", "probe_val"):
        raise RuntimeError("P1 trainer may only read probe_train/probe_val")
    manifest = pd.read_csv(REPORT / "frozen_scene_manifest.csv")
    scenes = manifest[manifest.split.astype(str) == split].scene_token.astype(str).tolist()
    expected = 419 if split == "probe_train" else 133
    if len(scenes) != expected:
        raise RuntimeError(f"unexpected {split} scene count")
    keys = (
        *PREDICTOR_KEYS, "target_class", "clean_query", "fault_query",
        "source_features", "source_reliability", "clean_score", "fault_score",
        "fault_topk_threshold", "cross_topk", "valid_mask",
    )
    collected = {key: [] for key in keys}
    for scene in scenes:
        prefix = REPORT / "incremental/supervision" / scene
        marker_path, feature_path = prefix.with_suffix(".complete.json"), prefix.with_suffix(".features.npz")
        if not marker_path.exists() or not feature_path.exists():
            raise RuntimeError(f"missing P1 supervision: {scene}")
        marker = json.loads(marker_path.read_text())
        if not marker.get("complete") or marker.get("split") != split:
            raise RuntimeError(f"invalid P1 supervision marker: {scene}")
        with np.load(feature_path) as packed:
            for key in keys:
                if key not in packed.files:
                    raise RuntimeError(f"{scene} missing P1 array {key}")
                collected[key].append(np.asarray(packed[key]).copy())
    return {key: np.concatenate(value, axis=0) for key, value in collected.items()}


def flatten_protocols(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    valid = arrays["valid_mask"].astype(bool)
    names = (
        *PREDICTOR_KEYS, "target_class", "clean_query", "fault_query",
        "source_features", "source_reliability", "clean_score", "fault_score",
        "fault_topk_threshold", "cross_topk", "protocol_index",
    )
    output = {key: [] for key in names}
    for protocol_index in range(len(PROTOCOLS)):
        mask = valid[:, protocol_index]
        if not mask.any():
            continue
        for key in PREDICTOR_KEYS:
            output[key].append(arrays[key][mask])
        output["target_class"].append(arrays["target_class"][mask])
        output["clean_query"].append(arrays["clean_query"][mask])
        output["fault_query"].append(arrays["fault_query"][mask, protocol_index])
        output["source_features"].append(arrays["source_features"][mask, protocol_index])
        output["source_reliability"].append(arrays["source_reliability"][mask, protocol_index])
        output["clean_score"].append(arrays["clean_score"][mask])
        output["fault_score"].append(arrays["fault_score"][mask, protocol_index])
        output["fault_topk_threshold"].append(arrays["fault_topk_threshold"][mask, protocol_index])
        output["cross_topk"].append(arrays["cross_topk"][mask, protocol_index])
        output["protocol_index"].append(np.full(int(mask.sum()), protocol_index, dtype=np.int64))
    return {key: np.concatenate(value, axis=0) for key, value in output.items()}


class P1Dataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.arrays = arrays
        if len({len(value) for value in arrays.values()}) != 1:
            raise RuntimeError("P1 flattened arrays are misaligned")

    def __len__(self) -> int:
        return len(self.arrays["target_class"])

    def __getitem__(self, index: int):
        return {key: value[index] for key, value in self.arrays.items()}


def balanced_weights(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, int]]:
    protocol = arrays["protocol_index"].astype(int)
    label = arrays["cross_topk"].astype(int)
    counts, weights = {}, np.zeros(len(label), dtype=np.float64)
    for p_index, name in enumerate(PROTOCOLS):
        for y in (0, 1):
            mask = (protocol == p_index) & (label == y)
            count = int(mask.sum())
            counts[f"{name}:cross_topk={y}"] = count
            if count == 0:
                raise RuntimeError(f"P1 train lacks class {name}/{y}")
            weights[mask] = 1.0 / count
    weights *= len(weights) / weights.sum()
    return weights, counts


def build_p0(seed: int, device: torch.device, validation: dict):
    path = P0 / "training" / f"seed_{seed}" / "best.pth"
    if sha256(path) != validation["p0_checkpoint_sha256"][str(seed)]:
        raise RuntimeError(f"P0 seed {seed} checkpoint hash changed")
    payload = torch.load(path, map_location="cpu")
    model = CARE3DCore(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if model.router is not None:
        raise RuntimeError("frozen P0 unexpectedly contains a router")
    return freeze_module(model.to(device))


def build_classifier(device: torch.device, config: dict):
    cfg = Config.fromfile(str(ROOT / config["stream_petr_config"]))
    import_modules_from_strings(**cfg.custom_imports)
    cfg.model.pretrained = None
    detector = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(detector, str(ROOT / config["stream_petr_checkpoint"]), map_location="cpu")
    classifier = copy.deepcopy(detector.pts_bbox_head.cls_branches[-1]).to(device).eval()
    freeze_module(classifier)
    del detector
    return classifier


def to_device(batch, device):
    output = {}
    for key, value in batch.items():
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
        tensor = tensor.long() if key in {"target_class", "protocol_index"} else tensor.float()
        output[key] = tensor.to(device, non_blocking=True)
    return output


def p0_forward(p0, batch):
    return p0(
        object_features=batch["object_features"],
        camera_support=batch["camera_support"],
        camera_quality=batch["camera_quality"],
        temporal_features=batch["temporal_features"],
        decision_features=batch["decision_features"],
    )


def batch_loss(batch, p0, p1, classifier, loss_cfg):
    with torch.no_grad():
        p0_output = p0_forward(p0, batch)
        fault_logits = classifier(batch["fault_query"])
        computed_fault_score = torch.sigmoid(
            fault_logits.gather(1, batch["target_class"][:, None])[:, 0]
        )
        if not torch.allclose(computed_fault_score, batch["fault_score"], rtol=0.0, atol=5e-4):
            difference = float((computed_fault_score - batch["fault_score"]).abs().max().item())
            raise RuntimeError(f"P1 classifier replay diverged from exported score: {difference}")
    routed, aux = p1(
        batch["fault_query"], batch["source_features"], batch["source_reliability"],
        p0_output["vulnerability"].detach(), p0_output["boundary_crossing_logits"].detach(),
        batch["protocol_index"], fault_active=True,
    )
    routed_logits = classifier(routed)
    losses = p1_score_routing_loss(
        routed_query=routed, clean_query=batch["clean_query"], fault_query=batch["fault_query"],
        routed_logits=routed_logits, fault_logits=fault_logits.detach(),
        target_class=batch["target_class"], clean_score=batch["clean_score"],
        fault_score=batch["fault_score"], fault_topk_threshold=batch["fault_topk_threshold"],
        cross_topk=batch["cross_topk"], **loss_cfg,
    )
    losses["mean_risk"] = aux["risk_probability"].mean().detach()
    losses["mean_correction_norm"] = aux["correction"].norm(dim=-1).mean().detach()
    return losses


def evaluate(loader, p0, p1, classifier, loss_cfg, device):
    p1.eval()
    sums, rows = {}, 0
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            losses = batch_loss(batch, p0, p1, classifier, loss_cfg)
            count = len(batch["target_class"])
            rows += count
            for key in ("total", "score", "boundary", "query", "retained", "non_target",
                        "mean_risk", "mean_correction_norm"):
                sums[key] = sums.get(key, 0.0) + float(losses[key].item()) * count
    return {key: value / max(rows, 1) for key, value in sums.items()}


def update_progress() -> None:
    progress_path = REPORT / "progress_manifest.json"
    progress = json.loads(progress_path.read_text())
    completed = []
    for seed in SEEDS:
        path = REPORT / "training" / f"seed_{seed}" / "training_manifest.json"
        if path.exists():
            value = json.loads(path.read_text())
            if value.get("status") == "P1_TRAINING_COMPLETE_TEST_UNSEEN" and value.get("probe_test_read") is False:
                completed.append(seed)
    progress["stages"]["training"] = {"completed_seeds": completed, "expected_seeds": list(SEEDS)}
    if completed == list(SEEDS):
        progress["status"] = "P1_TRAINING_COMPLETE_TEST_EVALUATION_ELIGIBLE"
        progress["stages"]["probe_test_evaluation"] = "ELIGIBLE"
    else:
        progress["status"] = "P1_TRAINING_RUNNING"
    atomic_json(progress_path, progress)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")
    validation, _ = require_ready()
    config = runpy.run_path(str(P1_CONFIG))
    training_cfg, loss_cfg, router_cfg = dict(config["training"]), dict(config["loss"]), dict(config["router"])
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)

    train, val = flatten_protocols(load_split("probe_train")), flatten_protocols(load_split("probe_val"))
    weights, balance_counts = balanced_weights(train)
    train_dataset, val_dataset = P1Dataset(train), P1Dataset(val)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(train_dataset), True, generator=generator)
    batch_size = int(args.batch_size or training_cfg["batch_size"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=int(training_cfg["num_workers"]), pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=int(training_cfg["num_workers"]), pin_memory=True)

    p0 = build_p0(seed, device, validation)
    classifier = build_classifier(device, config)
    p1 = CARE3DP1ScoreRouter(
        object_dim=int(router_cfg["object_dim"]), source_dim=int(router_cfg["source_dim"]),
        vulnerability_dim=int(router_cfg["vulnerability_dim"]), hidden_dim=int(router_cfg["hidden_dim"]),
        top_k=int(router_cfg["top_k"]),
    ).to(device)
    optimizer = torch.optim.AdamW(p1.parameters(), lr=float(training_cfg["learning_rate"]),
                                  weight_decay=float(training_cfg["weight_decay"]))
    p1_parameter_ids = {id(p) for p in p1.parameters() if p.requires_grad}
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if p1_parameter_ids != optimizer_ids:
        raise RuntimeError("P1 optimizer does not contain exactly router parameters")
    if any(p.requires_grad for p in p0.parameters()) or any(p.requires_grad for p in classifier.parameters()):
        raise RuntimeError("frozen P0/detector unexpectedly trainable")

    epochs = int(args.epochs or training_cfg["epochs"])
    patience, clip = int(training_cfg["patience"]), float(training_cfg["gradient_clip_norm"])
    history, best_loss, best_epoch, stale = [], float("inf"), -1, 0
    directory = REPORT / "training" / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    best_path = directory / "best.pth"
    for epoch in range(epochs):
        p1.train()
        train_sum, train_rows = 0.0, 0
        for batch in train_loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            losses = batch_loss(batch, p0, p1, classifier, loss_cfg)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(p1.parameters(), clip)
            optimizer.step()
            count = len(batch["target_class"])
            train_sum += float(losses["total"].item()) * count
            train_rows += count
        validation_metrics = evaluate(val_loader, p0, p1, classifier, loss_cfg, device)
        row = {"epoch": epoch, "train_total": train_sum / max(train_rows, 1),
               **{f"val_{key}": value for key, value in validation_metrics.items()}}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        current = validation_metrics["total"]
        if current < best_loss - 1e-8:
            best_loss, best_epoch, stale = current, epoch, 0
            torch.save({
                "schema_version": SCHEMA, "seed": seed, "model_state_dict": p1.state_dict(),
                "router_config": {key: int(router_cfg[key]) for key in
                                  ("object_dim", "source_dim", "vulnerability_dim", "hidden_dim", "top_k")},
                "source_names": tuple(SOURCE_NAMES),
                "p0_checkpoint_sha256": validation["p0_checkpoint_sha256"][str(seed)],
                "best_epoch": best_epoch, "best_val_loss": best_loss,
            }, best_path)
        else:
            stale += 1
            if stale >= patience:
                break
    if best_epoch < 0 or not best_path.exists():
        raise RuntimeError("P1 training did not produce a checkpoint")
    write_history(directory / "history.csv", history)
    manifest = {
        "schema_version": SCHEMA, "status": "P1_TRAINING_COMPLETE_TEST_UNSEEN", "seed": seed,
        "best_epoch": best_epoch, "best_val_loss": best_loss,
        "train_rows_flattened": len(train_dataset), "val_rows_flattened": len(val_dataset),
        "probe_test_read": False, "probe_test_exported_before_training": False,
        "detector_parameters_in_optimizer": 0, "p0_parameters_in_optimizer": 0,
        "router_parameters_in_optimizer": sum(p.numel() for p in p1.parameters()),
        "source_names": list(SOURCE_NAMES), "top_k": int(router_cfg["top_k"]),
        "classification_only": True, "regression_unchanged": True,
        "balanced_sampling_source": "probe_train_protocol_x_cross_topk_only",
        "balanced_sampling_counts": balance_counts, "checkpoint_sha256": sha256(best_path),
    }
    atomic_json(directory / "training_manifest.json", manifest)
    update_progress()
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
