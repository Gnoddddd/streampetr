#!/usr/bin/env python3
"""Finalize the CARE-3D P1 engineering smoke gate before formal extraction."""

from __future__ import annotations

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

from analysis.care3d_counterfactual import freeze_module  # noqa: E402
from analysis.care3d_p1 import SOURCE_NAMES, assert_source_contract  # noqa: E402
from models.care3d import CARE3DCore  # noqa: E402
from models.care3d_p1 import CARE3DP1ScoreRouter  # noqa: E402


REPORT = ROOT / "reports/care3d/p1_sparse_evidence_router"
P0 = ROOT / "reports/care3d/p0_counterfactual_vulnerability"
P1_CONFIG = ROOT / "configs/care3d/p1_sparse_evidence_router.py"
SCHEMA = 1
CLASSIFIER_REPLAY_TOLERANCE = 5e-4
PREDICTOR_KEYS = (
    "object_features", "temporal_features", "decision_features",
    "camera_support", "camera_quality",
)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def p0_interface_replay(arrays: dict[str, np.ndarray]) -> bool:
    checkpoint = P0 / "training/seed_42/best.pth"
    if not checkpoint.exists():
        raise RuntimeError("missing frozen P0 seed-42 checkpoint")
    payload = torch.load(checkpoint, map_location="cpu")
    model = CARE3DCore(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if model.router is not None:
        raise RuntimeError("frozen P0 seed 42 unexpectedly contains a router")
    model = freeze_module(model.eval())
    count = min(64, len(arrays["object_features"]))
    with torch.no_grad():
        output = model(
            object_features=torch.as_tensor(arrays["object_features"][:count]).float().unsqueeze(1),
            camera_support=torch.as_tensor(arrays["camera_support"][:count]).float().unsqueeze(1),
            camera_quality=torch.as_tensor(arrays["camera_quality"][:count]).float().unsqueeze(1),
            temporal_features=torch.as_tensor(arrays["temporal_features"][:count]).float().unsqueeze(1),
            decision_features=torch.as_tensor(arrays["decision_features"][:count]).float().unsqueeze(1),
        )
    vulnerability = output["vulnerability"]
    crossing = output["boundary_crossing_logits"]
    expected = (count, 1, 3)
    return bool(
        tuple(vulnerability.shape) == expected
        and tuple(crossing.shape) == expected
        and torch.isfinite(vulnerability).all().item()
        and torch.isfinite(crossing).all().item()
    )


def classifier_replay(
    clean_query: np.ndarray,
    fault_query: np.ndarray,
    clean_score: np.ndarray,
    fault_score: np.ndarray,
    target_class: np.ndarray,
) -> tuple[bool, float]:
    """Verify exported final-query tensors still replay the frozen classifier."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for P1 classifier replay smoke")
    config = runpy.run_path(str(P1_CONFIG))
    cfg = Config.fromfile(str(ROOT / config["stream_petr_config"]))
    import_modules_from_strings(**cfg.custom_imports)
    cfg.model.pretrained = None
    detector = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(
        detector,
        str(ROOT / config["stream_petr_checkpoint"]),
        map_location="cpu",
    )
    detector = freeze_module(detector.cuda().eval())
    classifier = detector.pts_bbox_head.cls_branches[-1]
    target = torch.as_tensor(target_class, device="cuda:0", dtype=torch.long)
    differences = []
    with torch.no_grad():
        clean = torch.as_tensor(clean_query, device="cuda:0", dtype=torch.float32)
        clean_logits = classifier(clean)
        clean_replayed = torch.sigmoid(clean_logits.gather(1, target[:, None])[:, 0])
        clean_reference = torch.as_tensor(clean_score, device="cuda:0", dtype=torch.float32)
        differences.append(float((clean_replayed - clean_reference).abs().max().item()))
        for protocol_index in range(fault_query.shape[1]):
            query = torch.as_tensor(
                fault_query[:, protocol_index], device="cuda:0", dtype=torch.float32
            )
            logits = classifier(query)
            replayed = torch.sigmoid(logits.gather(1, target[:, None])[:, 0])
            reference = torch.as_tensor(
                fault_score[:, protocol_index], device="cuda:0", dtype=torch.float32
            )
            differences.append(float((replayed - reference).abs().max().item()))
    maximum = max(differences, default=float("inf"))
    del detector
    torch.cuda.empty_cache()
    return bool(maximum <= CLASSIFIER_REPLAY_TOLERANCE), float(maximum)


def main() -> None:
    engineering = pd.read_csv(REPORT / "engineering_scene_manifest.csv")
    if len(engineering) != 1:
        raise RuntimeError("expected one frozen P1 engineering scene")
    scene = str(engineering.iloc[0].scene_token)
    prefix = REPORT / "engineering_smoke" / scene
    marker_path = prefix.with_suffix(".complete.json")
    feature_path = prefix.with_suffix(".features.npz")
    if not marker_path.exists() or not feature_path.exists():
        raise RuntimeError("run export_care3d_p1_supervision.py --engineering-scene first")
    marker = json.loads(marker_path.read_text())
    base_keys = (
        "equivalence_pass",
        "predictor_input_identity_pass",
        "label_identity_pass",
        "source_contract_pass",
    )
    if not marker.get("complete") or not all(bool(marker.get(key)) for key in base_keys):
        raise RuntimeError(f"base P1 engineering smoke failed: {marker}")

    with np.load(feature_path) as packed:
        arrays = {key: np.asarray(packed[key]).copy() for key in PREDICTOR_KEYS}
        source_features = np.asarray(packed["source_features"]).astype(np.float32)
        reliability = np.asarray(packed["source_reliability"]).astype(np.float32)
        clean_query = np.asarray(packed["clean_query"]).astype(np.float32)
        fault_query = np.asarray(packed["fault_query"]).astype(np.float32)
        clean_score = np.asarray(packed["clean_score"]).astype(np.float32)
        fault_score = np.asarray(packed["fault_score"]).astype(np.float32)
        target_class = np.asarray(packed["target_class"]).astype(np.int64)
    assert_source_contract(source_features, reliability)
    if len(fault_query) == 0:
        raise RuntimeError("engineering scene contains no P1 object rows")

    p0_pass = p0_interface_replay(arrays)
    replay_pass, replay_difference = classifier_replay(
        clean_query, fault_query, clean_score, fault_score, target_class
    )

    count = min(16, len(fault_query))
    query = torch.as_tensor(fault_query[:count, 0], dtype=torch.float32)
    sources = torch.as_tensor(source_features[:count, 0], dtype=torch.float32)
    source_reliability = torch.as_tensor(reliability[:count, 0], dtype=torch.float32)
    vulnerability = torch.zeros(count, 3)
    boundary_logits = torch.zeros(count, 3)
    protocol = torch.zeros(count, dtype=torch.long)

    router = CARE3DP1ScoreRouter(
        object_dim=256,
        source_dim=256,
        vulnerability_dim=3,
        hidden_dim=256,
        top_k=2,
    ).eval()
    with torch.no_grad():
        routed, _ = router(
            query, sources, source_reliability, vulnerability, boundary_logits,
            protocol, fault_active=True,
        )
    zero_identity = bool(torch.equal(routed, query))

    with torch.no_grad():
        for parameter in router.parameters():
            parameter.uniform_(-0.25, 0.25)
        bypass, _ = router(
            query, sources, source_reliability, vulnerability, boundary_logits,
            protocol, fault_active=False,
        )
    clean_bypass = bool(torch.equal(bypass, query))
    source_names_pass = tuple(marker.get("source_names", ())) == tuple(SOURCE_NAMES)

    result = {
        "schema_version": SCHEMA,
        "scene_token": scene,
        "equivalence_pass": True,
        "predictor_input_identity_pass": True,
        "label_identity_pass": True,
        "source_contract_pass": True,
        "source_names_pass": bool(source_names_pass),
        "p0_interface_replay_pass": p0_pass,
        "classifier_replay_pass": replay_pass,
        "classifier_replay_max_abs_diff": replay_difference,
        "classifier_replay_tolerance": CLASSIFIER_REPLAY_TOLERANCE,
        "zero_init_fault_identity_pass": zero_identity,
        "trained_weight_clean_bypass_identity_pass": clean_bypass,
        "failed_cam_back_absent": "CAM_BACK" not in SOURCE_NAMES,
    }
    required = (
        "equivalence_pass",
        "predictor_input_identity_pass",
        "label_identity_pass",
        "source_contract_pass",
        "source_names_pass",
        "p0_interface_replay_pass",
        "classifier_replay_pass",
        "zero_init_fault_identity_pass",
        "trained_weight_clean_bypass_identity_pass",
        "failed_cam_back_absent",
    )
    result["passed"] = bool(all(result[key] for key in required))
    atomic_json(REPORT / "engineering_smoke_gate.json", result)
    progress_path = REPORT / "progress_manifest.json"
    progress = json.loads(progress_path.read_text())
    if result["passed"]:
        progress["status"] = "P1_ENGINEERING_SMOKE_PASSED"
        progress["stages"]["engineering_smoke"] = "PASSED"
        progress["stages"]["supervision_extraction"] = "ELIGIBLE"
    else:
        progress["status"] = "P1_ENGINEERING_SMOKE_FAILED"
        progress["stages"]["engineering_smoke"] = "FAILED"
        progress["stages"]["supervision_extraction"] = "LOCKED_ENGINEERING_SMOKE_FAILED"
    atomic_json(progress_path, progress)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
