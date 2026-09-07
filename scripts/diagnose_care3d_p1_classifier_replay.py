#!/usr/bin/env python3
"""Diagnose CARE-3D P1 classifier replay drift without opening probe-test.

This script reads only the frozen P1 probe_train/probe_val supervision cache. It
compares three replay paths against the authoritative exported target score:

1. a 512-row standalone classifier batch, matching the original P1 trainer;
2. a packed [1, 900, 256] classifier batch containing up to 512 cached queries,
   which preserves the deployed StreamPETR classifier GEMM shape while remaining
   practical for training;
3. a [1, 900, 256] shape-matched replay that restores each query's original
   detector query index inside its scene/frame/protocol group.

The goal is to distinguish cache-precision problems from execution-shape / math
path drift and to validate a deployment-shape training replacement. It does not
train, recalibrate, modify checkpoints, or read probe_test.
"""

from __future__ import annotations

import json
import runpy
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STREAM = ROOT / "repos/StreamPETR"
sys.dont_write_bytecode = True
sys.path.insert(0, str(STREAM))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from analysis.care3d_p1 import PROTOCOLS  # noqa: E402
from scripts.train_care3d_p1 import P1_CONFIG, build_classifier  # noqa: E402


REPORT = ROOT / "reports/care3d/p1_sparse_evidence_router"
STORAGE_POLICY = "fp32_router_supervision_v1"
TRAIN_BATCH = 512
QUERY_COUNT = 900
QUERY_DIM = 256


def target_scores(logits: torch.Tensor, target_class: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits.gather(1, target_class[:, None])[:, 0])


def update_max(record: dict, difference: torch.Tensor, metadata: list[dict]) -> None:
    if difference.numel() == 0:
        return
    value, index = torch.max(difference, dim=0)
    numeric = float(value.item())
    if numeric > float(record["max_abs_diff"]):
        record["max_abs_diff"] = numeric
        record["example"] = dict(metadata[int(index.item())])


def replay_batch(
    classifier,
    queries: list[np.ndarray],
    classes: list[int],
    references: list[float],
    metadata: list[dict],
    device: torch.device,
    standalone_record: dict,
    packed_record: dict,
) -> None:
    if not queries:
        return
    query = torch.as_tensor(np.stack(queries), device=device, dtype=torch.float32)
    target_class = torch.as_tensor(classes, device=device, dtype=torch.long)
    reference = torch.as_tensor(references, device=device, dtype=torch.float32)
    count = int(query.shape[0])
    if count > QUERY_COUNT:
        raise RuntimeError("diagnostic batch exceeds deployed query count")

    with torch.no_grad():
        standalone = target_scores(classifier(query), target_class)

        # Keep the exact deployed classifier invocation shape. The classifier is
        # pointwise across query rows; packing independent rows into the first
        # slots therefore preserves one [1,900,256] GEMM path without mixing
        # object information across rows.
        padded = torch.zeros(
            (1, QUERY_COUNT, QUERY_DIM), device=device, dtype=torch.float32
        )
        padded[0, :count] = query
        packed_logits = classifier(padded)[0, :count]
        packed = target_scores(packed_logits, target_class)

    update_max(standalone_record, (standalone - reference).abs(), metadata)
    update_max(packed_record, (packed - reference).abs(), metadata)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    progress = json.loads((REPORT / "progress_manifest.json").read_text())
    if progress.get("status") not in (
        "P1_SUPERVISION_COMPLETE_TRAINING_ELIGIBLE",
        "P1_TRAINING_RUNNING",
    ):
        raise RuntimeError(f"unexpected P1 state for replay diagnosis: {progress.get('status')}")

    test_dir = REPORT / "evaluation/probe_test"
    if test_dir.exists() and any(test_dir.glob("*.complete.json")):
        raise RuntimeError("probe-test has already been opened; stop diagnosis")

    device = torch.device("cuda:0")
    config = runpy.run_path(str(P1_CONFIG))
    classifier = build_classifier(device, config)
    parameter = next(classifier.parameters())

    manifest = pd.read_csv(REPORT / "frozen_scene_manifest.csv")
    manifest = manifest[manifest.split.astype(str).isin(("probe_train", "probe_val"))]
    if len(manifest) != 552:
        raise RuntimeError("expected exactly 552 frozen train/val scenes")

    batch_record = {"max_abs_diff": 0.0, "example": None}
    packed_record = {"max_abs_diff": 0.0, "example": None}
    shape_record = {"max_abs_diff": 0.0, "example": None}
    dtype_counts = Counter()
    protocol_rows = Counter()
    scenes_checked = 0
    rows_checked = 0

    batch_queries: list[np.ndarray] = []
    batch_classes: list[int] = []
    batch_references: list[float] = []
    batch_metadata: list[dict] = []

    def flush(force: bool = False) -> None:
        nonlocal batch_queries, batch_classes, batch_references, batch_metadata
        while len(batch_queries) >= TRAIN_BATCH or (force and batch_queries):
            count = TRAIN_BATCH if len(batch_queries) >= TRAIN_BATCH else len(batch_queries)
            replay_batch(
                classifier,
                batch_queries[:count],
                batch_classes[:count],
                batch_references[:count],
                batch_metadata[:count],
                device,
                batch_record,
                packed_record,
            )
            batch_queries = batch_queries[count:]
            batch_classes = batch_classes[count:]
            batch_references = batch_references[count:]
            batch_metadata = batch_metadata[count:]

    for scene_row in manifest.itertuples(index=False):
        scene = str(scene_row.scene_token)
        prefix = REPORT / "incremental/supervision" / scene
        marker_path = prefix.with_suffix(".complete.json")
        feature_path = prefix.with_suffix(".features.npz")
        sample_path = prefix.with_suffix(".samples.csv")
        if not marker_path.exists() or not feature_path.exists() or not sample_path.exists():
            raise RuntimeError(f"missing P1 supervision artifact: {scene}")
        marker = json.loads(marker_path.read_text())
        if marker.get("storage_precision_policy") != STORAGE_POLICY:
            raise RuntimeError(f"scene is not FP32-query supervision: {scene}")

        frame = pd.read_csv(sample_path)
        with np.load(feature_path) as packed:
            clean_query = np.asarray(packed["clean_query"])
            fault_query = np.asarray(packed["fault_query"])
            fault_score = np.asarray(packed["fault_score"])
            target_class = np.asarray(packed["target_class"]).astype(np.int64)
            target_query = np.asarray(packed["target_query"]).astype(np.int64)
            valid_mask = np.asarray(packed["valid_mask"]).astype(bool)
            source_features = np.asarray(packed["source_features"])

        dtype_counts[f"clean_query:{clean_query.dtype}"] += len(clean_query)
        dtype_counts[f"fault_query:{fault_query.dtype}"] += len(fault_query)
        dtype_counts[f"source_features:{source_features.dtype}"] += len(source_features)
        if clean_query.dtype != np.float32 or fault_query.dtype != np.float32:
            raise RuntimeError(f"query cache is not float32: {scene}")

        if len(frame) != len(fault_query):
            raise RuntimeError(f"metadata/feature row mismatch: {scene}")

        target_frames = frame.target_frame_idx.to_numpy(dtype=int)
        sample_ids = frame.sample_id.astype(str).to_numpy()
        for target_frame_idx in sorted(set(target_frames.tolist())):
            frame_rows = np.flatnonzero(target_frames == target_frame_idx)
            for protocol_index, protocol in enumerate(PROTOCOLS):
                selected = frame_rows[valid_mask[frame_rows, protocol_index]]
                if selected.size == 0:
                    continue
                protocol_rows[protocol] += int(selected.size)
                rows_checked += int(selected.size)

                q_index = target_query[selected]
                if len(set(q_index.tolist())) != len(q_index):
                    raise RuntimeError(f"shared query survived P1 eligibility: {scene}/{target_frame_idx}")
                query_np = fault_query[selected, protocol_index].astype(np.float32, copy=False)
                class_np = target_class[selected]
                reference_np = fault_score[selected, protocol_index].astype(np.float32, copy=False)

                metadata = [
                    {
                        "scene_token": scene,
                        "split": str(scene_row.split),
                        "target_frame_idx": int(target_frame_idx),
                        "sample_id": str(sample_ids[int(row)]),
                        "protocol": protocol,
                        "query_index": int(target_query[int(row)]),
                        "target_class": int(target_class[int(row)]),
                        "reference_score": float(fault_score[int(row), protocol_index]),
                    }
                    for row in selected
                ]

                # Restore original detector classifier shape and positions.
                padded = torch.zeros(
                    (1, QUERY_COUNT, QUERY_DIM), device=device, dtype=torch.float32
                )
                q_tensor = torch.as_tensor(q_index, device=device, dtype=torch.long)
                padded[0, q_tensor] = torch.as_tensor(query_np, device=device)
                class_tensor = torch.as_tensor(class_np, device=device, dtype=torch.long)
                reference = torch.as_tensor(reference_np, device=device, dtype=torch.float32)
                with torch.no_grad():
                    logits = classifier(padded)[0, q_tensor]
                    replay = target_scores(logits, class_tensor)
                update_max(shape_record, (replay - reference).abs(), metadata)

                for local in range(len(selected)):
                    batch_queries.append(query_np[local].copy())
                    batch_classes.append(int(class_np[local]))
                    batch_references.append(float(reference_np[local]))
                    batch_metadata.append(metadata[local])
                flush(force=False)

        scenes_checked += 1

    flush(force=True)

    result = {
        "schema_version": 2,
        "status": "P1_CLASSIFIER_REPLAY_DIAGNOSIS_COMPLETE",
        "probe_test_read": False,
        "scenes_checked": scenes_checked,
        "protocol_rows": dict(protocol_rows),
        "rows_checked": rows_checked,
        "storage_precision_policy": STORAGE_POLICY,
        "dtype_counts": dict(dtype_counts),
        "classifier_parameter_dtype": str(parameter.dtype),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "trainer_style_batch_size": TRAIN_BATCH,
        "trainer_style_replay": batch_record,
        "packed_900_replay": packed_record,
        "shape_matched_900_replay": shape_record,
        "frozen_replay_tolerance": 5e-4,
    }
    out = REPORT / "classifier_replay_diagnosis.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
