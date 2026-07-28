"""Single-GPU evaluation with per-frame Evidence3D diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import mmcv
import torch


def _find_first_dict(value: Any) -> Optional[Dict]:
    """Recursively locate the first metadata dictionary."""
    if hasattr(value, "data"):
        return _find_first_dict(value.data)

    if isinstance(value, dict):
        return value

    if isinstance(value, (list, tuple)):
        for item in value:
            found = _find_first_dict(item)
            if found is not None:
                return found

    return None


def _to_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()

    if isinstance(value, dict):
        return {
            str(key): _to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def _resolve_trace_path(trace_root: str) -> Path:
    root = Path(trace_root).expanduser()

    protocol_file = os.environ.get("EVIDENCE3D_PROTOCOL", "")
    scenario = (
        Path(protocol_file).stem
        if protocol_file
        else "clean"
    )

    if root.suffix.lower() == ".jsonl":
        path = root
    else:
        path = root / f"{scenario}_diagnostic_trace.jsonl"

    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def single_gpu_test_with_evidence_trace(
    model,
    data_loader,
    trace_root: str,
):
    """Evaluate and export one diagnostic JSON record per sample."""
    model.eval()
    results = []

    dataset = data_loader.dataset
    progress = mmcv.ProgressBar(len(dataset))

    trace_path = _resolve_trace_path(trace_root)
    protocol_file = os.environ.get("EVIDENCE3D_PROTOCOL")

    with trace_path.open("w", encoding="utf-8") as handle:
        for batch_index, data in enumerate(data_loader):
            with torch.no_grad():
                result = model(
                    return_loss=False,
                    rescale=True,
                    **data,
                )

            if not isinstance(result, list):
                result = [result]

            if len(result) != 1:
                raise RuntimeError(
                    "Evidence diagnostic export currently requires "
                    "samples_per_gpu=1, but model returned "
                    f"{len(result)} samples."
                )

            results.extend(result)

            base_model = (
                model.module
                if hasattr(model, "module")
                else model
            )
            head = getattr(base_model, "pts_bbox_head", None)

            summary = {}
            diagnostics = {}

            if head is not None:
                if hasattr(head, "get_last_evidence_summary"):
                    summary = head.get_last_evidence_summary()

                if hasattr(head, "get_last_evidence_diagnostics"):
                    diagnostics = (
                        head.get_last_evidence_diagnostics()
                    )

            meta = _find_first_dict(data.get("img_metas", {})) or {}

            record = {
                "batch_index": int(batch_index),
                "sample_idx": str(meta.get("sample_idx", "")),
                "scene_token": str(meta.get("scene_token", "")),
                "frame_idx": int(meta.get("frame_idx", -1)),
                "protocol_file": protocol_file,
                "summary": _to_jsonable(summary),
                "diagnostics": _to_jsonable(diagnostics),
            }

            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            handle.flush()

            progress.update()

    print(f"\nEvidence diagnostic trace writes to {trace_path}")
    return results
