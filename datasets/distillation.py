"""Training-only data helpers for observability-guided distillation."""

from __future__ import annotations

from typing import Dict, Sequence

import mmcv
import numpy as np

try:
    from mmdet.datasets.builder import PIPELINES
except Exception:  # pragma: no cover
    PIPELINES = None


def _register(cls):
    return PIPELINES.register_module()(cls) if PIPELINES is not None else cls


@_register
class FinalizePairedCleanImages:
    """Apply the student's normalization/padding to the saved clean views."""

    def __init__(self, mean: Sequence[float], std: Sequence[float], to_rgb: bool, size_divisor: int):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.to_rgb = bool(to_rgb)
        self.size_divisor = int(size_divisor)

    def __call__(self, results: Dict) -> Dict:
        clean = results.get("clean_img")
        if clean is None:
            raise KeyError("FinalizePairedCleanImages requires clean_img")
        results["clean_img"] = [
            mmcv.impad_to_multiple(
                mmcv.imnormalize(image, self.mean, self.std, self.to_rgb),
                self.size_divisor,
            ).transpose(2, 0, 1)
            for image in clean
        ]
        return results


@_register
class MatchFilteredGTInstanceTokens:
    """Align filtered GT boxes to raw annotation instance tokens.

    This transform runs after range/name filtering and before geometric data
    augmentation, so exact class-aware center matching is deterministic.
    """

    def __call__(self, results: Dict) -> Dict:
        raw_boxes = np.asarray(results.pop("_raw_token_boxes"), dtype=np.float32)
        raw_labels = np.asarray(results.pop("_raw_token_labels"), dtype=np.int64)
        raw_tokens = list(results.pop("_raw_gt_instance_tokens"))
        boxes = results["gt_bboxes_3d"].tensor.detach().cpu().numpy()
        labels = np.asarray(results["gt_labels_3d"], dtype=np.int64)
        tokens = []
        used = set()
        for box, label in zip(boxes, labels):
            candidates = np.flatnonzero(raw_labels == label)
            if len(candidates) == 0:
                raise RuntimeError("Filtered GT has no class-aligned instance token")
            # LiDARInstance3DBoxes changes the z-origin from geometric center
            # to bottom center; x/y remain an exact converter-order key.
            distances = np.linalg.norm(raw_boxes[candidates, :2] - box[:2], axis=1)
            order = candidates[np.argsort(distances)]
            match = next((int(index) for index in order if int(index) not in used), None)
            if match is None or float(np.linalg.norm(raw_boxes[match, :2] - box[:2])) > 1e-4:
                raise RuntimeError("Could not exactly align filtered GT instance token")
            used.add(match)
            tokens.append(raw_tokens[match])
        results["gt_instance_tokens"] = tokens
        return results
