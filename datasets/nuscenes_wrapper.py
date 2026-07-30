"""nuScenes-mini validation and StreamPETR dataset wrapper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from .corruption import CAMERA_NAMES

try:
    from mmdet.datasets import DATASETS
    from projects.mmdet3d_plugin.datasets.nuscenes_dataset import CustomNuScenesDataset
except Exception:  # pragma: no cover
    DATASETS = None
    CustomNuScenesDataset = object


def validate_nuscenes_mini_layout(data_root: Path) -> Dict[str, bool]:
    root = Path(data_root).expanduser().resolve()
    required = ["maps", "samples", "sweeps", "v1.0-mini"]
    status = {name: (root / name).exists() for name in required}
    status["train_info"] = (root / "nuscenes2d_temporal_infos_train.pkl").is_file()
    status["val_info"] = (root / "nuscenes2d_temporal_infos_val.pkl").is_file()
    return status


def _register_dataset(cls):
    if DATASETS is not None:
        return DATASETS.register_module()(cls)
    return cls


@_register_dataset
class EvidenceNuScenesDataset(CustomNuScenesDataset):
    """Thin wrapper preserving official StreamPETR temporal behavior."""

    CAMERA_NAMES: Sequence[str] = CAMERA_NAMES

    def __init__(self, *args, include_instance_tokens: bool = False, **kwargs):
        self.include_instance_tokens = bool(include_instance_tokens)
        self._sample_instance_tokens = {}
        super().__init__(*args, **kwargs)
        if self.include_instance_tokens:
            version_dir = Path(self.data_root) / "v1.0-mini"
            with (version_dir / "sample.json").open(encoding="utf-8") as handle:
                samples = json.load(handle)
            with (version_dir / "sample_annotation.json").open(encoding="utf-8") as handle:
                annotations = json.load(handle)
            grouped = {sample["token"]: [] for sample in samples}
            for annotation in annotations:
                grouped[annotation["sample_token"]].append(
                    annotation["instance_token"]
                )
            self._sample_instance_tokens = grouped

    def get_data_info(self, index):  # type: ignore[override]
        data = super().get_data_info(index)
        data["camera_names"] = list(self.CAMERA_NAMES)
        if self.include_instance_tokens and not self.test_mode:
            info = self.data_infos[index]
            mask = (
                np.asarray(info["valid_flag"], dtype=bool)
                if self.use_valid_flag
                else np.asarray(info["num_lidar_pts"]) > 0
            )
            tokens = self._sample_instance_tokens[info["token"]]
            if len(tokens) != len(mask):
                raise RuntimeError("nuScenes annotation/token order mismatch")
            names = np.asarray(info["gt_names"])[mask]
            labels = np.asarray(
                [self.CLASSES.index(name) if name in self.CLASSES else -1 for name in names],
                dtype=np.int64,
            )
            data["_raw_gt_instance_tokens"] = [
                token for token, keep in zip(tokens, mask) if keep
            ]
            data["_raw_token_boxes"] = np.asarray(info["gt_boxes"])[mask]
            data["_raw_token_labels"] = labels
        return data
