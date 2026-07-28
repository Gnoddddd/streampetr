"""nuScenes-mini validation and StreamPETR dataset wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

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

    def get_data_info(self, index):  # type: ignore[override]
        data = super().get_data_info(index)
        data["camera_names"] = list(self.CAMERA_NAMES)
        return data
