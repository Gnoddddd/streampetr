from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from datasets.temporal_occ_nuscenes import (
    DirtyPathMapper,
    ImageAugmentation,
    NUSCENES_CAMERA_ORDER,
    SynchronizedImageAugmentor,
    TemporalOccNuScenesDataset,
)


def test_dirty_filename_mapping_is_stable_and_severity_is_configurable(tmp_path):
    mapper = DirtyPathMapper(
        str(tmp_path / "occ"),
        "{corruption_type}/level-{severity}/{camera}/{filename}",
    )
    source = "samples/CAM_FRONT/n008__CAM_FRONT__1.jpg"
    expected = tmp_path / "occ/Dirt/level-heavy/CAM_FRONT/n008__CAM_FRONT__1.jpg"
    assert mapper.resolve(source, "Dirt", "heavy") == expected
    assert mapper.resolve(source, "Dirt", "heavy") == expected
    with pytest.raises(ValueError):
        mapper.resolve("../escape.jpg", "Dirt", 1)


def _write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 12, 3), value, dtype=np.uint8)).save(path)


def test_clean_dirty_share_augmentation_shape_order_and_parameters(tmp_path, monkeypatch):
    clean_root = tmp_path / "clean"
    dirty_root = tmp_path / "dirty"
    current = {}
    history = {}
    for index, camera in enumerate(NUSCENES_CAMERA_ORDER):
        current[camera] = "samples/%s/current.png" % camera
        history[camera] = "samples/%s/history.png" % camera
        _write_rgb(clean_root / current[camera], 20 + index)
        _write_rgb(clean_root / history[camera], 40 + index)
        _write_rgb(dirty_root / "Dirt/3" / current[camera], 60 + index)
    augmentor = SynchronizedImageAugmentor(
        original_dim=(8, 12),
        final_dim=(6, 8),
        resize_lim=(1.0, 1.0),
        bot_pct_lim=(0.0, 0.0),
        rand_flip=True,
        pad_shape=(8, 10),
        mean=(0, 0, 0),
        std=(1, 1, 1),
    )
    params = ImageAugmentation(1.0, (12, 8), (2, 1, 10, 7), True, 0.0, (6, 8), (8, 10))
    calls = []

    def sample_once():
        calls.append(True)
        return params

    monkeypatch.setattr(augmentor, "sample", sample_once)
    record = dict(
        history_clean=history,
        current_clean=current,
        scene_token="scene",
        sample_token="current",
        prev_sample_token="history",
        timestamp=1.5,
        corruption_type="Dirt",
        severity=3,
    )
    dataset = TemporalOccNuScenesDataset(
        [record],
        str(clean_root),
        DirtyPathMapper(str(dirty_root), "{corruption_type}/{severity}/{relative_path}"),
        augmentor,
    )
    item = dataset[0]
    assert len(calls) == 1
    assert item["current_clean"].shape == item["current_dirty"].shape == (6, 3, 8, 10)
    assert item["history_clean"].shape == (1, 6, 3, 8, 10)
    assert item["metadata"]["camera"] == NUSCENES_CAMERA_ORDER
    assert item["metadata"]["augmentation"]["crop"] == (2, 1, 10, 7)
    assert item["metadata"]["augmentation"]["resize"] == 1.0
    assert item["metadata"]["augmentation"]["flip"] is True
    assert torch.count_nonzero(item["current_clean"][..., 6:, :]) == 0
    assert torch.count_nonzero(item["current_dirty"][..., 6:, :]) == 0
    assert torch.count_nonzero(item["current_clean"][..., 8:]) == 0
    assert torch.count_nonzero(item["current_dirty"][..., 8:]) == 0
    # Uniform camera values survive geometry and prove both branches preserve
    # exactly the declared camera order.
    assert item["current_clean"][:, 0, 0, 0].tolist() == list(range(20, 26))
    assert item["current_dirty"][:, 0, 0, 0].tolist() == list(range(60, 66))


def test_augmentation_matrix_matches_resize_crop_flip():
    params = ImageAugmentation(2.0, (20, 16), (2, 3, 10, 9), True, 0.0, (6, 8), (6, 8))
    point = torch.tensor([1.0, 2.0, 1.0])
    transformed = params.matrix() @ point
    # resize -> (2,4), crop -> (0,1), horizontal flip in width 8 -> (8,1)
    assert torch.allclose(transformed, torch.tensor([8.0, 1.0, 1.0]))
