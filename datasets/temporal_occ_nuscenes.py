"""Temporal clean/OccNuScenes pairing with synchronized image augmentation.

This module intentionally does not assume an OccNuScenes on-disk convention.
``DirtyPathMapper`` makes that convention an explicit configuration value.  A
record passed to :class:`TemporalOccNuScenesDataset` contains nuScenes relative
camera filenames; no annotations or ground-truth boxes are consumed.
"""

from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


NUSCENES_CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)
SUPPORTED_CORRUPTIONS = ("Dirt", "Water-blur")


@dataclass(frozen=True)
class ImageAugmentation:
    """One set of image-domain parameters shared by clean and dirty branches."""

    resize: float
    resize_dims: Tuple[int, int]  # PIL order: (width, height)
    crop: Tuple[int, int, int, int]  # left, top, right, bottom
    flip: bool
    rotate: float
    final_dim: Tuple[int, int]  # tensor order: (height, width)
    pad_shape: Tuple[int, int]  # tensor order: (height, width)

    def matrix(self) -> Tensor:
        """Return the 3x3 original-pixel -> augmented-pixel homography."""
        angle = np.deg2rad(self.rotate)
        rot = torch.tensor(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]],
            dtype=torch.float32,
        )
        linear = torch.eye(2, dtype=torch.float32) * self.resize
        translation = -torch.tensor(self.crop[:2], dtype=torch.float32)
        if self.flip:
            flip_matrix = torch.tensor([[-1.0, 0.0], [0.0, 1.0]])
            flip_translation = torch.tensor(
                [float(self.crop[2] - self.crop[0]), 0.0]
            )
            linear = flip_matrix @ linear
            translation = flip_matrix @ translation + flip_translation
        center = torch.tensor(
            [self.crop[2] - self.crop[0], self.crop[3] - self.crop[1]],
            dtype=torch.float32,
        ) / 2.0
        translation = rot @ translation + rot @ (-center) + center
        linear = rot @ linear
        output = torch.eye(3, dtype=torch.float32)
        output[:2, :2] = linear
        output[:2, 2] = translation
        return output


class SynchronizedImageAugmentor:
    """Sample once and apply exactly the same transform to every branch/view.

    The sampling equations mirror StreamPETR's ``ResizeCropFlipRotImage``.
    Padding is bottom/right only, matching ``mmcv.impad`` for a fixed shape.
    Input images are RGB PIL images and output is ``[N, C, H_pad, W_pad]``.
    """

    def __init__(
        self,
        original_dim: Tuple[int, int],
        final_dim: Tuple[int, int],
        resize_lim: Tuple[float, float],
        bot_pct_lim: Tuple[float, float],
        rand_flip: bool,
        rot_lim: Tuple[float, float] = (0.0, 0.0),
        pad_shape: Optional[Tuple[int, int]] = None,
        mean: Sequence[float] = (103.530, 116.280, 123.675),
        std: Sequence[float] = (57.375, 57.120, 58.395),
        to_rgb: bool = True,
        training: bool = True,
    ) -> None:
        self.original_dim = tuple(original_dim)
        self.final_dim = tuple(final_dim)
        self.resize_lim = tuple(resize_lim)
        self.bot_pct_lim = tuple(bot_pct_lim)
        self.rand_flip = rand_flip
        self.rot_lim = tuple(rot_lim)
        self.pad_shape = tuple(pad_shape or final_dim)
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]
        self.to_rgb = to_rgb
        self.training = training
        if self.pad_shape[0] < final_dim[0] or self.pad_shape[1] < final_dim[1]:
            raise ValueError("pad_shape must contain final_dim")

    def sample(self, rng: Optional[np.random.RandomState] = None) -> ImageAugmentation:
        rng = rng or np.random
        height, width = self.original_dim
        final_h, final_w = self.final_dim
        if self.training:
            resize = float(rng.uniform(*self.resize_lim))
            resize_dims = (int(width * resize), int(height * resize))
            new_w, new_h = resize_dims
            crop_h = int((1 - rng.uniform(*self.bot_pct_lim)) * new_h) - final_h
            crop_w = int(rng.uniform(0, max(0, new_w - final_w)))
            flip = bool(self.rand_flip and rng.choice([0, 1]))
            rotate = float(rng.uniform(*self.rot_lim))
        else:
            resize = max(final_h / height, final_w / width)
            resize_dims = (int(width * resize), int(height * resize))
            new_w, new_h = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * new_h) - final_h
            crop_w = int(max(0, new_w - final_w) / 2)
            flip = False
            rotate = 0.0
        crop = (crop_w, crop_h, crop_w + final_w, crop_h + final_h)
        return ImageAugmentation(
            resize, resize_dims, crop, flip, rotate, self.final_dim, self.pad_shape
        )

    def apply_one(self, image: Image.Image, params: ImageAugmentation) -> Tensor:
        image = image.convert("RGB").resize(params.resize_dims, Image.BILINEAR)
        image = image.crop(params.crop)
        if params.flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if params.rotate:
            image = image.rotate(params.rotate)
        array = np.asarray(image, dtype=np.float32).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        # mmcv images enter NormalizeMultiviewImage as BGR. PIL is RGB, so
        # convert only when reproducing mmcv's to_rgb=False behavior.
        if not self.to_rgb:
            tensor = tensor.flip(0)
        tensor = (tensor - self.mean) / self.std
        # StreamPETR normalizes before PadMultiViewImage, so padded tensor
        # values are zero (rather than normalized black-image values).
        pad_h = params.pad_shape[0] - params.final_dim[0]
        pad_w = params.pad_shape[1] - params.final_dim[1]
        return F.pad(tensor, (0, pad_w, 0, pad_h), value=0.0)

    def apply_many(
        self, images: Sequence[Image.Image], params: ImageAugmentation
    ) -> Tensor:
        return torch.stack([self.apply_one(image, params) for image in images])


class DirtyPathMapper:
    """Map a nuScenes sample_data filename through an explicit path template.

    ``layout`` may use ``{corruption_type}``, ``{severity}``, ``{camera}``,
    ``{relative_path}``, and ``{filename}``.  Example::

        {corruption_type}/{severity}/{relative_path}
    """

    def __init__(self, root: str, layout: str) -> None:
        self.root = Path(root)
        self.layout = layout

    def resolve(self, clean_filename: str, corruption_type: str, severity: Any) -> Path:
        if corruption_type not in SUPPORTED_CORRUPTIONS:
            raise ValueError("unsupported corruption_type: %s" % corruption_type)
        normalized = clean_filename.replace("\\", "/")
        relative = PurePosixPath(normalized)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("clean_filename must be a safe relative path")
        camera = next((part for part in relative.parts if part.startswith("CAM_")), "")
        mapped = self.layout.format(
            corruption_type=corruption_type,
            severity=str(severity),
            camera=camera,
            relative_path=relative.as_posix(),
            filename=relative.name,
        )
        mapped_path = PurePosixPath(mapped)
        if mapped_path.is_absolute() or ".." in mapped_path.parts:
            raise ValueError("layout produced an unsafe path")
        return self.root.joinpath(*mapped_path.parts)


class TemporalOccNuScenesDataset(Dataset):
    """Minimal ``Clean_(t-1) -> (Clean_t, Dirty_t)`` episode dataset.

    Each record must contain ``history_clean`` and ``current_clean`` mappings
    keyed by camera name, plus ``scene_token``, ``sample_token``,
    ``prev_sample_token``, ``timestamp``, ``corruption_type`` and ``severity``.
    ``history_clean`` may also be a list of mappings to reserve a t-2 frame.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        clean_root: str,
        dirty_mapper: DirtyPathMapper,
        augmentor: SynchronizedImageAugmentor,
        camera_order: Sequence[str] = NUSCENES_CAMERA_ORDER,
        require_files: bool = True,
    ) -> None:
        self.records = list(records)
        self.clean_root = Path(clean_root)
        self.dirty_mapper = dirty_mapper
        self.augmentor = augmentor
        self.camera_order = tuple(camera_order)
        self.require_files = require_files

    def __len__(self) -> int:
        return len(self.records)

    def _ordered_paths(self, filenames: Mapping[str, str]) -> List[Path]:
        missing = set(self.camera_order).difference(filenames)
        if missing:
            raise KeyError("missing cameras: %s" % sorted(missing))
        return [self.clean_root / filenames[camera] for camera in self.camera_order]

    def _open(self, paths: Sequence[Path]) -> List[Image.Image]:
        if self.require_files:
            absent = [str(path) for path in paths if not path.is_file()]
            if absent:
                raise FileNotFoundError("missing episode images: %s" % absent)
        return [Image.open(path).convert("RGB") for path in paths]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        histories = record["history_clean"]
        if isinstance(histories, Mapping):
            histories = [histories]
        current_names = record["current_clean"]
        clean_paths = self._ordered_paths(current_names)
        dirty_paths = [
            self.dirty_mapper.resolve(
                current_names[camera], record["corruption_type"], record["severity"]
            )
            for camera in self.camera_order
        ]
        history_paths = [self._ordered_paths(history) for history in histories]
        params = self.augmentor.sample()
        current_clean = self.augmentor.apply_many(self._open(clean_paths), params)
        current_dirty = self.augmentor.apply_many(self._open(dirty_paths), params)
        history_clean = torch.stack(
            [self.augmentor.apply_many(self._open(paths), params) for paths in history_paths]
        )
        metadata = {
            key: record[key]
            for key in (
                "scene_token",
                "sample_token",
                "prev_sample_token",
                "timestamp",
                "corruption_type",
                "severity",
            )
        }
        metadata.update(
            camera=self.camera_order,
            augmentation=asdict(params),
            image_aug_matrix=params.matrix(),
            current_clean_paths=tuple(map(str, clean_paths)),
            current_dirty_paths=tuple(map(str, dirty_paths)),
        )
        return {
            "history_clean": history_clean,
            "current_dirty": current_dirty,
            "current_clean": current_clean,
            "metadata": metadata,
        }


__all__ = [
    "DirtyPathMapper",
    "ImageAugmentation",
    "NUSCENES_CAMERA_ORDER",
    "SUPPORTED_CORRUPTIONS",
    "SynchronizedImageAugmentor",
    "TemporalOccNuScenesDataset",
]
