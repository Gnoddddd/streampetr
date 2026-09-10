"""Project-side StreamPETR pipeline hooks for canonical paired images."""

from __future__ import annotations

from copy import deepcopy

from mmdet.datasets import DATASETS
from mmdet.datasets.builder import PIPELINES
from projects.mmdet3d_plugin.datasets.nuscenes_dataset import CustomNuScenesDataset
from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import (
    ResizeCropFlipRotImage,
)


@PIPELINES.register_module()
class OEPRecordResizeCropFlipRotImage(ResizeCropFlipRotImage):
    """Run the official augmentation and record its exact sampled parameters."""

    def __call__(self, results):
        parameters = self._sample_augmentation()
        sampler = self._sample_augmentation
        self._sample_augmentation = lambda: parameters
        try:
            results = super().__call__(results)
        finally:
            self._sample_augmentation = sampler
        resize, resize_dims, crop, flip, rotate = parameters
        results["oep_ida_params"] = dict(
            resize=float(resize), resize_dims=tuple(resize_dims),
            crop=tuple(crop), flip=bool(flip), rotate=float(rotate),
        )
        return results


@DATASETS.register_module()
class OEPSequenceNuScenesDataset(CustomNuScenesDataset):
    """Use the official dataset while exposing image-augmentation replay data."""

    def __init__(self, pipeline, *args, **kwargs):
        pipeline = deepcopy(pipeline)
        for transform in pipeline:
            if transform.get("type") == "ResizeCropFlipRotImage":
                transform["type"] = "OEPRecordResizeCropFlipRotImage"
            if transform.get("type") == "Collect3D":
                meta_keys = list(transform.get("meta_keys", ()))
                for key in ("sample_idx", "frame_idx", "oep_ida_params"):
                    if key not in meta_keys:
                        meta_keys.append(key)
                transform["meta_keys"] = tuple(meta_keys)
        super().__init__(pipeline=pipeline, *args, **kwargs)
