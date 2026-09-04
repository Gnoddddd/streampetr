"""Training-only nuScenes plumbing for aligned LiDAR point-count evidence."""

from __future__ import annotations

import numpy as np
import torch
from mmdet.datasets import DATASETS
from mmdet.datasets.builder import PIPELINES
from .nuscenes_wrapper import EvidenceNuScenesDataset


@DATASETS.register_module()
class LiDARPrivilegedNuScenesDataset(EvidenceNuScenesDataset):
    """Expose annotation point counts only on the training annotation path."""

    def get_ann_info(self, index):
        result = super().get_ann_info(index)
        info = self.data_infos[index]
        mask = info["valid_flag"] if self.use_valid_flag else info["num_lidar_pts"] > 0
        result["gt_lidar_point_counts"] = np.asarray(info["num_lidar_pts"])[mask]
        return result


@PIPELINES.register_module()
class AttachLiDARPointCounts:
    """Align original point counts after standard range/name GT filtering."""

    def __call__(self, results):
        original_boxes = results["ann_info"]["gt_bboxes_3d"].gravity_center
        current_boxes = results["gt_bboxes_3d"].gravity_center
        original_counts = np.asarray(
            results["ann_info"]["gt_lidar_point_counts"], dtype=np.float32
        )
        if len(current_boxes) == 0:
            counts = np.empty(0, dtype=np.float32)
        else:
            distances = torch.linalg.vector_norm(
                current_boxes.float()[:, None, :] - original_boxes.float()[None, :, :],
                dim=-1,
            )
            minimum, indexes = distances.min(dim=1)
            if bool(torch.any(minimum > 1e-4)):
                raise RuntimeError(
                    "could not align LiDAR point counts to filtered GT: "
                    f"max nearest-center distance={float(minimum.max())}"
                )
            counts = original_counts[indexes.cpu().numpy()]
        results["gt_lidar_point_counts"] = counts
        return results
