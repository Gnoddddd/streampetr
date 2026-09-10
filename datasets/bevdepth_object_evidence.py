"""Project-side BEVDepth dataset metadata for canonical paired images."""

from __future__ import annotations

from bevdepth.datasets.nusc_det_dataset import NuscDetDataset


class OEPairedNuscDetDataset(NuscDetDataset):
    """Official BEVDepth dataset with raw paths and sampled IDA replay metadata."""

    def sample_ida_augmentation(self):
        parameters = super().sample_ida_augmentation()
        if getattr(self, "_oep_capture_ida", False):
            self._oep_ida_parameters.append(parameters)
        return parameters

    def get_image(self, cam_infos, cams, lidar_infos=None):
        self._oep_capture_ida = True
        self._oep_ida_parameters = []
        try:
            result = super().get_image(cam_infos, cams, lidar_infos)
        finally:
            self._oep_capture_ida = False
        if len(self._oep_ida_parameters) != len(cams):
            raise RuntimeError("failed to capture one BEVDepth IDA transform per camera")
        result[6]["oep_raw_filenames"] = [
            [sweep[cam]["filename"] for cam in cams] for sweep in cam_infos
        ]
        result[6]["oep_ida_params"] = [
            dict(
                resize=float(values[0]), resize_dims=tuple(values[1]),
                crop=tuple(values[2]), flip=bool(values[3]), rotate=float(values[4]),
            )
            for values in self._oep_ida_parameters
        ]
        return result

    def __getitem__(self, idx):
        effective_index = self.sample_indices[idx] if self.use_cbgs else idx
        result = super().__getitem__(idx)
        result[7]["scene_token"] = self.infos[effective_index]["scene_token"]
        return result
