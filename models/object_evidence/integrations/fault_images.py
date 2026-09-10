"""Canonical raw-pixel corruption and exact augmentation replay."""

import os
from typing import Sequence

import mmcv
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from datasets.corruption import ApplyPartialObservation, CAMERA_NAMES
from protocols.partial_observation import ProtocolEvent, ProtocolSchedule
from ..fault_sampler import FaultEpisode


def episode_schedule(
    episode: FaultEpisode, scene_token: str = "*", camera_names: Sequence[str] = CAMERA_NAMES
) -> ProtocolSchedule:
    camera = camera_names[episode.spec.camera]
    active = [index for index, enabled in enumerate(episode.active) if enabled]
    if not active:
        return ProtocolSchedule()
    values = dict(
        start_frame=min(active), end_frame=max(active), failed_cameras=[],
        dark={}, motion_blur={},
    )
    if episode.spec.fault_type == "crash":
        values["failed_cameras"] = [camera]
    elif episode.spec.fault_type == "dark":
        values["dark"] = {camera: episode.spec.severity}
    elif episode.spec.fault_type == "blur":
        values["motion_blur"] = {camera: episode.spec.severity}
    else:
        raise ValueError(f"unsupported fault type {episode.spec.fault_type}")
    return ProtocolSchedule({str(scene_token): [ProtocolEvent(**values)]})


def corrupt_raw_images(
    images: Sequence[np.ndarray], episode: FaultEpisode, scene_token: str,
    camera_names: Sequence[str],
):
    """Apply the existing protocol transform before any image augmentation."""
    transform = ApplyPartialObservation(
        camera_names=camera_names, training=False, visual_ablation_mode="full"
    )
    transform.schedule = episode_schedule(episode, str(scene_token), camera_names)
    return transform(dict(
        img=[np.array(image, copy=True) for image in images],
        sample_idx=f"{scene_token}:raw", scene_token=str(scene_token), frame_idx=0,
    ))["img"]


def replay_stream_augmentation(
    raw_images: Sequence[np.ndarray], parameters, mean, std, to_rgb=True,
    pad_divisor: int = 32,
) -> Tensor:
    """Replay StreamPETR's recorded resize/crop/flip/rotation exactly."""
    output = []
    for raw in raw_images:
        image = Image.fromarray(np.uint8(raw))
        image = image.resize(tuple(parameters["resize_dims"]))
        image = image.crop(tuple(parameters["crop"]))
        if parameters["flip"]:
            image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
        image = image.rotate(float(parameters["rotate"]))
        array = mmcv.imnormalize(
            np.array(image).astype(np.float32), np.asarray(mean, dtype=np.float32),
            np.asarray(std, dtype=np.float32), to_rgb
        )
        array = mmcv.impad_to_multiple(array, int(pad_divisor), pad_val=0)
        output.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(output)


def stream_fault_images_from_raw(clean: Tensor, img_metas, decisions) -> Tensor:
    """Create StreamPETR fault tensors from source files using recorded IDA."""
    output = clean.detach().clone()
    current_metas = img_metas[-1]
    for batch_index, decision in enumerate(decisions):
        if not decision.fault_active:
            continue
        meta = current_metas[batch_index]
        raw = [mmcv.imread(path, flag="unchanged") for path in meta["filename"]]
        episode = FaultEpisode(decision.fault_spec, (True,))
        fault = corrupt_raw_images(raw, episode, decision.scene_token, CAMERA_NAMES)
        norm = meta["img_norm_cfg"]
        replayed = replay_stream_augmentation(
            fault, meta["oep_ida_params"], norm["mean"], norm["std"],
            norm.get("to_rgb", True),
        )
        if replayed.shape != clean[batch_index, -1].shape:
            raise RuntimeError("replayed StreamPETR fault image shape differs from clean")
        output[batch_index, -1] = replayed.to(clean)
    return output


def bevdepth_fault_images_from_raw(
    clean: Tensor, img_metas, fault_specs, data_root: str,
    mean, std, camera_names: Sequence[str], to_rgb: bool = True,
) -> Tensor:
    """Create persistent current+previous BEVDepth faults before shared IDA."""
    from bevdepth.datasets.nusc_det_dataset import img_transform

    output = clean.detach().clone()
    for batch_index, spec in enumerate(fault_specs):
        if spec is None:
            continue
        meta = img_metas[batch_index]
        filenames = meta["oep_raw_filenames"]
        parameters = meta["oep_ida_params"]
        for sweep_index, sweep_filenames in enumerate(filenames):
            raw = [
                np.array(Image.open(os.path.join(data_root, filename)))
                for filename in sweep_filenames
            ]
            episode = FaultEpisode(spec, (True,))
            fault = corrupt_raw_images(raw, episode, meta["scene_token"], camera_names)
            replayed = []
            for camera_index, image in enumerate(fault):
                values = parameters[camera_index]
                augmented, _ = img_transform(
                    Image.fromarray(np.uint8(image)),
                    resize=values["resize"], resize_dims=tuple(values["resize_dims"]),
                    crop=tuple(values["crop"]), flip=values["flip"],
                    rotate=values["rotate"],
                )
                normalized = mmcv.imnormalize(
                    np.array(augmented), np.asarray(mean, dtype=np.float32),
                    np.asarray(std, dtype=np.float32), to_rgb
                )
                replayed.append(torch.from_numpy(normalized).permute(2, 0, 1))
            replayed = torch.stack(replayed)
            if replayed.shape != clean[batch_index, sweep_index].shape:
                raise RuntimeError("replayed BEVDepth fault image shape differs from clean")
            output[batch_index, sweep_index] = replayed.to(clean)
    return output
