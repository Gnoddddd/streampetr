"""Convert sampled episodes to the existing pixel-domain corruption protocol."""

from typing import Sequence

import numpy as np
import torch
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


def fault_normalized_images(
    clean: Tensor,
    episode: FaultEpisode,
    mean: Sequence[float],
    std: Sequence[float],
    scene_tokens: Sequence[str],
    camera_names: Sequence[str] = CAMERA_NAMES,
) -> Tensor:
    """Apply the canonical corruption after returning normalized images to pixels.

    Input is [B,T,C,3,H,W]. The output retains its device, dtype and layout.
    """
    if clean.ndim != 6 or clean.shape[2] != len(camera_names) or clean.shape[3] != 3:
        raise ValueError("clean images must have shape [B,T,C,3,H,W]")
    if len(scene_tokens) != clean.shape[0] or len(episode.active) != clean.shape[1]:
        raise ValueError("scene tokens or episode length do not match the image batch")
    pixel_mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    pixel_std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    output = clean.detach().float().cpu().clone()
    transform = ApplyPartialObservation(
        camera_names=camera_names, training=False, visual_ablation_mode="full"
    )
    for batch_index, scene_token in enumerate(scene_tokens):
        transform.schedule = episode_schedule(episode, str(scene_token), camera_names)
        for frame_index in range(clean.shape[1]):
            images = []
            for camera_index in range(clean.shape[2]):
                normalized = output[batch_index, frame_index, camera_index].permute(1, 2, 0).numpy()
                pixels = normalized * pixel_std + pixel_mean
                images.append(np.clip(pixels, 0, 255).astype(np.float32))
            result = transform(dict(
                img=images, sample_idx=f"{scene_token}:{frame_index}",
                scene_token=str(scene_token), frame_idx=frame_index,
            ))
            for camera_index, pixels in enumerate(result["img"]):
                normalized = (pixels.astype(np.float32) - pixel_mean) / pixel_std
                output[batch_index, frame_index, camera_index] = torch.from_numpy(
                    normalized
                ).permute(2, 0, 1)
    return output.to(device=clean.device, dtype=clean.dtype)
