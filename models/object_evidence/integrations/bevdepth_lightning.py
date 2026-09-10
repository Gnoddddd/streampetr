"""Native BEVDepth R0/OE training and optional Lightning entry point."""

from __future__ import annotations

import argparse
import runpy
from functools import partial
from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..adapters.base import align_by_object_id
from ..adapters.bevdepth import BEVDepthAdapter, camera_reliability, weighted_depth_loss
from ..fault_sampler import FaultEpisode
from ..observability import camera_support, observability_gap
from ..paired_training import ObjectEvidenceObjective, temporary_native_teacher
from .fault_images import bevdepth_fault_images_from_raw
from .sequence_episode import deterministic_sample_faults


def torch_voxel_pooling_train(
    geom_xyz: Tensor, input_features: Tensor, voxel_num: Tensor
) -> Tensor:
    """Differentiable reference implementation of BEVDepth voxel pooling.

    The official CUDA extension sums all valid point features into the x/y BEV
    cell (z is used only for bounds checking).  ``index_add`` has the same
    forward and feature-gradient semantics and keeps native training usable on
    machines where that optional extension was not compiled.
    """
    batch_size = input_features.shape[0]
    features = input_features.reshape(batch_size, -1, input_features.shape[-1])
    geometry = geom_xyz.reshape(batch_size, -1, 3)
    if geometry.shape[1] != features.shape[1]:
        raise ValueError("geometry and input features must contain the same points")
    size_x, size_y, size_z = (int(value) for value in voxel_num)
    valid = (
        (geometry[..., 0] >= 0) & (geometry[..., 0] < size_x)
        & (geometry[..., 1] >= 0) & (geometry[..., 1] < size_y)
        & (geometry[..., 2] >= 0) & (geometry[..., 2] < size_z)
    )
    batch = torch.arange(batch_size, device=geometry.device)[:, None]
    flat_index = (
        batch * (size_y * size_x)
        + geometry[..., 1].long() * size_x
        + geometry[..., 0].long()
    )
    output = features.new_zeros(batch_size * size_y * size_x, features.shape[-1])
    output = output.index_add(0, flat_index[valid], features[valid])
    return output.view(batch_size, size_y, size_x, features.shape[-1]).permute(0, 3, 1, 2)


def install_voxel_pooling_fallback() -> bool:
    """Install the reference op only when BEVDepth's optional CUDA op is absent."""
    import bevdepth.layers.backbones.base_lss_fpn as base_lss_fpn

    if hasattr(base_lss_fpn, "voxel_pooling_train"):
        return False
    base_lss_fpn.voxel_pooling_train = torch_voxel_pooling_train
    return True


def reshape_depth_prediction(prediction: Tensor, batch_size: int, num_cameras: int) -> Tensor:
    """Convert official [B*C,D,H,W] depth output to explicit camera layout."""
    if prediction.ndim == 5:
        if prediction.shape[:2] != (batch_size, num_cameras):
            raise ValueError("five-dimensional depth prediction has wrong B/C layout")
        return prediction
    if prediction.ndim != 4 or prediction.shape[0] != batch_size * num_cameras:
        raise ValueError(
            "depth prediction must be [B*C,D,H,W] with runtime-derived B and C"
        )
    return prediction.reshape(batch_size, num_cameras, *prediction.shape[1:])


def downsample_depth_labels(owner, labels: Tensor) -> Tensor:
    """Official BEVDepth nearest-depth bin target construction."""
    batch_size, num_cameras, height, width = labels.shape
    factor = int(owner.downsample_factor)
    labels = labels.view(
        batch_size * num_cameras, height // factor, factor,
        width // factor, factor, 1,
    ).permute(0, 1, 3, 5, 2, 4).contiguous()
    labels = labels.view(-1, factor * factor)
    nearest = torch.where(labels == 0, torch.full_like(labels, 1e5), labels).min(dim=-1).values
    nearest = nearest.view(
        batch_size * num_cameras, height // factor, width // factor
    )
    bins = (nearest - (owner.dbound[0] - owner.dbound[2])) / owner.dbound[2]
    bins = torch.where(
        (bins < owner.depth_channels + 1) & (bins >= 0), bins, torch.zeros_like(bins)
    )
    return F.one_hot(bins.long(), num_classes=owner.depth_channels + 1).view(
        -1, owner.depth_channels + 1
    )[:, 1:].float()


def native_weighted_depth_loss(owner, labels: Tensor, prediction: Tensor, weights: Tensor):
    if labels.ndim == 5:
        labels = labels[:, 0]
    batch_size, num_cameras = labels.shape[:2]
    prediction = reshape_depth_prediction(prediction, batch_size, num_cameras)
    downsampled = downsample_depth_labels(owner, labels)
    _, _, depth_bins, height, width = prediction.shape
    expected = batch_size * num_cameras * height * width * depth_bins
    if downsampled.numel() != expected:
        raise ValueError("downsampled depth target and prediction shapes disagree")
    target = downsampled.reshape(batch_size, num_cameras, height, width, depth_bins)
    return weighted_depth_loss(prediction, target, weights)


def bev_lidar_to_image(mats: Dict[str, Tensor]) -> Tensor:
    """Projection for BDA-augmented ego boxes into IDA-augmented camera images."""
    sensor_to_ego = mats["sensor2ego_mats"][:, 0]
    intrinsics = mats["intrin_mats"][:, 0]
    ida = mats["ida_mats"][:, 0]
    inverse_bda = torch.linalg.inv(mats["bda_mat"])[:, None]
    return ida @ intrinsics @ torch.linalg.inv(sensor_to_ego) @ inverse_bda


def initialize_native_state(owner, config: Dict) -> None:
    owner.object_evidence_enabled = bool(config.get("enabled", True))
    owner._oep_seed = int(config.get("seed", 2026))
    owner._oep_pair_probability = float(config.get("pair_probability", 0.5))
    owner._oep_force_paired = False
    owner._oep_iteration = 0
    owner.object_evidence = ObjectEvidenceObjective(
        teacher_dim=None,
        lambda_oe=float(config.get("lambda_oe", 0.5)),
        lambda_pg=0.0,
        warmup_iters=int(config.get("auxiliary_warmup_iters", 1000)),
    ) if owner.object_evidence_enabled else None


def native_paired_step(owner, batch, fault_specs=None):
    """Execute one BEVDepth paired-fault R0/OE native training step."""
    sweep_imgs, mats, timestamps, img_metas, gt_boxes, gt_labels, depth_labels = batch
    batch_size, clip_length, num_cameras = sweep_imgs.shape[:3]
    if num_cameras != len(owner.ida_aug_conf["cams"]):
        raise ValueError("runtime camera count disagrees with BEVDepth camera configuration")
    sample_tokens = [str(meta.get("token", index)) for index, meta in enumerate(img_metas)]
    if fault_specs is None:
        fault_specs = deterministic_sample_faults(
            owner._oep_seed, sample_tokens, owner._oep_pair_probability,
            num_cameras=num_cameras, force_fault=owner._oep_force_paired,
        )
    if len(fault_specs) != batch_size or not any(spec is not None for spec in fault_specs):
        raise ValueError("native paired step requires at least one batch fault")
    fault_images = bevdepth_fault_images_from_raw(
        sweep_imgs, img_metas, fault_specs, owner.data_root,
        owner.img_conf["img_mean"], owner.img_conf["img_std"],
        owner.ida_aug_conf["cams"], owner.img_conf.get("to_rgb", True),
    )
    adapter = BEVDepthAdapter(owner.model)
    clean_feature = None
    if owner.object_evidence.lambda_oe > 0:
        # BaseLSSFPN selects its train-time voxel-pooling path from this flag,
        # while all parameterized children remain in eval and no-grad mode.
        with temporary_native_teacher(
            owner.model, (owner.model.backbone,)
        ), adapter.capture() as capture:
            owner.model(sweep_imgs, mats, timestamps)
        clean_feature = capture.tensor.detach()
    with adapter.capture() as capture:
        predictions, depth_prediction = owner.model(fault_images, mats, timestamps)
    fault_feature = capture.tensor
    targets = owner.model.get_targets(gt_boxes, gt_labels)
    detection_loss = owner.model.loss(targets, predictions)
    reliability = sweep_imgs.new_ones(batch_size, num_cameras)
    for batch_index, spec in enumerate(fault_specs):
        if spec is not None:
            reliability[batch_index] = camera_reliability(
                num_cameras, spec.camera, spec.fault_type, spec.severity
            ).to(sweep_imgs)
    depth_loss = native_weighted_depth_loss(owner, depth_labels, depth_prediction, reliability)
    oe_loss = detection_loss * 0
    gap_tensor = detection_loss.new_empty((0,))
    matched = 0
    if clean_feature is not None:
        identities = [list(range(len(boxes))) for boxes in gt_boxes]
        clean_evidence = adapter.extract(clean_feature, gt_boxes, identities, gt_labels)
        fault_evidence = adapter.extract(fault_feature, gt_boxes, identities, gt_labels)
        clean_evidence, fault_evidence = align_by_object_id(clean_evidence, fault_evidence)
        projections = bev_lidar_to_image(mats)
        strengths = sweep_imgs.new_zeros(batch_size, num_cameras)
        for batch_index, spec in enumerate(fault_specs):
            if spec is not None:
                strengths[batch_index, spec.camera] = spec.strength
        all_gaps, all_supported = [], []
        for batch_index, boxes in enumerate(gt_boxes):
            support = camera_support(
                boxes.to(sweep_imgs), projections[batch_index], sweep_imgs.shape[-2:]
            )
            gap, supported = observability_gap(support, strengths[batch_index])
            all_gaps.append(gap)
            all_supported.append(supported)
        selected_gap, selected_valid = [], []
        for batch_index, gt_index in zip(
            clean_evidence.batch_indices.tolist(), clean_evidence.gt_indices.tolist()
        ):
            selected_gap.append(all_gaps[batch_index][gt_index])
            selected_valid.append(all_supported[batch_index][gt_index])
        gap_tensor = torch.stack(selected_gap) if selected_gap else detection_loss.new_empty((0,))
        valid = (
            torch.stack(selected_valid).bool() if selected_valid
            else torch.empty(0, dtype=torch.bool, device=detection_loss.device)
        )
        valid = valid & clean_evidence.valid_mask & fault_evidence.valid_mask
        auxiliary = owner.object_evidence(
            clean_evidence.tokens, fault_evidence.tokens, gap_tensor, valid,
            owner._oep_iteration,
        )
        oe_loss = auxiliary["loss_object_evidence_total"]
        raw_oe = auxiliary["loss_object_evidence"].detach()
        matched = len(gap_tensor)
    else:
        raw_oe = detection_loss.detach() * 0
    owner._oep_iteration += 1
    total = detection_loss + depth_loss + oe_loss
    diagnostics = dict(
        detection_loss=detection_loss.detach(), depth_loss=depth_loss.detach(),
        oe_loss=raw_oe, total_loss=total.detach(),
        gap_mean=gap_tensor.mean().detach() if len(gap_tensor) else total.detach() * 0,
        gap_max=gap_tensor.max().detach() if len(gap_tensor) else total.detach() * 0,
        matched_objects=matched,
        fault_specs=fault_specs,
        episode=FaultEpisode(next(spec for spec in fault_specs if spec is not None),
                             tuple(True for _ in range(clip_length))),
    )
    owner._object_evidence_last = diagnostics
    return total


try:  # The StreamPETR environment intentionally does not install Lightning.
    import pytorch_lightning as pl
    from bevdepth.exps.nuscenes.base_exp import BEVDepthLightningModel
    from bevdepth.models.base_bev_depth import BaseBEVDepth
except (ImportError, ModuleNotFoundError):  # pragma: no cover - exercised in BEVDepth env
    pl = None
    BEVDepthLightningModel = None


if BEVDepthLightningModel is not None:
    class OEBevDepthLightningModel(BEVDepthLightningModel):
        """Official Lightning model with an exact disabled path and native OE step."""

        def __init__(self, object_evidence=None, **kwargs):
            super().__init__(**kwargs)
            install_voxel_pooling_fallback()
            self.key_idxes = [-1]
            self.head_conf["bev_backbone_conf"]["in_channels"] = 160
            self.head_conf["bev_neck_conf"]["in_channels"] = [160, 160, 320, 640]
            self.head_conf["train_cfg"]["code_weights"] = [1.0] * 10
            self.model = BaseBEVDepth(
                self.backbone_conf, self.head_conf, is_train_depth=True
            )
            initialize_native_state(self, dict(object_evidence or {}))

        def training_step(self, batch, batch_idx):
            if not self.object_evidence_enabled:
                return super().training_step(batch)
            tokens = [str(meta.get("token", index)) for index, meta in enumerate(batch[3])]
            fault_specs = deterministic_sample_faults(
                self._oep_seed, tokens, self._oep_pair_probability,
                num_cameras=batch[0].shape[2], force_fault=self._oep_force_paired,
            )
            if not any(spec is not None for spec in fault_specs):
                self._oep_iteration += 1
                return super().training_step(batch)
            loss = native_paired_step(self, batch, fault_specs)
            for name in ("detection_loss", "depth_loss", "oe_loss", "total_loss"):
                self.log(name, self._object_evidence_last[name])
            return loss

        def train_dataloader(self):
            from datasets.bevdepth_object_evidence import OEPairedNuscDetDataset
            from bevdepth.datasets.nusc_det_dataset import collate_fn

            dataset = OEPairedNuscDetDataset(
                ida_aug_conf=self.ida_aug_conf, bda_aug_conf=self.bda_aug_conf,
                classes=self.class_names, data_root=self.data_root,
                info_paths=self.train_info_paths, is_train=True,
                use_cbgs=self.data_use_cbgs, img_conf=self.img_conf,
                num_sweeps=self.num_sweeps, sweep_idxes=self.sweep_idxes,
                key_idxes=self.key_idxes, return_depth=self.data_return_depth,
                use_fusion=self.use_fusion,
            )
            return torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size_per_device, num_workers=4,
                drop_last=True, shuffle=False,
                collate_fn=partial(
                    collate_fn,
                    is_return_depth=self.data_return_depth or self.use_fusion,
                ),
                sampler=None,
            )


def main():  # pragma: no cover - requires the official BEVDepth Lightning environment
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("r0", "oe"), required=True)
    parser.add_argument("--ckpt-path", required=True)
    args, trainer_args = parser.parse_known_args()
    if pl is None:
        raise RuntimeError("pytorch_lightning is required for the BEVDepth training entry")
    config = runpy.run_path(args.config)
    method = dict(config["object_evidence"])
    method["lambda_oe"] = 0.0 if args.mode == "r0" else 0.5
    model = OEBevDepthLightningModel(
        object_evidence=method, data_root=config["data_root"],
        batch_size_per_device=config["batch_size_per_device"],
    )
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
    trainer = pl.Trainer(max_steps=int(config["adaptation_max_steps"]), gpus=1)
    trainer.fit(model)


if __name__ == "__main__":
    main()
