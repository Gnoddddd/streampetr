#!/usr/bin/env python3
"""One real nuScenes train-batch optimization step for native OE integrations."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import random
import sys
import tempfile
import types
from pathlib import Path

# Required only when --disabled-equivalence enables strict CUDA determinism;
# setting it before the first CUDA context is harmless for normal smoke runs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import mmcv
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
STREAM = ROOT / "repos/StreamPETR"
BEVDEPTH = ROOT / "repos/BEVDepth"
PYTHON = "/home/research/miniconda3/envs/streampetr/bin/python"
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("streampetr", "bevdepth"), required=True)
    parser.add_argument("--mode", choices=("r0", "oe"), required=True)
    parser.add_argument("--dataset-index", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--disabled-equivalence", action="store_true")
    parser.add_argument("--temporal-continuity", action="store_true")
    return parser.parse_args()


def seed_all(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_sum(losses):
    values = []
    for name, value in losses.items():
        if "loss" not in name:
            continue
        values.extend(value if isinstance(value, (list, tuple)) else [value])
    return sum(values)


def tensor_leaves(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        return [tensor for key in sorted(value) for tensor in tensor_leaves(value[key])]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in tensor_leaves(item)]
    return []


def max_abs_difference(left, right):
    left_tensors, right_tensors = tensor_leaves(left), tensor_leaves(right)
    if len(left_tensors) != len(right_tensors):
        raise RuntimeError("equivalence outputs have different tensor structures")
    differences = []
    for left_tensor, right_tensor in zip(left_tensors, right_tensors):
        if left_tensor.shape != right_tensor.shape:
            raise RuntimeError("equivalence outputs have different tensor shapes")
        differences.append(float((left_tensor.float() - right_tensor.float()).abs().max()))
    return max(differences, default=0.0)


def detached_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: detached_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(detached_tree(item) for item in value)
    if isinstance(value, list):
        return [detached_tree(item) for item in value]
    return value


def gradient_and_step(model, total, optimizer):
    optimizer.zero_grad()
    total.backward()
    squared = total.new_zeros(())
    selected = None
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += parameter.grad.detach().float().square().sum()
            if selected is None and bool(parameter.grad.detach().abs().max() > 0):
                selected = parameter
    if selected is None:
        raise RuntimeError("detector received no nonzero gradient")
    before = selected.detach().clone()
    optimizer.step()
    delta = float((selected.detach() - before).abs().max())
    return float(squared.sqrt()), delta


def stream_smoke(args):
    sys.path.insert(0, str(STREAM))
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import build_optimizer, load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    config_path = ROOT / "configs/object_evidence/streampetr_r0_full.py"
    config = Config.fromfile(str(config_path))
    import_modules_from_strings(**config.custom_imports)
    config.model.object_evidence.lambda_oe = 0.0 if args.mode == "r0" else 0.5
    dataset = build_dataset(config.data.train)
    index = int(10 if args.dataset_index is None else args.dataset_index) % len(dataset)
    seed_all()
    sample = dataset[index]
    device_index = int(args.device.split(":")[-1])
    data = scatter(collate([sample], samples_per_gpu=1), [device_index])[0]
    model = build_model(config.model, train_cfg=config.get("train_cfg"), test_cfg=config.get("test_cfg"))
    load_checkpoint(model, str(ROOT / config.load_from), map_location="cpu", strict=False)
    model = model.to(args.device).train()
    model._oep_force_paired = True
    if args.mode == "oe":
        model._oep_iteration = model.object_evidence.warmup_iters
    optimizer = build_optimizer(model, config.optimizer)
    torch.cuda.reset_peak_memory_stats()
    losses = model(return_loss=True, **data)
    total = tensor_sum(losses)
    original = total - losses.get("loss_oe", total.new_zeros(()))
    gradient, delta = gradient_and_step(model, total, optimizer)
    episode = model._object_evidence_last_episode
    result = dict(
        architecture="streampetr", mode=args.mode,
        sample_token=str(dataset.data_infos[index]["token"]),
        scene_token=str(dataset.data_infos[index]["scene_token"]),
        fault_camera=episode.spec.camera, fault_type=episode.spec.fault_type,
        severity=episode.spec.severity,
        original_detection_loss=float(original.detach()), depth_loss=None,
        oe_loss=float(losses.get("oe_value", total.new_zeros(())).detach()),
        total_loss=float(total.detach()),
        observability_gap_mean=float(losses.get("oe_gap_mean", total.new_zeros(())).detach()),
        observability_gap_max=float(losses.get("oe_gap_max", total.new_zeros(())).detach()),
        matched_object_count=int(losses.get("oe_matched_objects", total.new_zeros(()))),
        detector_gradient_norm=gradient, detector_parameter_delta=delta,
        peak_cuda_memory_mb=torch.cuda.max_memory_allocated() / 1024 ** 2,
    )
    return result


def stream_disabled_equivalence(args):
    sys.path.insert(0, str(STREAM))
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from projects.mmdet3d_plugin.models.detectors.petr3d import Petr3D
    from models.object_evidence.adapters.streampetr import StreamPETRAdapter

    config = Config.fromfile(str(ROOT / "configs/object_evidence/streampetr_r0_full.py"))
    import_modules_from_strings(**config.custom_imports)
    config.model.object_evidence.enabled = False
    dataset = build_dataset(config.data.train)
    index = int(10 if args.dataset_index is None else args.dataset_index) % len(dataset)
    seed_all()
    sample = dataset[index]
    device_index = int(args.device.split(":")[-1])
    data = scatter(collate([sample], samples_per_gpu=1), [device_index])[0]
    model = build_model(config.model, train_cfg=config.get("train_cfg"), test_cfg=config.get("test_cfg"))
    load_checkpoint(model, str(ROOT / config.load_from), map_location="cpu", strict=False)
    model = model.to(args.device).train()
    initial_buffers = {
        key: value.detach().cpu().clone() for key, value in model.named_buffers()
    }

    def execute(use_wrapper):
        for key, value in model.named_buffers():
            value.copy_(initial_buffers[key].to(value))
        model.pts_bbox_head.reset_memory()
        seed_all()
        outputs = []
        handle = model.pts_bbox_head.register_forward_hook(
            lambda _module, _inputs, value: outputs.append(detached_tree(value))
        )
        try:
            with torch.no_grad():
                arguments = copy.deepcopy(data)
                losses = (
                    model.forward_train(**arguments) if use_wrapper
                    else Petr3D.forward_train(model, **arguments)
                )
        finally:
            handle.remove()
        return detached_tree(losses), outputs[-1], detached_tree(
            StreamPETRAdapter(model).snapshot_memory()
        )

    disabled_loss, disabled_output, disabled_memory = execute(True)
    vanilla_loss, vanilla_output, vanilla_memory = execute(False)
    if set(disabled_loss) != set(vanilla_loss):
        raise RuntimeError("disabled StreamPETR loss keys differ from vanilla")
    return dict(
        architecture="streampetr", sample_token=str(dataset.data_infos[index]["token"]),
        scene_token=str(dataset.data_infos[index]["scene_token"]),
        loss_keys=sorted(disabled_loss),
        loss_max_abs_diff=max_abs_difference(disabled_loss, vanilla_loss),
        output_max_abs_diff=max_abs_difference(disabled_output, vanilla_output),
        memory_max_abs_diff=max_abs_difference(disabled_memory, vanilla_memory),
    )


def stream_temporal_continuity(args):
    sys.path.insert(0, str(STREAM))
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from models.object_evidence.adapters.streampetr import (
        StreamPETRAdapter, memory_checksum,
    )

    config = Config.fromfile(str(ROOT / "configs/object_evidence/streampetr_oe_full.py"))
    import_modules_from_strings(**config.custom_imports)
    dataset = build_dataset(config.data.train)
    starts = [0] + (np.nonzero(dataset.flag[1:] != dataset.flag[:-1])[0] + 1).tolist()
    candidates = [
        start for position, start in enumerate(starts[:-1])
        if starts[position + 1] - start >= 6 and start >= 10
    ]
    if not candidates:
        raise RuntimeError("no six-frame StreamPETR training sequence found")
    start = candidates[0]
    next_start = next(value for value in starts if value > start)
    model = build_model(config.model, train_cfg=config.get("train_cfg"), test_cfg=config.get("test_cfg"))
    load_checkpoint(model, str(ROOT / config.load_from), map_location="cpu", strict=False)
    model = model.to(args.device).train()
    model._oep_force_fault_sequence = True
    model._oep_iteration = model.object_evidence.warmup_iters
    device_index = int(args.device.split(":")[-1])
    trace = []

    def execute(index):
        seed_all(2026 + index)
        sample = dataset[index]
        data = scatter(collate([sample], samples_per_gpu=1), [device_index])[0]
        prev = bool(data["prev_exists"].detach().cpu().flatten()[-1])
        with torch.no_grad():
            model(return_loss=True, **data)
        decision = model._object_evidence_last_decisions[0]
        student = StreamPETRAdapter(model).snapshot_memory()
        shadow = model._oep_clean_shadow_memory
        item = dict(
            frame=decision.frame_index, scene_token=decision.scene_token,
            sample_token=decision.sample_token, prev_exists=prev,
            sequence_id=decision.sequence_id,
            clean_or_fault="fault" if decision.fault_active else "clean",
            fault_camera=decision.fault_spec.camera,
            fault_type=decision.fault_spec.fault_type,
            severity=decision.fault_spec.severity,
            fault_active=decision.fault_active,
            student_memory_checksum=memory_checksum(student),
            clean_teacher_memory_checksum=memory_checksum(shadow),
            memory_max_abs_diff=max_abs_difference(student, shadow),
        )
        trace.append(item)

    for index in range(start, start + 6):
        execute(index)
    execute(next_start)
    sequence = trace[:6]
    specs = {(row["fault_camera"], row["fault_type"], row["severity"]) for row in sequence}
    if [row["fault_active"] for row in sequence] != [False, False, True, True, True, True]:
        raise RuntimeError("StreamPETR onset continuity check failed")
    if len(specs) != 1 or any(row["sequence_id"] != sequence[0]["sequence_id"] for row in sequence):
        raise RuntimeError("StreamPETR fault spec changed within a sequence")
    if sequence[0]["memory_max_abs_diff"] != 0 or sequence[1]["memory_max_abs_diff"] != 0:
        raise RuntimeError("clean shadow diverged before fault onset")
    if not any(row["memory_max_abs_diff"] > 0 for row in sequence[2:]):
        raise RuntimeError("teacher/student memories did not diverge after onset")
    if trace[-1]["frame"] != 0 or trace[-1]["sequence_id"] == sequence[0]["sequence_id"]:
        raise RuntimeError("StreamPETR sequence reset check failed")
    return dict(architecture="streampetr", trace=trace)


def official_bevdepth_values():
    path = BEVDEPTH / "bevdepth/exps/nuscenes/base_exp.py"
    tree = ast.parse(path.read_text())
    names = {
        "H", "W", "final_dim",
        "backbone_conf", "bev_backbone", "bev_neck", "CLASSES", "TASKS",
        "common_heads", "bbox_coder", "train_cfg", "test_cfg", "head_conf",
        "ida_aug_conf", "bda_aug_conf", "img_conf",
    }
    body = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id in names for target in targets):
            body.append(node)
    values = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), values)
    values["head_conf"]["bev_backbone_conf"]["in_channels"] = 160
    values["head_conf"]["bev_neck_conf"]["in_channels"] = [160, 160, 320, 640]
    values["head_conf"]["train_cfg"]["code_weights"] = [1.0] * 10
    return values


def bevdepth_smoke_info():
    path = Path(tempfile.gettempdir()) / "evidence3d_bevdepth_train_scene.pkl"
    if path.is_file():
        return path
    sys.path.insert(0, str(BEVDEPTH))
    sys.path.insert(0, str(BEVDEPTH / "scripts"))
    from gen_info import generate_info
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits

    nusc = NuScenes(version="v1.0-trainval", dataroot=str(ROOT / "data/nuscenes"), verbose=False)
    infos = generate_info(nusc, [splits.train[0]])
    mmcv.dump(infos, str(path))
    return path


class BEVDepthSmokeOwner(nn.Module):
    def __init__(self, model, values, mode):
        super().__init__()
        self.model = model
        self.data_root = str(ROOT / "data/nuscenes")
        self.ida_aug_conf = values["ida_aug_conf"]
        self.img_conf = values["img_conf"]
        self.downsample_factor = values["backbone_conf"]["downsample_factor"]
        self.dbound = values["backbone_conf"]["d_bound"]
        self.depth_channels = int((self.dbound[1] - self.dbound[0]) / self.dbound[2])
        from models.object_evidence.integrations.bevdepth_lightning import initialize_native_state
        initialize_native_state(self, dict(
            enabled=True, seed=2026, pair_probability=0.5,
            lambda_oe=0.0 if mode == "r0" else 0.5,
            lambda_pg=0.0, auxiliary_warmup_iters=1000,
        ))
        if mode == "oe":
            self._oep_iteration = self.object_evidence.warmup_iters


def bevdepth_smoke(args):
    sys.path.insert(0, str(BEVDEPTH))
    from models.object_evidence.integrations.bevdepth_lightning import (
        install_voxel_pooling_fallback, native_paired_step,
    )
    install_voxel_pooling_fallback()
    from bevdepth.datasets.nusc_det_dataset import collate_fn
    from bevdepth.models.base_bev_depth import BaseBEVDepth
    from datasets.bevdepth_object_evidence import OEPairedNuscDetDataset
    from scripts.export_object_evidence_detector_only import detector_only_state_dict

    values = official_bevdepth_values()
    dataset = OEPairedNuscDetDataset(
        ida_aug_conf=values["ida_aug_conf"], bda_aug_conf=values["bda_aug_conf"],
        classes=values["CLASSES"], data_root=str(ROOT / "data/nuscenes"),
        info_paths=str(bevdepth_smoke_info()), is_train=True, use_cbgs=False,
        img_conf=values["img_conf"], num_sweeps=1, sweep_idxes=[], key_idxes=[-1],
        return_depth=True, use_fusion=False,
    )
    index = int(0 if args.dataset_index is None else args.dataset_index) % len(dataset)
    seed_all()
    batch = collate_fn([dataset[index]], is_return_depth=True)
    device = torch.device(args.device)
    batch = tuple(
        {key: value.to(device) for key, value in item.items()} if isinstance(item, dict)
        else [value.to(device) if torch.is_tensor(value) else value for value in item]
        if isinstance(item, list) else item.to(device) if torch.is_tensor(item) else item
        for item in batch
    )
    model = BaseBEVDepth(values["backbone_conf"], values["head_conf"], is_train_depth=True)
    checkpoint = torch.load(
        ROOT / "checkpoints/official/bev_depth_lss_r50_256x704_128x128_24e_2key.pth",
        map_location="cpu",
    )["state_dict"]
    state = detector_only_state_dict(checkpoint, detector_prefixes=("model.",))
    model.load_state_dict(state, strict=True)
    owner = BEVDepthSmokeOwner(model, values, args.mode).to(device).train()
    optimizer = torch.optim.AdamW(owner.model.parameters(), lr=2e-4 / 64, weight_decay=1e-7)
    torch.cuda.reset_peak_memory_stats()
    total = native_paired_step(owner, batch)
    diagnostics = owner._object_evidence_last
    gradient, delta = gradient_and_step(owner.model, total, optimizer)
    episode = diagnostics["episode"]
    info = dataset.infos[index]
    return dict(
        architecture="bevdepth", mode=args.mode,
        sample_token=str(info["sample_token"]), scene_token=str(info["scene_token"]),
        fault_camera=episode.spec.camera, fault_type=episode.spec.fault_type,
        severity=episode.spec.severity,
        original_detection_loss=float(diagnostics["detection_loss"]),
        depth_loss=float(diagnostics["depth_loss"]), oe_loss=float(diagnostics["oe_loss"]),
        total_loss=float(diagnostics["total_loss"]),
        observability_gap_mean=float(diagnostics["gap_mean"]),
        observability_gap_max=float(diagnostics["gap_max"]),
        matched_object_count=int(diagnostics["matched_objects"]),
        detector_gradient_norm=gradient, detector_parameter_delta=delta,
        peak_cuda_memory_mb=torch.cuda.max_memory_allocated() / 1024 ** 2,
    )


def bevdepth_temporal_continuity(args):
    sys.path.insert(0, str(BEVDEPTH))
    from datasets.bevdepth_object_evidence import OEPairedNuscDetDataset
    from models.object_evidence.integrations.fault_images import bevdepth_fault_images_from_raw
    from models.object_evidence.integrations.sequence_episode import deterministic_sample_faults

    values = official_bevdepth_values()
    dataset = OEPairedNuscDetDataset(
        ida_aug_conf=values["ida_aug_conf"], bda_aug_conf=values["bda_aug_conf"],
        classes=values["CLASSES"], data_root=str(ROOT / "data/nuscenes"),
        info_paths=str(bevdepth_smoke_info()), is_train=True, use_cbgs=False,
        img_conf=values["img_conf"], num_sweeps=1, sweep_idxes=[], key_idxes=[-1],
        return_depth=True, use_fusion=False,
    )
    index = int(0 if args.dataset_index is None else args.dataset_index) % len(dataset)
    seed_all()
    sample = dataset[index]
    clean, meta = sample[0].unsqueeze(0), [sample[7]]
    spec = deterministic_sample_faults(2026, [meta[0]["token"]], 0.5, force_fault=True)[0]
    fault = bevdepth_fault_images_from_raw(
        clean, meta, [spec], str(ROOT / "data/nuscenes"),
        values["img_conf"]["img_mean"], values["img_conf"]["img_std"],
        values["ida_aug_conf"]["cams"], values["img_conf"]["to_rgb"],
    )
    rows = []
    for storage_index, role in enumerate(("current", "previous")):
        fault_delta = float((fault[0, storage_index, spec.camera] - clean[0, storage_index, spec.camera]).abs().max())
        healthy = [camera for camera in range(clean.shape[2]) if camera != spec.camera]
        healthy_delta = float((fault[0, storage_index, healthy] - clean[0, storage_index, healthy]).abs().max())
        rows.append(dict(
            storage_index=storage_index, role=role, fault_camera=spec.camera,
            fault_type=spec.fault_type, severity=spec.severity,
            fault_active=True, fault_camera_pixel_delta=fault_delta,
            healthy_camera_pixel_delta=healthy_delta,
        ))
    if rows[0]["fault_camera_pixel_delta"] <= 0 or rows[1]["fault_camera_pixel_delta"] <= 0:
        raise RuntimeError("BEVDepth fault did not affect both key frames")
    if any(row["healthy_camera_pixel_delta"] != 0 for row in rows):
        raise RuntimeError(f"BEVDepth persistent fault changed a healthy camera: {rows}")
    return dict(
        architecture="bevdepth", sample_token=meta[0]["token"],
        scene_token=meta[0]["scene_token"], trace=rows,
    )


def bevdepth_disabled_equivalence(args):
    # The training environment lacks the optional Lightning package.  A minimal
    # no-op shell is sufficient to execute the official LightningModule's real
    # training_step and the subclass's disabled dispatch on the same batch.
    lightning = types.ModuleType("pytorch_lightning")
    lightning_core = types.ModuleType("pytorch_lightning.core")

    class LightningModule(nn.Module):
        def save_hyperparameters(self, *unused_args, **unused_kwargs):
            return None

        def log(self, name, value, *unused_args, **unused_kwargs):
            self._equivalence_log_names.append(name)

    lightning.LightningModule = LightningModule
    lightning_core.LightningModule = LightningModule
    sys.modules["pytorch_lightning"] = lightning
    sys.modules["pytorch_lightning.core"] = lightning_core
    sys.path.insert(0, str(BEVDEPTH))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    from bevdepth.datasets.nusc_det_dataset import NuscDetDataset, collate_fn
    from bevdepth.exps.nuscenes.base_exp import BEVDepthLightningModel
    from models.object_evidence.integrations.bevdepth_lightning import (
        OEBevDepthLightningModel,
    )

    values = official_bevdepth_values()
    dataset = NuscDetDataset(
        ida_aug_conf=values["ida_aug_conf"], bda_aug_conf=values["bda_aug_conf"],
        classes=values["CLASSES"], data_root=str(ROOT / "data/nuscenes"),
        info_paths=str(bevdepth_smoke_info()), is_train=True, use_cbgs=False,
        img_conf=values["img_conf"], num_sweeps=1, sweep_idxes=[], key_idxes=[-1],
        return_depth=True, use_fusion=False,
    )
    index = int(0 if args.dataset_index is None else args.dataset_index) % len(dataset)
    seed_all()
    batch = collate_fn([dataset[index]], is_return_depth=True)
    owner = OEBevDepthLightningModel(
        object_evidence=dict(enabled=False), data_root=str(ROOT / "data/nuscenes"),
        batch_size_per_device=1, gpus=1,
    ).to(args.device).train()
    checkpoint = torch.load(
        ROOT / "checkpoints/official/bev_depth_lss_r50_256x704_128x128_24e_2key.pth",
        map_location="cpu",
    )["state_dict"]
    owner.load_state_dict(checkpoint, strict=True)
    initial_parameters = {key: value.detach().cpu().clone() for key, value in owner.state_dict().items()}

    def execute(use_wrapper):
        owner.load_state_dict(initial_parameters, strict=True)
        owner._equivalence_log_names = []
        seed_all()
        outputs = []
        handle = owner.model.register_forward_hook(
            lambda _module, _inputs, value: outputs.append(detached_tree(value))
        )
        try:
            with torch.no_grad():
                loss = (
                    owner.training_step(batch, 0) if use_wrapper
                    else BEVDepthLightningModel.training_step(owner, batch)
                )
        finally:
            handle.remove()
        return detached_tree(loss), outputs[-1], sorted(owner._equivalence_log_names)

    disabled_loss, disabled_output, disabled_keys = execute(True)
    vanilla_loss, vanilla_output, vanilla_keys = execute(False)
    if disabled_keys != vanilla_keys:
        raise RuntimeError("disabled BEVDepth loss keys differ from vanilla")
    return dict(
        architecture="bevdepth", sample_token=str(dataset.infos[index]["sample_token"]),
        scene_token=str(dataset.infos[index]["scene_token"]), loss_keys=disabled_keys,
        loss_max_abs_diff=max_abs_difference(disabled_loss, vanilla_loss),
        output_max_abs_diff=max_abs_difference(disabled_output, vanilla_output),
        memory_max_abs_diff=None,
    )


def validate(result):
    numeric = [
        result["original_detection_loss"], result["total_loss"],
        result["detector_gradient_norm"], result["detector_parameter_delta"],
    ]
    if result["depth_loss"] is not None:
        numeric.append(result["depth_loss"])
    if not all(np.isfinite(value) for value in numeric):
        raise RuntimeError("native smoke produced a non-finite result")
    if result["detector_gradient_norm"] <= 0 or result["detector_parameter_delta"] <= 0:
        raise RuntimeError("native detector did not update")
    if result["mode"] == "oe":
        if not np.isfinite(result["oe_loss"]) or result["matched_object_count"] <= 0:
            raise RuntimeError("OE did not receive matched objects")
        if result["observability_gap_max"] <= 0:
            raise RuntimeError("OE batch contains no positive observability gap")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("native batch smoke requires CUDA")
    if args.temporal_continuity:
        result = (
            stream_temporal_continuity(args) if args.architecture == "streampetr"
            else bevdepth_temporal_continuity(args)
        )
    elif args.disabled_equivalence:
        result = (
            stream_disabled_equivalence(args) if args.architecture == "streampetr"
            else bevdepth_disabled_equivalence(args)
        )
        if result["loss_max_abs_diff"] > 1e-6 or result["output_max_abs_diff"] > 1e-6:
            raise RuntimeError(f"disabled native path differs from vanilla: {result}")
        if result["memory_max_abs_diff"] not in (None, 0.0):
            raise RuntimeError("disabled StreamPETR memory differs from vanilla")
    else:
        result = stream_smoke(args) if args.architecture == "streampetr" else bevdepth_smoke(args)
        validate(result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
