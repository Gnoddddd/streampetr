"""Hooks for the mini convergence/loss-balance experiment.

The ramp hook changes only the scalar returned by the existing ternary loss
module.  It does not add model state or alter the detector forward graph.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import torch
from mmcv.runner import Fp16OptimizerHook, HOOKS, Hook


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


@HOOKS.register_module()
class AuxiliaryLossRampHook(Hook):
    """Ramp the existing Evidence3D auxiliary loss over training progress."""

    def __init__(
        self,
        zero_until: float = 0.2,
        full_at: float = 0.5,
    ) -> None:
        if not 0.0 <= zero_until < full_at <= 1.0:
            raise ValueError("Expected 0 <= zero_until < full_at <= 1")
        self.zero_until = float(zero_until)
        self.full_at = float(full_at)
        self._scale = 0.0
        self._handle = None

    def before_run(self, runner) -> None:
        head = getattr(_unwrap(runner.model), "pts_bbox_head", None)
        loss_module = getattr(head, "ternary_loss", None)
        if loss_module is None:
            raise RuntimeError(
                "M1-Ramp requires the existing ternary_loss module; "
                "no forward or memory fallback is permitted"
            )

        def scale_output(_module, _inputs, output):
            return output * self._scale

        self._handle = loss_module.register_forward_hook(scale_output)

    def before_train_iter(self, runner) -> None:
        progress = float(runner.iter) / max(float(runner.max_iters), 1.0)
        if progress <= self.zero_until:
            scale = 0.0
        elif progress >= self.full_at:
            scale = 1.0
        else:
            scale = (progress - self.zero_until) / (
                self.full_at - self.zero_until
            )
        self._scale = float(scale)
        outputs = getattr(runner, "outputs", None)
        runner.log_buffer.update(
            {"auxiliary_loss_scale": self._scale},
            outputs.get("num_samples", 1) if outputs else 1,
        )

    def after_run(self, runner) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def _grouped_grad_norms(model) -> Dict[str, float]:
    """Return mutually exclusive pre-clipping gradient L2 norms."""

    squared = {"backbone": 0.0, "head": 0.0, "evidence": 0.0}
    evidence_markers = (
        "pts_bbox_head.ternary_branches",
        "pts_bbox_head.observability_head",
        "pts_bbox_head.evidence_ledger",
    )
    for name, parameter in _unwrap(model).named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if name.startswith("img_backbone."):
            group = "backbone"
        elif any(marker in name for marker in evidence_markers):
            group = "evidence"
        else:
            group = "head"
        norm = parameter.grad.detach().float().norm(2).item()
        squared[group] += norm * norm
    return {key: math.sqrt(value) for key, value in squared.items()}


@HOOKS.register_module()
class GroupedFp16OptimizerHook(Fp16OptimizerHook):
    """FP16 optimizer hook that logs pre-clipping parameter-group norms."""

    def after_train_iter(self, runner) -> None:
        runner.model.zero_grad()
        runner.optimizer.zero_grad()
        self.loss_scaler.scale(runner.outputs["loss"]).backward()
        self.loss_scaler.unscale_(runner.optimizer)

        grouped = _grouped_grad_norms(runner.model)
        runner.log_buffer.update(
            {f"grad_norm_{key}": value for key, value in grouped.items()},
            runner.outputs["num_samples"],
        )

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                runner.log_buffer.update(
                    {"grad_norm": float(grad_norm)},
                    runner.outputs["num_samples"],
                )
        self.loss_scaler.step(runner.optimizer)
        self.loss_scaler.update(self._scale_update_param)
        runner.meta.setdefault("fp16", {})[
            "loss_scaler"
        ] = self.loss_scaler.state_dict()


@HOOKS.register_module()
class MilestoneCheckpointHook(Hook):
    """Save explicit iteration milestones without retaining every epoch."""

    def __init__(self, milestones: Iterable[int]) -> None:
        self.milestones: Tuple[int, ...] = tuple(
            sorted({int(value) for value in milestones if int(value) > 0})
        )

    def after_train_iter(self, runner) -> None:
        iteration = int(runner.iter + 1)
        if iteration not in self.milestones:
            return
        runner.save_checkpoint(
            runner.work_dir,
            filename_tmpl="iter_{}.pth",
            save_optimizer=True,
            create_symlink=False,
        )
