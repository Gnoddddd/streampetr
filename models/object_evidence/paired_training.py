"""Architecture-neutral paired objective and exact disabled-path wrapper."""

from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Sequence

import torch
from torch import Tensor, nn

from .losses import object_evidence_loss, privileged_geometry_loss
from .privileged_geometry.projector import PrivilegedGeometryProjector


@contextmanager
def temporary_eval_no_grad(module: nn.Module):
    """Disable gradients and running-stat updates, then restore every train flag."""
    states = {child: child.training for child in module.modules()}
    module.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for child, training in states.items():
            child.training = training


@contextmanager
def temporary_native_teacher(
    module: nn.Module, train_shaped_modules: Sequence[nn.Module] = ()
):
    """Eval/no-grad teacher context which preserves train-shaped detector dispatch.

    Some native detectors use their root ``training`` flag to reshape temporal
    inputs. The root therefore stays train-shaped while every child executes in
    eval mode. Forward pre-hooks defend against native code calling ``train()``
    internally between history frames.
    """
    states = {child: child.training for child in module.modules()}
    handles = []
    train_shaped = {module, *train_shaped_modules}
    module.eval()
    for current in train_shaped:
        current.training = True
    for child in list(module.modules())[1:]:
        if child in train_shaped:
            continue
        handles.append(child.register_forward_pre_hook(
            lambda current, _arguments: setattr(current, "training", False)
        ))
    try:
        with torch.no_grad():
            yield
    finally:
        for handle in handles:
            handle.remove()
        for child, training in states.items():
            child.training = training


def auxiliary_scale(global_iter: int, warmup_iters: int = 1000) -> float:
    if global_iter < 0 or warmup_iters <= 0:
        raise ValueError("global_iter must be non-negative and warmup_iters positive")
    return min(1.0, float(global_iter) / float(warmup_iters))


class ObjectEvidenceObjective(nn.Module):
    """OE-PG V1 core over already matched per-object tokens."""

    def __init__(
        self,
        teacher_dim: Optional[int] = None,
        lambda_oe: float = 0.5,
        lambda_pg: float = 0.25,
        warmup_iters: int = 1000,
    ):
        super().__init__()
        self.lambda_oe = float(lambda_oe)
        self.lambda_pg = float(lambda_pg)
        self.warmup_iters = int(warmup_iters)
        self.pg_projector = (
            PrivilegedGeometryProjector(int(teacher_dim)) if teacher_dim is not None else None
        )

    def forward(
        self,
        clean_object_tokens: Tensor,
        fault_object_tokens: Tensor,
        observability_gap: Tensor,
        valid_mask: Tensor,
        global_iter: int,
        lidar_teacher_tokens: Optional[Tensor] = None,
        num_lidar_pts: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        clean_object_tokens = clean_object_tokens.detach()
        loss_oe = object_evidence_loss(
            fault_object_tokens, clean_object_tokens, observability_gap, valid_mask
        )
        loss_pg = fault_object_tokens.sum() * 0
        if lidar_teacher_tokens is not None:
            if self.pg_projector is None or num_lidar_pts is None:
                raise ValueError("PG requires a configured projector and num_lidar_pts")
            loss_pg = privileged_geometry_loss(
                self.pg_projector(fault_object_tokens), lidar_teacher_tokens.detach(),
                observability_gap, valid_mask, num_lidar_pts,
            )
        scale = auxiliary_scale(global_iter, self.warmup_iters)
        auxiliary = scale * (self.lambda_oe * loss_oe + self.lambda_pg * loss_pg)
        return {
            "loss_object_evidence": loss_oe,
            "loss_privileged_geometry": loss_pg,
            "auxiliary_scale": fault_object_tokens.new_tensor(scale),
            "loss_object_evidence_total": auxiliary,
        }


class ObjectEvidenceTrainingWrapper(nn.Module):
    """Thin integration boundary; eval and disabled mode call only the detector."""

    def __init__(
        self,
        detector: nn.Module,
        enabled: bool = True,
        paired_runner: Optional[Callable[..., Any]] = None,
        objective: Optional[ObjectEvidenceObjective] = None,
    ):
        super().__init__()
        self.detector = detector
        self.enabled = bool(enabled)
        self.paired_runner = paired_runner
        self.object_evidence = objective

    def forward(self, *args, **kwargs):
        if not self.enabled or not self.training:
            return self.detector(*args, **kwargs)
        if self.paired_runner is None:
            raise RuntimeError("enabled training requires a paired_runner")
        return self.paired_runner(self, *args, **kwargs)
