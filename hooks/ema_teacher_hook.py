"""EMA teacher lifecycle kept outside the deployable module tree."""

from __future__ import annotations

import copy

import torch
from mmcv.runner import HOOKS, Hook


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


@HOOKS.register_module()
class TrainingOnlyEMATeacherHook(Hook):
    def __init__(self, momentum: float = 0.999):
        self.momentum = float(momentum)
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("EMA momentum must be in [0, 1)")

    def before_run(self, runner) -> None:
        student = _unwrap(runner.model)
        if not getattr(student, "enable_observability_distillation", False):
            raise RuntimeError("EMA hook requires enabled distillation detector")
        teacher = copy.deepcopy(student)
        teacher.set_ema_teacher(None)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        student.set_ema_teacher(teacher)
        if any(parameter.requires_grad for parameter in teacher.parameters()):
            raise RuntimeError("Teacher parameters must never require gradients")

    @torch.no_grad()
    def after_train_iter(self, runner) -> None:
        student = _unwrap(runner.model)
        teacher = student._ema_teacher
        student_parameters = dict(student.named_parameters())
        for name, teacher_parameter in teacher.named_parameters():
            source = student_parameters[name].detach()
            teacher_parameter.mul_(self.momentum).add_(
                source, alpha=1.0 - self.momentum
            )
        student_buffers = dict(student.named_buffers())
        for name, teacher_buffer in teacher.named_buffers():
            if name in student_buffers:
                teacher_buffer.copy_(student_buffers[name])
        teacher.eval()
        runner.log_buffer.update(
            {
                "teacher_has_grad": float(
                    any(parameter.grad is not None for parameter in teacher.parameters())
                )
            },
            runner.outputs["num_samples"],
        )
