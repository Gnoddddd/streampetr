#!/usr/bin/env python3
"""Dependency-light CPU engineering smoke for OE-PG V1."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.object_evidence.paired_training import (  # noqa: E402
    ObjectEvidenceObjective, ObjectEvidenceTrainingWrapper,
)
from scripts.export_object_evidence_detector_only import detector_only_state_dict  # noqa: E402


def main():
    torch.manual_seed(2026)
    clean = torch.randn(4, 256, requires_grad=True)
    fault_detector = nn.Linear(12, 256)
    fault = fault_detector(torch.randn(4, 12))
    fault.retain_grad()
    teacher = torch.randn(4, 32, requires_grad=True)
    objective = ObjectEvidenceObjective(teacher_dim=32)
    losses = objective(
        clean, fault, torch.tensor([1.0, 0.6, 0.3, 0.0]),
        torch.ones(4, dtype=torch.bool), 1000, teacher, torch.tensor([20, 10, 5, 0]),
    )
    losses["loss_object_evidence_total"].backward()

    detector = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2)).eval()
    wrapped = ObjectEvidenceTrainingWrapper(detector, objective=objective).eval()
    inputs = torch.randn(2, 3)
    before = wrapped(inputs)
    exported = detector_only_state_dict(wrapped.state_dict())
    vanilla = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2)).eval()
    vanilla.load_state_dict(exported, strict=True)
    after = vanilla(inputs)
    result = {
        "oe_finite": bool(torch.isfinite(losses["loss_object_evidence"])),
        "pg_finite": bool(torch.isfinite(losses["loss_privileged_geometry"])),
        "clean_grad_is_none": clean.grad is None,
        "fault_grad_nonzero": bool(fault.grad is not None and fault.grad.abs().sum() > 0),
        "fault_detector_grad_nonzero": bool(
            fault_detector.weight.grad is not None and fault_detector.weight.grad.abs().sum() > 0
        ),
        "teacher_grad_is_none": teacher.grad is None,
        "export_strict": True,
        "max_abs_diff": float((before - after).abs().max()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not all(value is True or value == 0.0 for value in result.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
