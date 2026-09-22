"""CPU-only synthetic smoke for GeoCorr Stage 2 correspondence mechanics."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.geocorr_recovery.stage1_infrastructure import candidate_offsets  # noqa: E402
from models.geocorr_recovery import (  # noqa: E402
    ObjectCentricCorrelationField,
    correspondence_distillation,
)


def main() -> None:
    torch.manual_seed(2026)
    batch, objects, candidates, time, views, channels = 1, 4, 9, 1, 6, 256
    clean = torch.randn(batch, objects, candidates, views, channels)
    dirty = clean + 0.2 * torch.randn_like(clean)
    history = torch.randn(batch, objects, candidates, time, views, channels)
    current_valid = torch.ones(batch, objects, candidates, views, dtype=torch.bool)
    history_valid = torch.ones(batch, objects, candidates, time, views, dtype=torch.bool)
    module = ObjectCentricCorrelationField(candidate_offsets, feature_dim=channels)
    output = module(clean, dirty, history, current_valid, history_valid)
    loss, diagnostics = correspondence_distillation(
        output["teacher_logits"], output["student_logits"], output["valid_mask"],
        temperature=0.1, center_candidate_index=module.center_candidate_index,
    )
    loss.backward()
    gradient_finite = all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.adapter.parameters()
    )
    print("teacher_logits", tuple(output["teacher_logits"].shape))
    print("student_logits", tuple(output["student_logits"].shape))
    for key in (
        "num_valid_queries", "teacher_entropy", "student_entropy",
        "teacher_center_candidate_mass", "student_center_candidate_mass",
        "teacher_student_kl", "top1_agreement",
    ):
        print(key, diagnostics[key].item())
    print("adapter_parameter_count", sum(p.numel() for p in module.adapter.parameters()))
    print("gradient_finite", gradient_finite)


if __name__ == "__main__":
    main()
