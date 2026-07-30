"""Engineering-only disabled-path config; parameter/inference identical to B0."""

_base_ = "./mini_observability_b0.py"

model = dict(
    type="ObservabilityDistillationPetr3D",
    enable_observability_distillation=False,
    pts_bbox_head=dict(
        type="ObservabilityDistillationStreamPETRHead",
        enable_observability_distillation=False,
    ),
)

