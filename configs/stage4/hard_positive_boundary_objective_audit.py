"""No-update train-only hard-positive boundary activation audit."""

_base_ = "./feq_common.py"

data = dict(train=dict(pipeline={{_base_.fault_train_pipeline}}))

work_dir = "outputs/stage4/hard_positive_boundary_objective_audit/audit_only"
load_from = "outputs/stage3/observability_distillation/b0/iter_969.pth"
model = dict(pts_bbox_head=dict(
    type="HardPositiveBoundaryStreamPETRHead",
    enable_hard_positive_boundary=True,
    # Unit weight exposes the raw objective; no optimizer is constructed.
    hard_positive_boundary_weight=1.0,
    hard_positive_boundary_margin=0.10,
    hard_positive_geometry_threshold=2.0,
))
