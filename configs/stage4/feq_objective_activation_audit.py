"""Train-only, no-update FEQ objective activation audit."""

_base_ = "./feq_r0.py"

work_dir = "outputs/stage4/feq_query_objective_activation/audit_only"
model = dict(pts_bbox_head=dict(
    type="FEQStreamPETRHead",
    enable_feq_core=True,
    # Unit weights expose raw loss/gradients; no optimizer is constructed.
    feq_otm_weight=1.0,
    feq_boundary_weight=1.0,
    feq_max_aux=3,
    feq_boundary_margin=0.10,
))
