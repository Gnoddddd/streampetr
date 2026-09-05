"""F1: the R0 episode stream plus the preregistered FEQ-Core losses."""
_base_ = "./feq_r0.py"
work_dir = "outputs/stage4/feq_core/f1"
model = dict(pts_bbox_head=dict(
    type="FEQStreamPETRHead",
    enable_feq_core=True,
    # Filled only after the frozen 20-train-batch calibration.
    feq_otm_weight=0.0,
    feq_boundary_weight=0.0,
    feq_max_aux=3,
    feq_boundary_margin=0.10,
))
