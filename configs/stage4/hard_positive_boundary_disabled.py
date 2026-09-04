"""B0 inference with the hard-positive objective explicitly disabled."""

_base_ = "../stage3/mini_convergence_b0.py"

model = dict(pts_bbox_head=dict(
    type="HardPositiveBoundaryStreamPETRHead",
    enable_hard_positive_boundary=False,
    hard_positive_boundary_weight=0.0,
))

