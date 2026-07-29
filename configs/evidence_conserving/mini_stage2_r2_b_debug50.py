"""Fair 50-iteration formal training configuration for R2-B."""

_base_ = "./mini_stage2_r2_a_debug50.py"

work_dir = "outputs/stage2/s2_3_r2_formal/debug_50/r2_b"
model = dict(pts_bbox_head=dict(
    enable_two_phase_reacquisition=True,
))
