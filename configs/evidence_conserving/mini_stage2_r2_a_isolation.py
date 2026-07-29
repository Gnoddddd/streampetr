"""R2-A: isolate every unconfirmed reacquisition from formal memory."""

_base_ = "./mini_stage2_r2_base.py"

work_dir = "outputs/stage2/s2_3_r2_formal/zero_shot/r2_a"
model = dict(pts_bbox_head=dict(
    enable_two_phase_reacquisition=False,
))
