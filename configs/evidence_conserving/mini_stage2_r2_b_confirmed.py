"""R2-B: two-frame, class/center/motion confirmed reacquisition."""

_base_ = "./mini_stage2_r2_base.py"

work_dir = "outputs/stage2/s2_3_r2_formal/zero_shot/r2_b"
model = dict(pts_bbox_head=dict(
    enable_two_phase_reacquisition=True,
))
