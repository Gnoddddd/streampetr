"""S2.3 rescue B1: conservative source-gated restoration."""

_base_ = "./mini_stage2_reacquisition_base.py"
work_dir = "outputs/stage2/s2_3_rescue/zero_shot/b1"
model = dict(pts_bbox_head=dict(innovation_cfg=dict(
    restore_ratio=0.25,
    max_relative_bonus=0.05,
    max_absolute_bonus=0.02908994,
    minimum_gap_age=2,
    use_motion_gate=False,
    use_source_recovery_gate=True,
)))
