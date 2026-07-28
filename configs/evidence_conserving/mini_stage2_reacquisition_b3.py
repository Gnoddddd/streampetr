"""S2.3 rescue B3: larger, later restoration budget."""

_base_ = "./mini_stage2_reacquisition_base.py"
work_dir = "outputs/stage2/s2_3_rescue/zero_shot/b3"
model = dict(pts_bbox_head=dict(innovation_cfg=dict(
    restore_ratio=0.75,
    max_relative_bonus=0.10,
    max_absolute_bonus=0.05817988,
    minimum_gap_age=3,
)))
