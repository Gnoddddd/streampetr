"""S2.3 N5: N4 plus conflict and asymmetric negative evidence."""

_base_ = "./mini_stage2_innovation_n4.py"
work_dir = "outputs/stage2/s2_3_innovation/zero_shot_active/fixed_v3_s2_3_n5"
model = dict(pts_bbox_head=dict(innovation_cfg=dict(
    enable_conflict=True,
    enable_asymmetric_negative=True,
)))
