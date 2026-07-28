"""S2.3 N4: all innovation components with reliability calibration."""

_base_ = "./mini_stage2_innovation_n3.py"
work_dir = "outputs/stage2/s2_3_innovation/zero_shot_active/fixed_v3_s2_3_n4"
model = dict(pts_bbox_head=dict(innovation_cfg=dict(
    enable_semantic=True,
    enable_reliability=True,
)))
