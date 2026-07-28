"""S2.3 N6: N5 plus optional strength saturation."""

_base_ = "./mini_stage2_innovation_n5.py"
work_dir = "outputs/stage2/s2_3_innovation/zero_shot_active/fixed_v3_s2_3_n6"
model = dict(pts_bbox_head=dict(innovation_cfg=dict(
    enable_strength_saturation=True,
)))
