"""S2.3 N3: source, feature, aligned geometry, and reacquisition."""

_base_ = "./mini_stage2_innovation_n2.py"
work_dir = "outputs/stage2/s2_3_innovation/zero_shot_active/fixed_v3_s2_3_n3"
model = dict(pts_bbox_head=dict(innovation_cfg=dict(enable_geometry=True)))
