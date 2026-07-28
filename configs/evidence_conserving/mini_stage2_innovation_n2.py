"""S2.3 N2: source, aligned feature, and temporal reacquisition."""

_base_ = "./mini_stage2_innovation_n1.py"
work_dir = "outputs/stage2/s2_3_innovation/zero_shot_active/fixed_v3_s2_3_n2"
model = dict(pts_bbox_head=dict(innovation_cfg=dict(enable_feature=True)))
