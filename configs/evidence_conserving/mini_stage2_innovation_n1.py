"""S2.3 N1: source innovation plus temporal reacquisition."""

_base_ = "./mini_stage2_innovation_track.py"
work_dir = "outputs/stage2/s2_3_innovation/zero_shot_active/fixed_v3_s2_3_n1"
model = dict(pts_bbox_head=dict(innovation_cfg=dict(
    mode="active",
    enable_source=True,
    enable_feature=False,
    enable_geometry=False,
    enable_semantic=False,
    enable_reliability=False,
    enable_conflict=False,
    enable_asymmetric_negative=False,
    enable_strength_saturation=False,
)))
