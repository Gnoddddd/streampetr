"""S2.3 rescue B5: B2 source-only gate ablation."""

_base_ = "./mini_stage2_reacquisition_base.py"
work_dir = "outputs/stage2/s2_3_rescue/zero_shot/b5"
model = dict(pts_bbox_head=dict(innovation_cfg=dict(
    use_motion_gate=False,
    use_source_recovery_gate=True,
)))
