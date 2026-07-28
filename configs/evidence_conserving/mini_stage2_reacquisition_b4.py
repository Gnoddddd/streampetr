"""S2.3 rescue B4: B2 motion-only gate ablation."""

_base_ = "./mini_stage2_reacquisition_base.py"
work_dir = "outputs/stage2/s2_3_rescue/zero_shot/b4"
model = dict(pts_bbox_head=dict(innovation_cfg=dict(
    use_motion_gate=True,
    use_source_recovery_gate=False,
)))
