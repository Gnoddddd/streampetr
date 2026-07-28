"""S2.3 rescue B6: five-percent trust region around legacy N1."""

_base_ = "./mini_stage2_innovation_n1.py"
work_dir = "outputs/stage2/s2_3_rescue/zero_shot/b6"
model = dict(pts_bbox_head=dict(
    reacquisition_warmup_iters=10,
    innovation_cfg=dict(
        innovation_active_strategy="residual_preserving",
        residual_preserving_mix=0.05,
    ),
))
