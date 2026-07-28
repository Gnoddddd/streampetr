"""Shared S2.3 evidence-budgeted reacquisition configuration."""

_base_ = "./mini_stage2_reacquisition_b0.py"

model = dict(
    pts_bbox_head=dict(
        reacquisition_warmup_iters=10,
        innovation_cfg=dict(
            mode="active",
            innovation_active_strategy="budgeted_reacquisition",
            enable_source=True,
            enable_feature=True,
            enable_geometry=True,
            enable_semantic=True,
            enable_reliability=True,
            enable_conflict=True,
            enable_asymmetric_negative=True,
            enable_strength_saturation=False,
            reliable_observation_threshold=0.05,
            restore_ratio=0.50,
            max_relative_bonus=0.08,
            max_absolute_bonus=0.04654390,
            minimum_gap_age=2,
            reacquisition_time_tau=3.0,
            motion_sigma=5.0,
            use_motion_gate=True,
            use_source_recovery_gate=True,
        ),
    ),
)
