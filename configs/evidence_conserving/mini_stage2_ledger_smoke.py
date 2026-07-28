"""S2.1 ledger-only smoke configuration.

This inherits the frozen Stage1 active evaluation setup and changes only the
conservation switch. It does not enable provenance, correlation, or new hard
gating behavior.
"""

_base_ = "./mini_ternary_official_r50_stage1_active_eval.py"

model = dict(
    pts_bbox_head=dict(
        temporal_update_cfg=dict(
            gamma=0.9,
            evidence_scale=2.0,
            max_effective_count=6.0,
            enable_conservation=True,
            reliable_observation_threshold=0.05,
            conservation_tolerance=1e-5,
        ),
    ),
)
