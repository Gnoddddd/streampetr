# Shared method defaults. This file is documentation-friendly and can be
# imported by standalone experiments; the main MMDetection config keeps values
# explicit for reproducibility.

evidence_defaults = dict(
    observability_cfg=dict(
        min_depth=0.1,
        boundary_softness=8.0,
        depth_temperature=4.0,
        learned_residual=False,
        residual_weight=0.0,
    ),
    temporal_update_cfg=dict(
        gamma=0.90,
        evidence_scale=2.0,
        max_effective_count=6.0,
    ),
    policy_cfg=dict(
        keep_observability=0.45,
        keep_presence=0.55,
        keep_max_uncertainty=0.55,
        recover_presence=0.55,
        recover_max_uncertainty=0.80,
        recover_max_age=3,
        recover_min_prior_strength=0.5,
        strong_negative=0.60,
        recover_score_scale=0.75,
        defer_score_scale=0.20,
    ),
)
