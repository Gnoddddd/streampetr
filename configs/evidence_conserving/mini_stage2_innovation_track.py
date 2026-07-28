"""S2.3 full diagnostics in track-only mode; S2.2 behavior is unchanged."""

_base_ = "./mini_stage2_source_ledger_debug_50.py"

work_dir = (
    "outputs/stage2/s2_3_innovation/track/"
    "fixed_v3_s2_3_full_track"
)
load_from = (
    "/home/research/research/evidence3d/outputs/stage2/"
    "s2_2_source_ledger_debug_50/iter_50.pth"
)
resume_from = None

model = dict(
    pts_bbox_head=dict(
        innovation_cfg=dict(
            mode="track",
            source_weight=0.25,
            feature_weight=0.25,
            geometry_weight=0.20,
            semantic_weight=0.15,
            enable_source=True,
            enable_feature=True,
            enable_geometry=True,
            enable_semantic=True,
            enable_reliability=True,
            enable_conflict=True,
            enable_asymmetric_negative=True,
            novelty_floor=0.30,
            tau_reacquisition=3.0,
            conflict_power=1.0,
            reliable_observation_threshold=0.05,
            negative_observability_threshold=0.20,
            negative_source_quality_threshold=0.20,
            enable_strength_saturation=False,
            strength_temperature=10.0,
        ),
        innovation_warmup_iters=10,
        innovation_transition_iters=20,
    )
)
