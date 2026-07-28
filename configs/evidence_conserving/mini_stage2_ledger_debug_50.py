"""Real 50-iteration S2.1 integration run initialized from frozen T1."""

_base_ = "./mini_ternary_official_r50_stage1_200_fp32.py"

work_dir = "outputs/stage2/s2_1_ledger_debug_50"
load_from = (
    "/home/research/research/evidence3d/outputs/final_snapshots/"
    "stage1_ternary_r50_200/checkpoint/iter_200.pth"
)
resume_from = None

runner = dict(type="IterBasedRunner", max_iters=50)
checkpoint_config = dict(interval=50, by_epoch=False, max_keep_ckpts=1)
log_config = dict(
    interval=1,
    hooks=[dict(type="TextLoggerHook", by_epoch=False)],
)

# Preserve mixed precision and the Stage1 max-norm gradient clipping.
fp16 = dict(loss_scale="dynamic")
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

model = dict(
    pts_bbox_head=dict(
        evidence_probability_source="ternary",
        enable_evidence_memory=True,
        trace_enabled=True,
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

custom_hooks = [
    dict(
        type="EvidenceTraceHook",
        interval=1,
        out_file="stage2_ledger_train_trace.jsonl",
    ),
    dict(
        type="FreezeExceptHook",
        trainable_patterns=[
            "ternary_branches",
            "observability_head",
            "evidence_step",
        ],
        priority="VERY_HIGH",
    ),
]
