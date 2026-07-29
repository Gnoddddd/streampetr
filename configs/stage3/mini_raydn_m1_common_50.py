"""Shared M1 screening setup: S2.2 with all detector parameters trainable."""

_base_ = "../evidence_conserving/mini_stage2_source_ledger_debug_50.py"

load_from = (
    "/home/research/research/evidence3d/outputs/stage2/"
    "s2_2_source_ledger_debug_50/iter_50.pth"
)
resume_from = None
seed = 2026
runner = dict(type="IterBasedRunner", max_iters=50)
checkpoint_config = dict(interval=50, by_epoch=False, max_keep_ckpts=1)
log_config = dict(
    interval=1,
    hooks=[dict(type="TextLoggerHook", by_epoch=False)],
)
fp16 = dict(loss_scale="dynamic")
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# Candidate-quality screening must update the detector.  The Stage2
# FreezeExceptHook is therefore intentionally absent in all four groups.
custom_hooks = [
    dict(
        type="EvidenceTraceHook",
        interval=1,
        out_file="evidence_trace.jsonl",
    ),
]

model = dict(
    pts_bbox_head=dict(
        enable_ray_denoising=False,
        raydn_group=1,
        raydn_num=5,
        raydn_alpha=8.0,
        raydn_beta=2.0,
        raydn_radius=3.0,
    ),
)

