"""Common 12-mini-epoch settings for the convergence experiment."""

_base_ = "../evidence_conserving/mini_stage2_source_ledger_debug_50.py"

# The generated mini annotations contain 323 training samples.  With one GPU
# and one temporal sample per GPU this is exactly 323 iterations per epoch.
mini_train_samples = 323
effective_batch_size = 1
iters_per_epoch = 323
mini_equivalent_epochs = 12
max_iters = 3876
checkpoint_milestones = (323, 969, 1938, 3876)

seed = 2026
work_dir = "outputs/stage3/mini_convergence_loss_balance/common"
load_from = (
    "/home/research/research/evidence3d/checkpoints/official/"
    "stream_petr_r50_flash_704_bs2_seq_90e.pth"
)
resume_from = None

runner = dict(type="IterBasedRunner", max_iters=max_iters)
evaluation = dict(interval=100000)
checkpoint_config = dict(
    interval=max_iters,
    by_epoch=False,
    max_keep_ckpts=1,
)
log_config = dict(
    interval=10,
    hooks=[dict(type="TextLoggerHook", by_epoch=False)],
)

# StreamPETR's wrapper force-constructs the stock hook whenever this top-level
# value is non-null.  Leave it null so the registered grouped hook below owns
# the complete FP16 path (model wrapping, dynamic scaling, unscale and step).
# Official StreamPETR DN is disabled in every group so DN loss keys are absent.
fp16 = None
optimizer_config = dict(
    type="GroupedFp16OptimizerHook",
    loss_scale="dynamic",
    grad_clip=dict(max_norm=35, norm_type=2),
)

model = dict(
    img_backbone=dict(pretrained="torchvision://resnet50"),
    pts_bbox_head=dict(with_dn=False),
)

custom_imports = dict(
    imports=[
        "evidence3d_plugin",
        "hooks.evidence_trace_hook",
        "hooks.loss_balance_hook",
    ],
    allow_failed_imports=False,
)
custom_hooks = [
    dict(type="EvidenceTraceHook", interval=10),
    dict(
        type="MilestoneCheckpointHook",
        milestones=checkpoint_milestones[:-1],
    ),
]
