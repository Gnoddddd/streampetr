"""Shared fair-training contract for the S2.4 C0/C1 disambiguation."""

_base_ = "./mini_stage2_source_ledger_debug_50.py"

# Both candidates start before S2.1/S2.2 ledger adaptation.  Do not replace
# this with the S2.2 iter_50 checkpoint: that checkpoint was trained through
# the historical fixed-correlation path.
load_from = (
    "/home/research/research/evidence3d/outputs/final_snapshots/"
    "stage1_ternary_r50_200/checkpoint/iter_200.pth"
)
resume_from = None

seed = 2026
runner = dict(type="IterBasedRunner", max_iters=50)
checkpoint_config = dict(interval=50, by_epoch=False, max_keep_ckpts=1)
log_config = dict(
    interval=1,
    hooks=[dict(type="TextLoggerHook", by_epoch=False)],
)

# Explicitly preserve the accepted Stage2 precision and clipping settings.
fp16 = dict(loss_scale="dynamic")
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
