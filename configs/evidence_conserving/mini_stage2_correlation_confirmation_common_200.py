"""Frozen common contract for the C0/C1 200-iteration confirmation."""

_base_ = "./mini_stage2_correlation_disambiguation_common_50.py"

seed = 2026
runner = dict(type="IterBasedRunner", max_iters=200)
checkpoint_config = dict(interval=200, by_epoch=False, max_keep_ckpts=1)
log_config = dict(
    interval=1,
    hooks=[dict(type="TextLoggerHook", by_epoch=False)],
)

# Reaffirm the frozen precision and clipping contract.
fp16 = dict(loss_scale="dynamic")
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
