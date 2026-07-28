_base_ = [
    "./mini_debug_source_aware_ft50_fp32.py"
]

work_dir = (
    "outputs/"
    "exp_006_source_aware_ft400_fp32"
)

runner = dict(
    max_iters=400,
)

checkpoint_config = dict(
    interval=100,
    max_keep_ckpts=4,
)

log_config = dict(
    interval=20,
)

# Keep the stability test in FP32.
fp16 = None

optimizer_config = dict(
    _delete_=True,
    type="OptimizerHook",
    grad_clip=dict(
        max_norm=35,
        norm_type=2,
    ),
)

# Evaluate separately after training.
evaluation = dict(
    interval=100000,
)
