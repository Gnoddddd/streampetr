_base_ = ['/home/research/research/evidence3d/configs/evidence_conserving/mini_debug_source_aware_400.py']

# Load model parameters only.
load_from = '/home/research/research/evidence3d/outputs/exp_003_evidence_conserving/iter_4000.pth'
resume_from = None

work_dir = '/home/research/research/evidence3d/outputs/exp_005_source_aware_ft50_fp32'

runner = dict(
    max_iters=50,
)

checkpoint_config = dict(
    interval=50,
    max_keep_ckpts=1,
)

log_config = dict(
    interval=10,
)

# Lower learning rate for short fine-tuning.
optimizer = dict(
    lr=2.5e-06,
)

lr_config = dict(
    _delete_=True,
    policy="fixed",
    warmup=None,
)

# Disable mixed precision.
fp16 = None

# Completely replace Fp16OptimizerHook.
optimizer_config = dict(
    _delete_=True,
    type="OptimizerHook",
    grad_clip=dict(
        max_norm=35,
        norm_type=2,
    ),
)

# Disable intermediate evaluation.
evaluation = dict(
    interval=100000,
)
