"""Two-step stock-head reference for disabled FEQ training invariance."""
_base_ = "../stage3/mini_convergence_b0.py"
work_dir = "outputs/stage4/feq_core/invariance_train_stock"
runner = dict(type="IterBasedRunner", max_iters=2)
checkpoint_config = dict(interval=2, by_epoch=False, max_keep_ckpts=1)
custom_hooks = []
