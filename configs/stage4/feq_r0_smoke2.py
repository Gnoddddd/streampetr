_base_ = "./feq_r0.py"
work_dir = "outputs/stage4/feq_core/smoke_r0"
runner = dict(type="IterBasedRunner", max_iters=2)
checkpoint_config = dict(interval=2, by_epoch=False, max_keep_ckpts=1)
custom_hooks = []
