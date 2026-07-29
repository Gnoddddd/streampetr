_base_ = "./mini_raydn_b0_raydn_50.py"

work_dir = "outputs/stage3/raydn_screening/b0_raydn_smoke2"
runner = dict(type="IterBasedRunner", max_iters=2)
checkpoint_config = dict(interval=2, by_epoch=False, max_keep_ckpts=1)

