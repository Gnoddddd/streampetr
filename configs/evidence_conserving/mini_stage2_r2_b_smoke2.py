"""Two-iteration engineering smoke for R2-B."""

_base_ = "./mini_stage2_r2_b_debug50.py"

work_dir = "outputs/stage2/s2_3_r2_formal/smoke/r2_b"
runner = dict(type="IterBasedRunner", max_iters=2)
checkpoint_config = dict(interval=2, by_epoch=False, max_keep_ckpts=1)
