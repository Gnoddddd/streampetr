"""C1 engineering smoke with the fair-training initialization."""

_base_ = "./mini_stage2_legacy_fixed_discount_50.py"

work_dir = "outputs/stage2/s2_4_baseline_disambiguation/c1_smoke2"
runner = dict(type="IterBasedRunner", max_iters=2)
checkpoint_config = dict(interval=2, by_epoch=False, max_keep_ckpts=1)
