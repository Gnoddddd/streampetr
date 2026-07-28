_base_ = [
    "./mini_debug_ternary_ft400_fp32.py"
]

work_dir = (
    "outputs/"
    "exp_007_ternary_smoke10_fp32"
)

runner = dict(
    max_iters=10,
)

checkpoint_config = dict(
    interval=10,
    max_keep_ckpts=1,
)

log_config = dict(
    interval=1,
)

evaluation = dict(
    interval=100000,
)
