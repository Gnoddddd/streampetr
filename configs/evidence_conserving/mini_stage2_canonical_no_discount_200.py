"""C0: canonical no-discount 200-iteration confirmation."""

_base_ = "./mini_stage2_correlation_confirmation_common_200.py"

work_dir = "outputs/stage2/s2_4_baseline_confirmation/c0_200"

model = dict(
    pts_bbox_head=dict(
        enable_correlation_discount=False,
    ),
)
