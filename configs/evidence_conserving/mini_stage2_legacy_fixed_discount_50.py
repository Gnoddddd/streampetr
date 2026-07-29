"""C1: historical fixed-correlation-discount 50-iteration run."""

_base_ = "./mini_stage2_correlation_disambiguation_common_50.py"

work_dir = "outputs/stage2/s2_4_baseline_disambiguation/c1_50"

model = dict(
    pts_bbox_head=dict(
        enable_correlation_discount=True,
    ),
)
