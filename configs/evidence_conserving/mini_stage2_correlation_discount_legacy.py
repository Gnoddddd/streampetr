"""S2.4 isolation audit: reproduce the historical implicit N_eff path."""

_base_ = "./mini_stage2_source_ledger_debug_50.py"

work_dir = "outputs/stage2/s2_4_isolation_audit/legacy"

model = dict(
    pts_bbox_head=dict(
        enable_correlation_discount=True,
    ),
)
