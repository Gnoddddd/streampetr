"""Prediction-consistency control differing only in source tracking."""

_base_ = "./mini_stage2_source_ledger_smoke.py"

model = dict(
    pts_bbox_head=dict(
        enable_source_ledger=False,
    ),
)
