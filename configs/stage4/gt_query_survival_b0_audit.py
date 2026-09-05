"""Frozen B0 with an opt-in, read-only GT-query survival trace."""

_base_ = "../stage3/mini_convergence_b0.py"

custom_imports = dict(
    imports=[
        "evidence3d_plugin",
        "hooks.evidence_trace_hook",
        "hooks.loss_balance_hook",
        "analysis.gt_query_survival_trace",
    ],
    allow_failed_imports=False,
)

