"""Frozen B0 inference with an optional read-only counterfactual trace."""

_base_ = "./mini_convergence_b0.py"

custom_imports = dict(
    imports=[
        "evidence3d_plugin",
        "hooks.evidence_trace_hook",
        "hooks.loss_balance_hook",
        "analysis.counterfactual_trace",
    ],
    allow_failed_imports=False,
)
