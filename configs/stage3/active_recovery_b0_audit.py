"""Frozen B0 with environment-gated active recovery injection audit."""

_base_ = "./mini_convergence_b0.py"

custom_imports = dict(
    imports=[
        "evidence3d_plugin",
        "hooks.evidence_trace_hook",
        "hooks.loss_balance_hook",
        "analysis.active_recovery_injection",
    ],
    allow_failed_imports=False,
)
