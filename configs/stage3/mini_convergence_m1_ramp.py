"""M1-Ramp: M1 forward semantics with only auxiliary-loss ramping."""

_base_ = "./mini_convergence_common.py"

work_dir = "outputs/stage3/mini_convergence_loss_balance/m1_ramp"

custom_hooks = [
    dict(type="EvidenceTraceHook", interval=10),
    dict(
        type="MilestoneCheckpointHook",
        milestones=(323, 969, 1938),
    ),
    dict(
        type="AuxiliaryLossRampHook",
        zero_until=0.2,
        full_at=0.5,
    ),
]
