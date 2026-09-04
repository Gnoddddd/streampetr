"""Frozen B0 configuration for paired, no-update rank objective audit."""

_base_ = "./feq_common.py"

work_dir = "outputs/stage4/paired_fault_rank_objective_audit/audit_only"
load_from = "outputs/stage3/observability_distillation/b0/iter_969.pth"

# The script builds Clean and three fixed-fault copies of this mini-train
# dataset. B0 itself is deliberately unchanged.
