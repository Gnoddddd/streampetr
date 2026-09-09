"""BEVDepth OE-PG V1: Object Evidence Preservation only."""

architecture = "bevdepth"
source_config = "repos/BEVDepth/bevdepth/exps/nuscenes/base_exp.py"
object_evidence = dict(
    enabled=True,
    adapter="BEVDepthAdapter",
    pair_probability=0.5,
    seed=2026,
    lambda_oe=0.5,
    lambda_pg=0.0,
    auxiliary_warmup_iters=1000,
    teacher_cache=None,
    fault_depth_reliability=True,
)
