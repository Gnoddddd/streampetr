"""StreamPETR OE-PG V1: Object Evidence Preservation only."""

_base_ = "../stage3/mini_convergence_b0.py"

object_evidence = dict(
    enabled=True,
    architecture="streampetr",
    adapter="StreamPETRAdapter",
    pair_probability=0.5,
    seed=2026,
    lambda_oe=0.5,
    lambda_pg=0.0,
    auxiliary_warmup_iters=1000,
    teacher_cache=None,
)
