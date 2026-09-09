"""BEVDepth OE-PG V1 with offline privileged geometry guidance."""

_base_ = "./bevdepth_oep.py"

object_evidence = dict(
    lambda_pg=0.25,
    teacher_cache="data/object_evidence/lidar_teacher_train",
)
