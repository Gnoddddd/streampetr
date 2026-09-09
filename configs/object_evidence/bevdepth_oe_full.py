"""Full-data BEVDepth 24e-2key persistent-fault adaptation with OE."""

from configs.object_evidence.bevdepth_r0_full import *  # noqa: F401,F403

object_evidence = dict(object_evidence, lambda_oe=0.5)
