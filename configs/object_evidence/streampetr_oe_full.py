"""Full-data StreamPETR persistent-fault adaptation with OE."""

_base_ = "./streampetr_r0_full.py"

model = dict(object_evidence=dict(lambda_oe=0.5))
