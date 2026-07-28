_base_ = ['./mini_debug.py']

# Observability-conditioned background + ternary supervision, while preserving
# the official StreamPETR memory write and uncalibrated detection scores.
model = dict(
    pts_bbox_head=dict(
        enable_evidence_memory=False,
        evidence_probability_source='ternary',
        calibrate_detection_scores=False,
    )
)
