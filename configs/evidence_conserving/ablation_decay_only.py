_base_ = ['./mini_debug.py']

# Provenance-aware Beta decay driven by the original class confidence. Ternary
# supervision and observability-conditioned negative redefinition are disabled.
model = dict(
    pts_bbox_head=dict(
        ternary_loss_weight=0.0,
        enable_observability_conditioning=False,
        enable_evidence_memory=True,
        evidence_probability_source='classification',
        calibrate_detection_scores=True,
    )
)
