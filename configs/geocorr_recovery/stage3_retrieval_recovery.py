"""Stage 3-B engineering defaults; loss weights remain experiment inputs."""

from .stage1_infrastructure import candidate_offsets

top_k = 25
supported_top_k = (25, 50, "all")
candidate_grid = (3, 3)
descriptor_dim = 256
confidence = "normalized_entropy"
learned_gate = False
recovery_zero_init = True

# Deliberately unset until the training protocol supplies registered values.
lambda_corr = None
lambda_rec = None

historical_retrieval = dict(confidence=confidence)
residual_recovery = dict(
    feature_dim=descriptor_dim,
    zero_init=recovery_zero_init,
)
sparse_writeback = dict(aggregation="confidence_weighted_mean", residual_add=True)
stage3_recovery = dict(
    feature_dim=descriptor_dim,
    top_k=top_k,
    recovery_zero_init=recovery_zero_init,
)

assert len(candidate_offsets) == candidate_grid[0] * candidate_grid[1]
