"""Stage-2 engineering defaults; none are experimentally tuned."""

from .stage1_infrastructure import candidate_offsets, candidate_radius_m

num_candidates = 9
candidate_radius = candidate_radius_m
history_frames = 1
feature_dim = 256

# Engineering defaults only. Stage 2 performs no hyperparameter search.
temperature = 0.1
geometry_prior_beta = 0.0
detach_teacher = True
detach_history = True
detach_dirty_base = True

correlation_field = dict(
    candidate_offsets=candidate_offsets,
    feature_dim=feature_dim,
    geometry_prior_beta=geometry_prior_beta,
    detach_teacher=detach_teacher,
    detach_history=detach_history,
    detach_dirty_base=detach_dirty_base,
)
