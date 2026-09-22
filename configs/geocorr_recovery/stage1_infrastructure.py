"""Engineering defaults only; these values are not experiment selections."""

candidate_radius_m = 1.0
candidate_offsets = [
    (-candidate_radius_m, -candidate_radius_m),
    (-candidate_radius_m, 0.0),
    (-candidate_radius_m, candidate_radius_m),
    (0.0, -candidate_radius_m),
    (0.0, 0.0),
    (0.0, candidate_radius_m),
    (candidate_radius_m, -candidate_radius_m),
    (candidate_radius_m, 0.0),
    (candidate_radius_m, candidate_radius_m),
]

geometry_sampler = dict(
    candidate_offsets=candidate_offsets,
    min_depth=1e-5,
    align_corners=False,
    feature_shape=(16, 44),
)

# Must be changed to the verified OccNuScenes layout after data is installed.
occ_nuscenes = dict(
    root="data/occ_nuscenes",
    layout="{corruption_type}/{severity}/{relative_path}",
    supported_corruptions=("Dirt", "Water-blur"),
)
