"""Frozen CARE-3D P2-A0 online query-association pre-gate configuration."""

protocols = ("blur_back", "crash_back", "dark_back")
protocol_files = {
    "blur_back": "protocols/presets/motion_blur_back_10f_s09.json",
    "crash_back": "protocols/presets/camera_crash_back_10f.json",
    "dark_back": "protocols/presets/dark_back_10f_s09.json",
}

stream_petr_config = "configs/full_nuscenes/stream_petr_r50_90e_ctep_train_audit.py"
stream_petr_checkpoint = "checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth"
train_info = "data/nuscenes/nuscenes2d_temporal_infos_train.pkl"
val_info = "data/nuscenes/nuscenes2d_temporal_infos_val.pkl"
p0_report = "reports/care3d/p0_counterfactual_vulnerability"
p1_report = "reports/care3d/p1_sparse_evidence_router"
report_dir = "reports/care3d/p2a_online_query_association"

association = dict(
    query_count=900,
    current_query_count=644,
    propagated_query_count=256,
    feature_tap="final_decoder_pre_cls_query",
    max_geometry_distance_m=12.0,
    one_to_one_solver="hungarian_with_private_unmatched_dummy",
    geometry_cost="min(xy_predicted_center_distance/12m,1)",
    embedding_cost="(1-cosine(anchor_query,fault_query))/2",
    class_cost="1-sigmoid(fault_logit[anchor_predicted_class])",
    query_collision_policy="exclude_all_rows_in_shared_anchor_or_target_query_frame",
    gt_used_as_association_input=False,
    clean_future_used_as_association_input=False,
    oracle_query_used_as_association_input=False,
)

weight_grid = (
    (0.5, 0.3, 0.2),
    (0.4, 0.4, 0.2),
    (0.6, 0.2, 0.2),
    (0.4, 0.3, 0.3),
    (0.5, 0.2, 0.3),
)
max_cost_grid = (0.35, 0.45, 0.55)

baselines = dict(
    geometry_only=(1.0, 0.0, 0.0),
    embedding_only=(0.0, 1.0, 0.0),
    class_geometry=(0.5, 0.0, 0.5),
    use_selected_full_max_cost=True,
)

selection = dict(
    fit_split="probe_train",
    confirmation_split="probe_val",
    probe_test_locked=True,
    protocol_specific_parameters=False,
    primary="protocol_macro_mean_exact_oracle_query_recall_at_1",
    tie_break_1="lower_protocol_macro_wrong_match_rate",
    tie_break_2="lower_protocol_macro_mean_accepted_cost",
    tie_break_3="config_id_lexicographic",
)

gate = dict(
    min_passing_fault_families=2,
    min_exact_recall=0.70,
    bootstrap_repetitions=5000,
    min_scene_cluster_ci_low=0.50,
    min_instance_cluster_ci_low=0.50,
    max_wrong_match_rate=0.10,
)

execution = dict(
    detector_frozen=True,
    p0_frozen=True,
    p1_frozen=True,
    association_neural_training=False,
    cost_matrix_torch_vectorized=True,
    model_initialized_once_per_process=True,
    fault_protocols_sequential=True,
    detector_classifier_execution_shape_unchanged=True,
    formal_probe_test_prohibited_in_p2a0=True,
)
