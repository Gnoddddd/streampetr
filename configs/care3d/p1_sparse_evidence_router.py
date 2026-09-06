"""Frozen CARE-3D P1 sparse evidence router configuration.

P1 is unlocked only after the main P0 and the preregistered cross-severity
transfer have passed.  StreamPETR and all P0 predictors remain frozen.
"""

protocols = ("blur_back", "crash_back", "dark_back")
protocol_files = {
    "blur_back": "protocols/presets/motion_blur_back_10f_s09.json",
    "crash_back": "protocols/presets/camera_crash_back_10f.json",
    "dark_back": "protocols/presets/dark_back_10f_s09.json",
}

stream_petr_config = "configs/full_nuscenes/stream_petr_r50_90e_ctep_train_audit.py"
stream_petr_checkpoint = "checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth"
main_p0_report = "reports/care3d/p0_counterfactual_vulnerability"
cross_severity_report = "reports/care3d/p0_cross_severity"
report_dir = "reports/care3d/p1_sparse_evidence_router"

# P1 source bank.  The failed CAM_BACK source is intentionally absent.
source_bank = dict(
    names=("CAM_BACK_LEFT", "CAM_BACK_RIGHT", "TEMPORAL_ANCHOR"),
    camera_names=("CAM_BACK_LEFT", "CAM_BACK_RIGHT"),
    temporal_source="final_decoder_pre_cls_query_at_t",
    camera_source="FPN_P0_bilinear_at_fault_query_predicted_center",
    failed_camera="CAM_BACK",
    source_dim=256,
)

router = dict(
    object_dim=256,
    source_dim=256,
    vulnerability_dim=3,
    hidden_dim=256,
    top_k=2,
    risk_gate="frozen_p0_protocol_crossing_probability",
    clean_bypass=True,
    classification_only=True,
    regression_unchanged=True,
)

loss = dict(
    score_weight=1.0,
    boundary_weight=1.0,
    query_weight=0.25,
    retained_weight=0.5,
    non_target_weight=0.25,
    boundary_margin=0.01,
)

training = dict(
    seeds=(42, 2027, 2028),
    epochs=30,
    batch_size=512,
    learning_rate=3e-4,
    weight_decay=1e-4,
    num_workers=0,
    patience=6,
    gradient_clip_norm=5.0,
    balanced_protocol_label_sampling=True,
)

# Formal P1 gate.  No threshold may be changed after probe-test is unlocked.
gate = dict(
    min_passing_fault_families=2,
    bootstrap_repetitions=5000,
    require_scene_cluster_ci=True,
    require_instance_cluster_ci=True,
    require_lost_recovery_ci_low_gt_zero=True,
    require_net_tp_ci_low_gt_zero=True,
    require_cross_topk_recovery_ci_low_gt_zero=True,
    require_target_score_delta_ci_low_gt_zero=True,
    max_retained_damage_rate=0.005,
    max_retained_damage_ci_high=0.01,
    max_fp_inflation_rate=0.01,
    max_fp_inflation_ci_high=0.02,
    require_clean_identity=True,
)

intervention = dict(
    topk=100,
    score_threshold=0.1,
    fixed_clean_query_experimental_unit=True,
    one_step_from_same_clean_history=True,
    allow_fault_history=False,
    gt_used_as_router_input=False,
    clean_future_used_as_router_input=False,
    test_locked_until_all_router_checkpoints_frozen=True,
)
