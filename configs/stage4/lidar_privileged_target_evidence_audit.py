"""No-update train-only LiDAR/GT target-evidence activation audit config."""

_base_ = "./feq_common.py"

target_train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    {{_base_.fault_observation_transform}},
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True,
         with_bbox=True, with_label=True, with_bbox_depth=True),
    dict(type="ObjectRangeFilter", point_cloud_range={{_base_.point_cloud_range}}),
    dict(type="ObjectNameFilter", classes={{_base_.class_names}}),
    dict(type="AttachLiDARPointCounts"),
    dict(type="ResizeCropFlipRotImage", data_aug_conf={{_base_.ida_aug_conf}}, training=True),
    dict(type="GlobalRotScaleTransImage", rot_range=[-0.3925, 0.3925],
         translation_std=[0, 0, 0], scale_ratio_range=[0.95, 1.05],
         reverse_angle=True, training=True),
    dict(type="NormalizeMultiviewImage", mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="PETRFormatBundle3D", class_names={{_base_.class_names}},
         collect_keys=["lidar2img", "intrinsics", "extrinsics", "timestamp",
                       "img_timestamp", "ego_pose", "ego_pose_inv",
                       "camera_online_mask", "camera_quality", "camera_fresh_mask",
                       "corruption_severity", "prev_exists"]),
    dict(type="Collect3D",
         keys=["gt_bboxes_3d", "gt_labels_3d", "img", "gt_bboxes",
               "gt_labels", "centers2d", "depths", "prev_exists",
               "lidar2img", "intrinsics", "extrinsics",
               "timestamp", "img_timestamp", "ego_pose", "ego_pose_inv",
               "camera_online_mask", "camera_quality", "camera_fresh_mask",
               "corruption_severity"],
         meta_keys=("filename", "ori_shape", "img_shape", "pad_shape",
                    "scale_factor", "flip", "box_mode_3d", "box_type_3d",
                    "img_norm_cfg", "scene_token", "sample_idx", "frame_idx",
                    "gt_bboxes_3d", "gt_labels_3d", "feq_history_centers",
                    "gt_lidar_point_counts")),
]

data = dict(train=dict(
    type="LiDARPrivilegedNuScenesDataset",
    pipeline=target_train_pipeline,
))

work_dir = "outputs/stage4/lidar_privileged_target_evidence_audit/audit_only"
load_from = "outputs/stage3/observability_distillation/b0/iter_969.pth"
model = dict(pts_bbox_head=dict(
    type="LiDARPrivilegedTargetEvidenceStreamPETRHead",
    enable_lidar_target_evidence=True,
    # Unit weight exposes the raw signal. This audit creates no optimizer.
    lidar_target_evidence_weight=1.0,
    lidar_target_geometry_threshold=2.0,
    lidar_target_fault_camera=3,
))
