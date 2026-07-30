"""R0: 50% clean / 50% uniformly sampled corruption, GT losses only."""

_base_ = "./mini_observability_b0.py"

work_dir = "outputs/stage3/observability_distillation/r0"
train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True, with_bbox=True, with_label=True, with_bbox_depth=True),
    dict(type="ObjectRangeFilter", point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(type="ObjectNameFilter", classes=["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]),
    dict(type="ResizeCropFlipRotImage", data_aug_conf=dict(resize_lim=(0.38, 0.55), final_dim=(256, 704), bot_pct_lim=(0.0, 0.0), rot_lim=(0.0, 0.0), H=900, W=1600, rand_flip=True), training=True),
    dict(type="GlobalRotScaleTransImage", rot_range=[-0.3925, 0.3925], translation_std=[0, 0, 0], scale_ratio_range=[0.95, 1.05], reverse_angle=True, training=True),
    dict(type="ApplyPartialObservation", training=True, seed=2026, exclusive_uniform=True, corruption_probability=0.5, max_severity=0.8),
    dict(type="NormalizeMultiviewImage", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="PETRFormatBundle3D", class_names=["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"], collect_keys=["lidar2img", "intrinsics", "extrinsics", "timestamp", "img_timestamp", "ego_pose", "ego_pose_inv", "camera_online_mask", "camera_quality", "camera_fresh_mask", "corruption_severity", "prev_exists"]),
    dict(type="Collect3D", keys=["gt_bboxes_3d", "gt_labels_3d", "img", "gt_bboxes", "gt_labels", "centers2d", "depths", "prev_exists", "lidar2img", "intrinsics", "extrinsics", "timestamp", "img_timestamp", "ego_pose", "ego_pose_inv", "camera_online_mask", "camera_quality", "camera_fresh_mask", "corruption_severity"], meta_keys=("filename", "ori_shape", "img_shape", "pad_shape", "scale_factor", "flip", "box_mode_3d", "box_type_3d", "img_norm_cfg", "scene_token", "sample_idx", "frame_idx", "gt_bboxes_3d", "gt_labels_3d")),
]
data = dict(train=dict(pipeline=train_pipeline))

