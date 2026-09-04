"""Frozen common settings for the FEQ-Core nuScenes-mini screen."""

_base_ = "../stage3/mini_convergence_b0.py"

seed = 2026
iters_per_epoch = 323
max_iters = 969
checkpoint_milestones = (323, 969)
runner = dict(type="IterBasedRunner", max_iters=max_iters)
checkpoint_config = dict(interval=max_iters, by_epoch=False, max_keep_ckpts=2)
custom_hooks = [
    dict(type="MilestoneCheckpointHook", milestones=(323,)),
]

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375], to_rgb=True)
ida_aug_conf = dict(resize_lim=(0.38, 0.55), final_dim=(256, 704),
                    bot_pct_lim=(0.0, 0.0), rot_lim=(0.0, 0.0), H=900,
                    W=1600, rand_flip=True)
collect_keys = [
    "lidar2img", "intrinsics", "extrinsics", "timestamp", "img_timestamp",
    "ego_pose", "ego_pose_inv", "camera_online_mask", "camera_quality",
    "camera_fresh_mask", "corruption_severity",
]

# Child configs replace only this transform to select clean versus frozen fault.
observation_transform = dict(
    type="ApplyPartialObservation", training=True, seed=314159,
    camera_crash_prob=0.0, frame_lost_prob=0.0, dark_prob=0.0,
    fog_prob=0.0, motion_blur_prob=0.0,
)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    observation_transform,
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True,
         with_bbox=True, with_label=True, with_bbox_depth=True),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="ResizeCropFlipRotImage", data_aug_conf=ida_aug_conf, training=True),
    dict(type="GlobalRotScaleTransImage", rot_range=[-0.3925, 0.3925],
         translation_std=[0, 0, 0], scale_ratio_range=[0.95, 1.05],
         reverse_angle=True, training=True),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="PETRFormatBundle3D", class_names=class_names,
         collect_keys=collect_keys + ["prev_exists"]),
    dict(type="Collect3D",
         keys=["gt_bboxes_3d", "gt_labels_3d", "img", "gt_bboxes",
               "gt_labels", "centers2d", "depths", "prev_exists"] + collect_keys,
         meta_keys=("filename", "ori_shape", "img_shape", "pad_shape",
                    "scale_factor", "flip", "box_mode_3d", "box_type_3d",
                    "img_norm_cfg", "scene_token", "sample_idx", "frame_idx",
                    "gt_bboxes_3d", "gt_labels_3d", "feq_history_centers")),
]

fault_observation_transform = dict(
    type="ApplyPartialObservation", training=True, seed=314159,
    schedule_file="protocols/feq_core/train_episode_seed314159.json",
    camera_crash_prob=0.0, frame_lost_prob=0.0, dark_prob=0.0,
    fog_prob=0.0, motion_blur_prob=0.0,
)
fault_train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    fault_observation_transform,
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True,
         with_bbox=True, with_label=True, with_bbox_depth=True),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="ResizeCropFlipRotImage", data_aug_conf=ida_aug_conf, training=True),
    dict(type="GlobalRotScaleTransImage", rot_range=[-0.3925, 0.3925],
         translation_std=[0, 0, 0], scale_ratio_range=[0.95, 1.05],
         reverse_angle=True, training=True),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="PETRFormatBundle3D", class_names=class_names,
         collect_keys=collect_keys + ["prev_exists"]),
    dict(type="Collect3D",
         keys=["gt_bboxes_3d", "gt_labels_3d", "img", "gt_bboxes",
               "gt_labels", "centers2d", "depths", "prev_exists"] + collect_keys,
         meta_keys=("filename", "ori_shape", "img_shape", "pad_shape",
                    "scale_factor", "flip", "box_mode_3d", "box_type_3d",
                    "img_norm_cfg", "scene_token", "sample_idx", "frame_idx",
                    "gt_bboxes_3d", "gt_labels_3d", "feq_history_centers")),
]

data = dict(train=dict(
    pipeline=train_pipeline,
    feq_history_file="protocols/feq_core/train_history_seed314159.json",
))
