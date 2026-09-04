"""Frozen full-nuScenes paired mechanism inference (deployment K=100)."""

_base_ = "./stream_petr_r50_90e_clean_val.py"

diagnostic_keys = [
    "camera_online_mask",
    "camera_quality",
    "camera_fresh_mask",
    "corruption_severity",
]
base_collect_keys = [
    "lidar2img",
    "intrinsics",
    "extrinsics",
    "timestamp",
    "img_timestamp",
    "ego_pose",
    "ego_pose_inv",
]
mechanism_collect_keys = base_collect_keys + diagnostic_keys
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
ida_aug_conf = dict(
    resize_lim=(0.38, 0.55),
    final_dim=(256, 704),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=900,
    W=1600,
    rand_flip=True,
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="ApplyPartialObservation",
        training=False,
        schedule_file=None,
        seed=2026,
        camera_crash_prob=0.0,
        frame_lost_prob=0.0,
        dark_prob=0.0,
        fog_prob=0.0,
        motion_blur_prob=0.0,
    ),
    dict(
        type="ResizeCropFlipRotImage",
        data_aug_conf=ida_aug_conf,
        training=False,
    ),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="PETRFormatBundle3D",
                collect_keys=mechanism_collect_keys,
                class_names=class_names,
                with_label=False,
            ),
            dict(
                type="Collect3D",
                keys=["img"] + mechanism_collect_keys,
                meta_keys=(
                    "filename",
                    "ori_shape",
                    "img_shape",
                    "pad_shape",
                    "scale_factor",
                    "flip",
                    "box_mode_3d",
                    "box_type_3d",
                    "img_norm_cfg",
                    "scene_token",
                    "sample_idx",
                    "frame_idx",
                ),
            ),
        ],
    ),
]

data = dict(
    val=dict(
        pipeline=test_pipeline,
        collect_keys=mechanism_collect_keys + ["img", "img_metas"],
    ),
    test=dict(
        pipeline=test_pipeline,
        collect_keys=mechanism_collect_keys + ["img", "img_metas"],
    ),
)

model = dict(pts_bbox_head=dict(bbox_coder=dict(max_num=100)))

custom_imports = dict(
    imports=["evidence3d_plugin", "analysis.gt_query_survival_trace"],
    allow_failed_imports=False,
)
