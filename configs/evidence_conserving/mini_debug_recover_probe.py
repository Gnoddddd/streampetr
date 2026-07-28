"""Single-GPU nuScenes-mini debug config for Evidence3D + StreamPETR.

Run from the project wrapper, not directly from Windows PowerShell:
    conda activate streampetr
    python tools/train.py --config configs/evidence_conserving/mini_debug.py
"""

import os
from pathlib import Path

_base_ = [
    '../../repos/StreamPETR/projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_24e.py'
]

PROJECT_ROOT = Path(os.environ.get("EVIDENCE3D_PROJECT_ROOT", os.getcwd())).resolve()
DATA_ROOT = Path(os.environ.get('EVIDENCE3D_DATA_ROOT', PROJECT_ROOT / 'data/nuscenes-mini'))
PROTOCOL_FILE = os.environ.get('EVIDENCE3D_PROTOCOL', None)

# The official plugin is loaded from the StreamPETR working directory. The
# custom bridge is importable through PYTHONPATH set by tools/train.py.
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
custom_imports = dict(imports=['evidence3d_plugin'], allow_failed_imports=False)

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)
collect_keys = [
    'lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp',
    'ego_pose', 'ego_pose_inv', 'camera_online_mask', 'camera_quality',
    'camera_fresh_mask', 'corruption_severity'
]
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True,
)

model = dict(
    type='Petr3D',
    use_grid_mask=False,
    num_frame_head_grads=1,
    num_frame_backbone_grads=1,
    num_frame_losses=1,
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        with_cp=True,
        frozen_stages=-1,
        norm_eval=True,
    ),
    pts_bbox_head=dict(
        type='EvidenceConservingStreamPETRHead',
        num_query=256,
        memory_len=256,
        topk_proposals=64,
        num_propagated=64,
        scalar=5,
        num_cameras=6,
        ternary_loss_weight=1.0,
        background_observability_floor=0.0,
        evidence_warmup_steps=200,
        calibrate_detection_scores=True,
        trace_enabled=True,
        observability_cfg=dict(
            min_depth=0.1,
            boundary_softness=8.0,
            depth_temperature=4.0,
            learned_residual=False,
            residual_weight=0.0,
        ),
        temporal_update_cfg=dict(
            gamma=0.90,
            evidence_scale=2.0,
            max_effective_count=6.0,
        ),
        policy_cfg=dict(
            keep_observability=0.45,
            keep_presence=0.55,
            keep_max_uncertainty=0.55,
            recover_presence=0.55,
            recover_max_uncertainty=0.80,
            recover_max_age=3,
            recover_min_prior_strength=0.5,
            strong_negative=0.60,
            recover_score_scale=0.75,
            defer_score_scale=0.20,
        ),
        transformer=dict(
            decoder=dict(
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        # FlashAttention 0.2.x is deliberately not required.
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                            fp16=True,
                        ),
                    ],
                    feedforward_channels=1024,
                    with_cp=True,
                )
            )
        ),
        bbox_coder=dict(max_num=100),
    ),
)

ida_aug_conf = dict(
    resize_lim=(0.38, 0.55),
    final_dim=(256, 704),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=900,
    W=1600,
    rand_flip=True,
)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='ApplyPartialObservation',
        training=True,
        seed=2026,
        camera_crash_prob=0.25,
        max_failed_cameras=2,
        frame_lost_prob=0.10,
        dark_prob=0.10,
        fog_prob=0.08,
        motion_blur_prob=0.08,
        max_severity=0.80,
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=True,
        with_label=True,
        with_bbox_depth=True,
    ),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=True),
    dict(
        type='GlobalRotScaleTransImage',
        rot_range=[-0.3925, 0.3925],
        translation_std=[0, 0, 0],
        scale_ratio_range=[0.95, 1.05],
        reverse_angle=True,
        training=True,
    ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='PETRFormatBundle3D',
        class_names=class_names,
        collect_keys=collect_keys + ['prev_exists'],
    ),
    dict(
        type='Collect3D',
        keys=[
            'gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels',
            'centers2d', 'depths', 'prev_exists'
        ] + collect_keys,
        meta_keys=(
            'filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor',
            'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
            'scene_token', 'sample_idx', 'frame_idx', 'gt_bboxes_3d',
            'gt_labels_3d'
        ),
    ),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='ApplyPartialObservation',
        training=False,
        schedule_file=PROTOCOL_FILE,
        seed=2026,
        camera_crash_prob=0.0,
        frame_lost_prob=0.0,
        dark_prob=0.0,
        fog_prob=0.0,
        motion_blur_prob=0.0,
    ),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys,
                class_names=class_names,
                with_label=False,
            ),
            dict(
                type='Collect3D',
                keys=['img'] + collect_keys,
                meta_keys=(
                    'filename', 'ori_shape', 'img_shape', 'pad_shape',
                    'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'scene_token', 'sample_idx', 'frame_idx'
                ),
            ),
        ],
    ),
]

data_root = str(DATA_ROOT) + '/'
dataset_type = 'EvidenceNuScenesDataset'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes2d_temporal_infos_train.pkl',
        num_frame_losses=1,
        seq_split_num=1,
        seq_mode=True,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas'],
        queue_length=1,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        box_type_3d='LiDAR',
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        collect_keys=collect_keys + ['img', 'img_metas'],
        queue_length=1,
        ann_file=data_root + 'nuscenes2d_temporal_infos_val.pkl',
        classes=class_names,
        modality=input_modality,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        collect_keys=collect_keys + ['img', 'img_metas'],
        queue_length=1,
        ann_file=data_root + 'nuscenes2d_temporal_infos_val.pkl',
        classes=class_names,
        modality=input_modality,
    ),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

optimizer = dict(
    type='AdamW',
    lr=2.5e-5,
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.25)}),
    weight_decay=0.01,
)
optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale='dynamic',
    grad_clip=dict(max_norm=35, norm_type=2),
)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
runner = dict(type='IterBasedRunner', max_iters=400)
checkpoint_config = dict(interval=100, max_keep_ckpts=3)
evaluation = dict(interval=400, pipeline=test_pipeline)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook', by_epoch=False)],
)
custom_hooks = [dict(type='EvidenceTraceHook', interval=20)]
workflow = [('train', 1)]
find_unused_parameters = False
load_from = None
resume_from = None
seed = 2026

# Remove imported Path from MMCV config namespace.
# Otherwise Config.dump() serializes:
# Path=<class 'pathlib.Path'>

# MMCV-safe project root and pathlib cleanup
# Config.fromfile may execute a copied config, so __file__ is not reliable.
# Convert pathlib values to plain strings before Config.dump().
def _mmcv_stringify_paths(_value):
    if isinstance(_value, Path):
        return str(_value)
    if isinstance(_value, dict):
        return {
            _key: _mmcv_stringify_paths(_item)
            for _key, _item in _value.items()
        }
    if isinstance(_value, list):
        return [_mmcv_stringify_paths(_item) for _item in _value]
    if isinstance(_value, tuple):
        return tuple(_mmcv_stringify_paths(_item) for _item in _value)
    return _value


for _config_name in list(globals()):
    if _config_name.startswith("__"):
        continue
    if _config_name in {
        "Path",
        "os",
        "_mmcv_stringify_paths",
        "_config_name",
    }:
        continue
    globals()[_config_name] = _mmcv_stringify_paths(
        globals()[_config_name]
    )

del _config_name
del _mmcv_stringify_paths
del Path
del os


model["pts_bbox_head"][
    "evidence_probability_source"
] = "classification"

# Temporary policy probe for propagated-query diagnostics.
model["pts_bbox_head"][
    "evidence_probability_source"
] = "classification"

model["pts_bbox_head"]["policy_cfg"].update(
    keep_presence=0.28,
    recover_presence=0.30,
)

# Diagnostic only: relax strong-negative gating.
model["pts_bbox_head"]["policy_cfg"].update(
    strong_negative=0.95,
)
