import os

_base_ = ['../evidence_conserving/mini_official_r50_900q_clean_eval.py']

# Full nuScenes v1.0-trainval configuration.
# The mounted dataset path is supplied at runtime so the repository does not
# hard-code one machine-specific disk mount.
full_data_root = os.environ.get(
    'EVIDENCE3D_DATA_ROOT',
    '/home/research/research/evidence3d/data/nuscenes',
)
train_info = os.path.join(full_data_root, 'nuscenes2d_temporal_infos_train.pkl')
val_info = os.path.join(full_data_root, 'nuscenes2d_temporal_infos_val.pkl')

data_root = full_data_root

data = dict(
    train=dict(
        data_root=full_data_root,
        ann_file=train_info,
    ),
    val=dict(
        data_root=full_data_root,
        ann_file=val_info,
    ),
    test=dict(
        data_root=full_data_root,
        ann_file=val_info,
    ),
)

work_dir = 'outputs/care3d/full_nuscenes_eval'
