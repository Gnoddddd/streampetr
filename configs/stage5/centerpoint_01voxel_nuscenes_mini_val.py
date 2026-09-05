"""Official CenterPoint 0.1 m voxel teacher on local nuScenes-mini val."""

_base_ = [
    "../../repos/mmdetection3d/configs/centerpoint/"
    "centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py"
]

data_root = "data/nuscenes-mini/"
ann_file = data_root + "nuscenes2d_temporal_infos_val.pkl"

model = dict(pretrained=None, train_cfg=None)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    val=dict(data_root=data_root, ann_file=ann_file, test_mode=True),
    test=dict(data_root=data_root, ann_file=ann_file, test_mode=True),
)
