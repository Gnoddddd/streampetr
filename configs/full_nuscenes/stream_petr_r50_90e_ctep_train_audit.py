"""Deterministic train-split input for the CTEP activation audit.

This keeps the checkpoint-matched detector and the frozen mechanism pipeline;
only the annotation split changes from val to train.  Optimization is not
configured here.
"""

_base_ = "./stream_petr_r50_90e_mechanism_val.py"

data_root = "/home/research/research/evidence3d/data/nuscenes/"
train_ann_file = data_root + "nuscenes2d_temporal_infos_train.pkl"

data = dict(
    val=dict(data_root=data_root, ann_file=train_ann_file),
    test=dict(data_root=data_root, ann_file=train_ann_file),
)

