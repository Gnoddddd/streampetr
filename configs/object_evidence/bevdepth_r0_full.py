"""Full-data BEVDepth 24e-2key persistent-fault adaptation baseline."""

import os

data_root = "data/nuscenes/"
train_info = "data/nuscenes/nuscenes_infos_train.pkl"
pretrained_checkpoint = (
    "checkpoints/official/bev_depth_lss_r50_256x704_128x128_24e_2key.pth"
)
batch_size_per_device = 1
adaptation_max_steps = int(os.environ.get("OE_ADAPTATION_MAX_STEPS", "0"))

object_evidence = dict(
    enabled=True,
    seed=2026,
    pair_probability=0.5,
    lambda_oe=0.0,
    lambda_pg=0.0,
    auxiliary_warmup_iters=1000,
)
