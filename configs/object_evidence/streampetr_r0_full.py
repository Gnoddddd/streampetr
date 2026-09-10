"""Full-data StreamPETR persistent-fault adaptation baseline (R0)."""

import os

_base_ = (
    "../../repos/StreamPETR/projects/configs/StreamPETR/"
    "stream_petr_r50_flash_704_bs2_seq_90e.py"
)

custom_imports = dict(imports=["evidence3d_plugin"], allow_failed_imports=False)
seed = 2026
adaptation_max_steps = int(os.environ.get("OE_ADAPTATION_MAX_STEPS", "0"))

model = dict(
    type="OEStreamPETR",
    img_backbone=dict(pretrained=None),
    pts_bbox_head=dict(
        transformer=dict(
            decoder=dict(
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention", embed_dims=256,
                            num_heads=8, dropout=0.1,
                        ),
                        dict(
                            type="PETRMultiheadAttention", embed_dims=256,
                            num_heads=8, dropout=0.1, fp16=True,
                        ),
                    ]
                )
            )
        )
    ),
    object_evidence=dict(
        enabled=True,
        seed=seed,
        pair_probability=0.5,
        onset_frames=2,
        lambda_oe=0.0,
        lambda_pg=0.0,
        auxiliary_warmup_iters=1000,
    ),
)

data = dict(
    train=dict(
        type="OEPSequenceNuScenesDataset",
        data_root="data/nuscenes/",
        ann_file="data/nuscenes/nuscenes2d_temporal_infos_train.pkl",
    ),
)

load_from = "checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth"
resume_from = None
runner = dict(type="IterBasedRunner", max_iters=adaptation_max_steps)
checkpoint_config = dict(interval=max(1, adaptation_max_steps), max_keep_ckpts=3)
