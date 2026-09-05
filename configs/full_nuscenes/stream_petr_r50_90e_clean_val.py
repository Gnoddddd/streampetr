"""Official StreamPETR R50-900q 90e full-nuScenes clean-val baseline.

The detector and evaluation settings inherit the checkpoint-matched upstream
configuration.  Only filesystem/runtime settings are changed here.  The
standard PETR cross-attention is the upstream-documented fallback for hosts
without the optional ``flash_attn`` package.
"""

_base_ = (
    "../../repos/StreamPETR/projects/configs/StreamPETR/"
    "stream_petr_r50_flash_704_bs2_seq_90e.py"
)

data_root = "/home/research/research/evidence3d/data/nuscenes/"
ann_file = data_root + "nuscenes2d_temporal_infos_val.pkl"

data = dict(
    workers_per_gpu=4,
    val=dict(data_root=data_root, ann_file=ann_file),
    test=dict(data_root=data_root, ann_file=ann_file),
)

model = dict(
    img_backbone=dict(pretrained=None),
    pts_bbox_head=dict(
        transformer=dict(
            decoder=dict(
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type="PETRMultiheadAttention",
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                            fp16=True,
                        ),
                    ]
                )
            )
        )
    ),
)

seed = 2026
