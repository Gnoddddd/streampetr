"""B0: original StreamPETR head at the same architecture and training budget."""

_base_ = "./mini_convergence_common.py"

work_dir = "outputs/stage3/mini_convergence_loss_balance/b0"

model = dict(
    pts_bbox_head=dict(
        _delete_=True,
        type="StreamPETRHead",
        num_classes=10,
        in_channels=256,
        num_query=644,
        memory_len=1024,
        topk_proposals=256,
        num_propagated=256,
        with_dn=False,
        with_ego_pos=True,
        match_with_velo=False,
        scalar=5,
        noise_scale=1.0,
        dn_weight=1.0,
        split=0.75,
        LID=True,
        with_position=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type="PETRTemporalTransformer",
            decoder=dict(
                type="PETRTransformerDecoder",
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type="PETRTemporalDecoderLayer",
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
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
        ),
        bbox_coder=dict(
            type="NMSFreeCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=100,
            voxel_size=[0.2, 0.2, 8],
            num_classes=10,
        ),
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        loss_iou=dict(type="GIoULoss", loss_weight=0.0),
    ),
)
