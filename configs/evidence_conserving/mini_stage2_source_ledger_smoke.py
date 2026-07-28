"""S2.2 source-ledger smoke: tracking only, with no decision coupling."""

_base_ = "./mini_stage2_ledger_smoke.py"

work_dir = "outputs/stage2/s2_2_source_ledger_smoke"
load_from = (
    "/home/research/research/evidence3d/outputs/final_snapshots/"
    "stage1_ternary_r50_200/checkpoint/iter_200.pth"
)
resume_from = None

# Match the accepted S2.1 integration precision and clipping policy.
fp16 = dict(loss_scale="dynamic")
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

source_camera_names = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

model = dict(
    pts_bbox_head=dict(
        enable_source_ledger=True,
        source_decay=0.9,
        source_mass_tolerance=1e-5,
        use_source_ledger_for_evidence=False,
        use_source_ledger_for_policy=False,
        source_camera_names=source_camera_names,
    ),
)
