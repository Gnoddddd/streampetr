"""Real 50-iteration S2.2 source-ledger tracking validation."""

_base_ = "./mini_stage2_ledger_debug_50.py"

work_dir = "outputs/stage2/s2_2_source_ledger_debug_50"

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
