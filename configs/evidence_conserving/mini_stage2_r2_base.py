"""Shared, pre-registered S2.3-R2 formal experiment configuration."""

_base_ = "./mini_stage2_reacquisition_b4.py"

load_from = (
    "/home/research/research/evidence3d/outputs/stage2/"
    "s2_2_source_ledger_debug_50/iter_50.pth"
)
resume_from = None

model = dict(pts_bbox_head=dict(
    enable_reacquisition_diagnostics=True,
    enable_memory_isolation=True,
    confirmation_frames=2,
    pending_max_age=3,
    class_consistency_required=True,
    center_distance_threshold=2.0,
    motion_distance_threshold=2.0,
    minimum_confirmation_score=0.075,
    minimum_confirmation_reliability=0.65,
    allow_pending_memory_write=False,
))
