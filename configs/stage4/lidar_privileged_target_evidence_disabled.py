"""Frozen B0 inference with privileged target evidence explicitly disabled."""

_base_ = "../stage3/mini_convergence_b0.py"

model = dict(pts_bbox_head=dict(
    type="LiDARPrivilegedTargetEvidenceStreamPETRHead",
    enable_lidar_target_evidence=False,
    lidar_target_evidence_weight=0.0,
))
