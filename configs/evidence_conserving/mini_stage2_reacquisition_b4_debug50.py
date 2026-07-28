"""Fifty-iteration rescue run for zero-shot-qualified B4."""

import json
import os

_base_ = "./mini_stage2_reacquisition_b4.py"

work_dir = "outputs/stage2/s2_3_rescue/debug_50/b4"
runner = dict(type="IterBasedRunner", max_iters=50)
checkpoint_config = dict(interval=50, by_epoch=False, max_keep_ckpts=1)
log_config = dict(
    interval=1,
    hooks=[dict(type="TextLoggerHook", by_epoch=False)],
)

reacquisition_curriculum = dict(
    ratios=dict(
        clean=0.45,
        crash_or_lost=0.20,
        visual=0.15,
        long_fault=0.10,
        compound=0.10,
    ),
    durations=(1, 3, 5, 10, 20),
    cycle_frames=40,
)
os.environ["EVIDENCE3D_REACQUISITION_CURRICULUM_JSON"] = json.dumps(
    reacquisition_curriculum
)

custom_hooks = [
    dict(
        type="EvidenceTraceHook",
        interval=1,
        out_file="reacquisition_train_trace.jsonl",
    ),
    dict(
        type="FreezeExceptHook",
        trainable_patterns=[
            "ternary_branches",
            "observability_head",
            "evidence_step",
        ],
        priority="VERY_HIGH",
    ),
]
