_base_ = [
    "./mini_debug_source_aware_ft400_fp32.py"
]

# 与Classification冻结基线分开保存。
work_dir = (
    "outputs/"
    "exp_007_ternary_ft400_fp32"
)

# 正式Ternary版本：
# 1. 三态概率进入证据账本；
# 2. 保留证据记忆；
# 3. 保留可观测性条件化；
# 4. 保留检测分数校准；
# 5. 使用原有200步证据门控预热。
model = dict(
    pts_bbox_head=dict(
        evidence_probability_source="ternary",
        ternary_loss_weight=1.0,
        enable_evidence_memory=True,
        enable_observability_conditioning=True,
        calibrate_detection_scores=True,
        evidence_warmup_steps=200,
    )
)

runner = dict(
    max_iters=400,
)

checkpoint_config = dict(
    interval=100,
    max_keep_ckpts=4,
)

log_config = dict(
    interval=20,
)

# 与冻结Classification基线一致，继续使用FP32。
fp16 = None

optimizer_config = dict(
    _delete_=True,
    type="OptimizerHook",
    grad_clip=dict(
        max_norm=35,
        norm_type=2,
    ),
)

# 训练结束后单独评测。
evaluation = dict(
    interval=100000,
)
