# 实验顺序与消融约定

## 阶段 1：基线与现象验证

1. 干净数据官方 StreamPETR；
2. Camera Crash：后、左、右、相邻双相机和随机组合；
3. Frame Lost：1/3/5/10/20 帧；
4. 统计不可观测区域中的 no-object、召回率和假阴性；
5. 统计目标离场后的置信度、虚假轨迹持续时间与证据强度。

只有确认 Unobserved-as-Background 与 Temporal Evidence Echo 后，才解释完整方法结果。

## 阶段 2：最小可行方法

- 几何 + 在线状态可观测性；
- 三态监督；
- 固定 `gamma` 证据衰减；
- Camera Crash 与 Frame Lost。

## 阶段 3：来源感知

- 来源向量；
- 新颖度；
- `N_eff`；
- 相关性折扣；
- Keep / Recover / Defer。

## 推荐实验目录

```text
outputs/exp_001_baseline
outputs/exp_002_camera_crash
outputs/exp_003_ternary_objectness
outputs/exp_004_evidence_decay
outputs/exp_005_full_ledger
```

## 消融配置与开关

已提供两个不会让未训练分支参与推理的独立配置：

```text
configs/evidence_conserving/ablation_ternary_only.py
configs/evidence_conserving/ablation_decay_only.py
```

- `ablation_ternary_only.py`：保留三态监督和可观测性条件背景，使用官方 StreamPETR 记忆写入，不做证据分数校准。
- `ablation_decay_only.py`：用原始类别置信度生成存在/不存在证据，关闭三态损失与可观测性条件负样本，保留来源账本和固定衰减。

完整模型还支持以下配置字段：

```text
enable_observability_conditioning
enable_evidence_memory
evidence_probability_source = ternary | classification
calibrate_detection_scores
background_observability_floor
evidence_warmup_steps
```

单项覆盖示例：

```bash
python tools/train.py --config configs/evidence_conserving/mini_debug.py -- \
  --cfg-options \
  model.pts_bbox_head.evidence_warmup_steps=0 \
  model.pts_bbox_head.calibrate_detection_scores=False
```

## 指标

标准 nuScenes 指标由官方 evaluator 计算。附加指标位于 `evaluation/metrics.py`：

- Unsupported False Positive Rate；
- Stale Object Persistence；
- Evidence Inflation Ratio；
- Reacquisition Delay；
- Risk–Coverage。

正式实验应保存随机种子、配置副本、日志、checkpoint、预测和协议 JSON。
