# S2.4 Correlation Discount 脚手架清单

## 审计基线

- 稳定锚点：`s2.2-stable` / `9958366`
- 审计分支：`stage2/s2.4-correlation-discount-audit`
- 初始化 checkpoint：
  `outputs/stage2/s2_2_source_ledger_debug_50/iter_50.pth`
- 协议：Clean、Camera Crash 5f、Camera Crash 10f、Compound 10f

## 历史脚手架

`models/observability_head.py` 在 `9958366` 中已经包含并无条件执行以下
逻辑：

1. `default_camera_correlation(6)` 构造固定相机相关矩阵；
2. 对角线为 `1`；
3. 相邻相机对为 `0.35`；
4. FRONT_RIGHT/FRONT_LEFT 与 BACK_LEFT/BACK_RIGHT 两对为 `0.15`；
5. 其余非对角元素为 `0`。

对每个 query 的相机支持权重 `w`，历史公式为：

```text
numerator    = (sum_i w_i)^2
diagonal     = sum_i w_i^2
pairwise     = sum_i sum_j w_i R_ij w_j
off_diagonal = max(pairwise - diagonal, 0)
N_eff        = numerator / (diagonal + off_diagonal + eps)
```

`N_eff` 被限制到 `[0, num_cameras]`；无相机支持时为 `0`。矩阵以
`camera_correlation`、`persistent=True` 注册，因此是 checkpoint 中的
模型配置参数，不是场景运行态。

## 配置与默认值

本次新增统一开关：

```python
enable_correlation_discount = False
```

- adapter 和 observability head 的默认值均为 `False`；
- adapter 强制把统一值传入 head，避免嵌套配置出现两个相互矛盾的值；
- `False` 时不读取相关矩阵、不执行 pairwise `einsum`、不计算 `N_eff`，
  并向 temporal update 传 `None`；
- temporal update 收到 `None` 时走原始无折扣路径，内部
  `effective_count=1`；
- 关闭路径返回的全一 `effective_count` 仅用于诊断展示，不参与证据更新；
- `True` 明确复现 `9958366` 的历史隐式路径。

审计配置：

- `configs/evidence_conserving/mini_stage2_correlation_discount_disabled.py`
- `configs/evidence_conserving/mini_stage2_correlation_discount_legacy.py`

## 完整调用链与影响

```text
config
  -> StreamPETREvidenceHead.enable_correlation_discount
  -> GeometricObservabilityHead
  -> per-camera support weights
  -> [enabled only] fixed correlation matrix + N_eff
  -> EvidenceLedger.update_queries
  -> EvidenceConservingTemporalUpdate
  -> actual positive/negative evidence
  -> alpha/beta and source evidence
  -> action / write mask / ranking score
  -> Top-K selection
  -> temporal memory write
  -> propagated queries on later frames
  -> classification / boxes / final prediction
```

因此历史脚手架不是旁路诊断：它直接缩放正负证据，继而影响 source
evidence、action、write mask、Top-K、memory、propagated query 和后续预测。
observability 与相机 source vector 的几何计算本身不依赖开关。

## 状态与兼容性

- `EvidenceLedger._STATE_NAMES` 全部为 `persistent=False`，不进入普通
  `state_dict` 或 checkpoint。
- 新开关是普通 Python 布尔值，不进入 `state_dict`。
- `camera_correlation` 保持 persistent，以便旧 checkpoint 严格加载；
  它是固定配置，不含 batch/query/scene 状态。
- 旧 checkpoint 的 629 个 state-dict key 中有 1 个 correlation 配置 key，
  运行态 key 为 0，开关 key 为 0。
- 同一旧 checkpoint 已分别完成 4 次 legacy 和 4 次 disabled 推理加载。

## 可复用边界

可以复用固定矩阵构造、矩阵形状校验、显式 enabled 分支中的 `N_eff`
公式、旧 checkpoint 的矩阵配置键，以及 temporal update 的
`effective_count=None` 无折扣接口。

当前不能据此开始固定矩阵实验：`s2.2-stable` 本身已经使用该固定矩阵，
真正关闭后与稳定锚点不等价。必须先决定无折扣 canonical baseline，或明确
把 `9958366` 重新定义为已启用固定相关性折扣的版本。
