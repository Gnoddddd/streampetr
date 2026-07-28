# Evidence3D 架构说明

## 数据流

```text
六相机图像
  -> ApplyPartialObservation
     -> camera_online_mask
     -> camera_quality
     -> camera_fresh_mask
  -> StreamPETR R50 + CPFPN
  -> 时序 Transformer 查询
  -> 3D 框与类别分支
  -> 三态目标性分支
  -> 三维查询中心投影到六相机
  -> 可观测性 / 来源向量 / N_eff
  -> Beta 证据守恒更新
  -> Keep / Recover / Defer
  -> 受控记忆写入与分数校准
```

## 与官方 StreamPETR 的最小侵入式接入

官方源码位于 `repos/StreamPETR`，不直接修改。`evidence3d_plugin` 先导入官方插件完成注册，再注册：

- `ApplyPartialObservation`；
- `EvidenceNuScenesDataset`；
- `EvidenceConservingStreamPETRHead`；
- `EvidenceTraceHook`。

自定义 head 仅覆盖两个关键机制：

1. `loss()`：Hungarian 未匹配查询的背景权重与三态目标；
2. 时序记忆写入：Top-K 排序、Beta 更新、来源账本和 Defer 门控。

Transformer、Hungarian assigner、框编码器、nuScenes evaluator 和官方时序记忆结构保持不变。

## 查询顺序

固定版 StreamPETR 的解码查询顺序为：

```text
[DN queries] + [base object queries] + [propagated memory queries]
```

证据更新先移除 DN 查询。只有传播查询从上一帧账本读取 Beta 先验；新基础查询从 `Beta(1,1)` 开始。Top-K 写回后的账本顺序与 StreamPETR memory bank 保持一致。

## 守恒不变量

令有效证据强度：

```text
S = alpha + beta - 2
```

当 `O * novelty * N_eff = 0` 时：

```text
S_t = gamma * S_{t-1}
```

代码返回：

- `no_new_evidence`；
- `conservation_ratio`；
- `conservation_violation`。

测试要求无新观测时 `conservation_ratio == gamma` 且 `conservation_violation == 0`。

## 三态监督与原始类别分支

三态分支并不替代十类 nuScenes 分类分支：

- 十类分支负责具体类别；
- 三态分支负责存在、缺失、不可观测；
- 可观测性只改变未匹配查询是否可作为可靠背景；
- 正查询仍接受原有类别与框回归损失。

## 证据来源与相关性

来源向量是每个查询在六相机上的归一化几何支持。默认相关矩阵对相邻相机给予适度相关性，对相对视角近似独立。`N_eff` 使用相关性折扣，避免重叠视角简单计为六份独立证据。

该矩阵可通过 `observability_cfg.correlation_matrix` 替换为经验估计或学习值。
