# S2.4 方法规格：相关性折扣

状态：仅预注册计划；未实现、未训练。

## 文档依据与正式定义

项目总纲 `docs/Stage2_Codex实验总纲.md` 的 4.5、8 和 17 节对 S2.4
给出唯一一致定义：

- 正式名称：**S2.4 相机相关性折扣（Correlation Discount）**。
- 第一版使用固定相机相关矩阵；固定版通过后才允许考虑动态矩阵。
- 有效独立证据数：

  ```text
  N_eff = (sum(w))^2 / (w^T R w + eps)
  ```

- `w` 为当前每相机来源支持或来源强度，`R` 为 `C×C` 相机相关矩阵。
- `N_eff` 必须限制在 `[0,C]`。
- 重复同一来源不能产生线性证据增长；多个低相关相机共同支持时仍应
  增强证据。

本计划不加入文档之外的新网络、teacher、GT runtime gate 或 S2.3
reacquisition 机制。

## 目标、输入和输出

目标是防止来自同一或高度重叠相机视角的证据被错误计成多份独立证据，
同时保留低相关多视角的一致支持。

输入：

- S2.2 的每查询六相机 `current_source` / `source_evidence`
- 固定、对称、对角线为 1 的六相机相关矩阵 `R`
- 当前 raw positive/negative evidence、observability
- S2.2 的 alpha/beta 前态及 source ledger 状态

输出：

- `effective_count` / `N_eff`
- 明确可诊断的 `correlation_discount`
- 经相同折扣后的 actual positive/negative added evidence
- alpha、beta、strength、conservation residual
- source mass residual、action 与 write mask（首轮不得由相关性模块改变
  policy 语义，除非后续单独预注册）

Beta 更新保持总纲公式：

```text
e_pos = raw_present_evidence
        * observability
        * novelty
        * correlation_discount
e_neg = raw_absent_evidence
        * observability
        * novelty
        * correlation_discount
```

直接从 S2.2 开始时不启用失败的 S2.3 active novelty/reacquisition；
novelty 保持 S2.2 锚点语义，唯一核心变量是相关性折扣。

## 修改边界

未来实现任务最多触及：

- `models/observability_head.py`：固定 `R` 的验证与 `N_eff` 计算
- `models/evidence_ledger.py`：将折扣后的实际增量纳入账本和守恒诊断
- `models/streampetr_adapter.py`：配置、张量接线和 diagnostics
- `evaluation/evidence_single_gpu_test.py`：向后兼容导出
- 新的 S2.4 配置和 `tests/test_correlation_discount.py`

不得修改 `repos/StreamPETR`。

## 与现有代码的关系及启动前置条件

现有 `GeometricObservabilityHead` 已包含固定相关矩阵和同形
`effective_count` 计算，`TemporalEvidenceUpdate` 也会使用
`effective_count`。这属于历史脚手架，并不自动构成一个可归因的 S2.4
实验：S2.2 稳定配置没有独立的相关性开关，因而当前无法证明“off”与
“fixed R”只相差一个变量。

正式实现前必须先完成一个不训练的语义审计：

1. 建立显式 `enable_correlation_discount=False` 的 S2.2 对照。
2. 证明关闭路径与 `s2.2-stable` 逐 tensor 一致。
3. 固定版与必要消融除 `R` 的 off-diagonal 折扣外完全一致。
4. 确认 conservation expected evidence 使用的正负增量与实际加入量相同。

在该审计通过前，项目**尚不具备启动 S2.4 smoke/训练的条件**。

## 是否依赖 S2.3

方法数学上不依赖 S2.3 成功：`N_eff` 只依赖来源权重和 `R`，可以直接
从 S2.2 稳定锚点开始。总纲给出的乘法公式包含 novelty，但这不要求采用
已经失败的 S2.3 active 候选；从 S2.2 开始可保持其 novelty baseline
不变。

因此：

- 不继承 S2.3-R2 分支作为方法基线。
- 未来分支应从 `s2.2-stable` 创建。
- S2.3 的 Case C 结果保持冻结，不回头救援。

## 候选限制

- 主候选：固定先验相关矩阵 `R_fixed`。
- 唯一必要消融：`R_identity`，保留相同公式但将非对角相关性置零，
  用于隔离跨相机相关折扣的贡献。
- 动态相关矩阵不属于首轮候选；只有固定版通过全部进入门后，才能在
  新任务中重新预注册。
