# Evidence3D S2.3 性能救援最终决定

## 决定

Evidence-Budgeted Reacquisition 的工程实现通过，但性能救援未通过。停止在
50 iter 门限，不进入 200 iter、额外随机种子、holdout 或 S2.4。

## 根因与方法

旧 N1 把基础正证据整体乘小：Clean 正证据约为 S2.2 的 0.608，导致当前帧
KEEP/写回减少，再由 memory 传播放大。新方法先逐 tensor 计算 S2.2
`base_positive_evidence` 和 `base_negative_evidence`，只在同一 gap 的首次可靠
恢复帧增加：

`bonus = min(restore_ratio * lost_strength,
max_relative_bonus * base_positive, max_absolute_bonus)
* reacquisition_gate`

最终正证据为 `base_positive + bonus`，负证据严格为 `base_negative`。score、
KeepRecoverDeferPolicy 和 write mask 没有被直接修改。

## 状态生命周期与隔离

七个 gap 状态均为 `persistent=False` buffer，随 device 迁移，runtime state
格式升级到 v4，可显式 export/import。scene、batch、query reset 和无效写回会
清空状态；Top-K 提交保持对齐。B4/B6 checkpoint 各 630 个 model key，未发现
任何 ledger runtime key，也未发现非有限 tensor。

## 测试与数值安全

全套 124 项测试通过，其中 38 项为预算重获专用契约。CPU FP16、旧 N0-N6
默认语义、S2.2 tensor 一致性、runtime/checkpoint 隔离、motion/source gate、
单次 bonus 和 curriculum 均有覆盖。28 组零样本和 8 组 debug50 评测的
conservation residual 最大值不超过 `3.814697e-6`；conservation violation、
unsupported growth 和 source-mass violation 均为 0。

## 零样本筛选

B1/B2/B3/B5 与 B0 指标一致且没有有效 bonus。B4 的 Clean 完全一致，
Crash10 NDS 提升 0.0006，Compound NDS 下降 0.0003，w2_t100 与 B0 持平。
B6 的 Clean NDS 下降 0.0006，Crash10 提升 0.0010，但 Compound 下降 0.0022；
w2_t100 均值从 7.83 改善到 6.50。选择 B4 与 B6 进入 50 iter。

## 50 iter

两次训练都完成 50/50，无 NaN、Inf、OOM 或 RuntimeError，峰值显存日志为
555 MiB。B4 末步 loss/grad_norm 为 13.6529/14.8412，B6 为
14.4070/14.9053。两者 Clean 都未下降，恢复延迟也改善；但 B4 的故障平均
mAP/NDS 为 0.406800/0.466133，B6 为 0.405833/0.465467，均低于 B0 的
0.407233/0.467000。因此违反 50iter 硬门。

## 风险与未执行项目

- source recovery gate 在真实轨迹中过严，B1/B2/B3/B5 的有效 bonus 为 0。
- motion-only B4 的触发极稀疏，并出现 Crash10 改善而 Compound 退化。
- 50 iter 更新 ternary/observability 后能改善严格恢复延迟，但不能保护故障平均检测。
- 没有候选满足教师分支前置条件，教师一致性未启用。
- 200 iter、三随机种子和 holdout 按门控规则未运行；holdout 保持锁定。
- 当前结果不满足进入 S2.4 的条件。
