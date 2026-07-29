# S2.4 关闭路径与语义隔离审计

## 结论

本次审计未通过“关闭 S2.4 与 `s2.2-stable` 逐 tensor 一致”的硬门。
`9958366` 已经无条件计算固定相机相关矩阵的 `N_eff`，并把它加入证据更新。
所以旧脚手架已经影响 S2.2 输出；新关闭路径是真正旁路，但不能复现这个
已受影响的锚点。

按停止条件，本分支没有实现新的相关矩阵、损失或性能候选，也没有运行
S2.4 smoke、训练或性能筛选。

## 逐 tensor 对照

历史逻辑被显式标记为 `legacy=True` 后，在四个协议上均逐 tensor 复现现有
S2.2 轨迹：

- final prediction；
- alpha / beta；
- actual positive / negative evidence；
- source evidence；
- action / write mask；
- conservation residual；
- source mass residual。

上述 `s2_2_stable_vs_legacy` 的 40 个协议/字段组合全部
`max_abs_diff=0`。这同时排除了本次开关重构或只读 hash 诊断改变历史预测
的可能。

真正关闭路径与 S2.2 的差异如下：

| 协议 | final prediction max abs diff | alpha | beta | source evidence |
|---|---:|---:|---:|---:|
| Clean | 99.591335 | 5.416232 | 8.463931 | 14.323524 |
| Crash5 | 101.155788 | 5.139517 | 8.945536 | 14.089731 |
| Crash10 | 101.088726 | 5.463307 | 8.999600 | 15.174771 |
| Compound | 98.818375 | 4.803109 | 9.008134 | 15.126703 |

四协议的 classification、box、decoder、propagated query 和 temporal
memory hash 均出现不一致；action、previous action、write mask 和 Top-K
也不一致。所有数值字段中的总体 `max_abs_diff=897`，来自 Top-K query
index；连续预测张量最大差为 `101.15578842163086`。

指标变化也与逐 tensor 分叉一致：

| 协议 | S2.2/legacy mAP | disabled mAP | S2.2/legacy NDS | disabled NDS |
|---|---:|---:|---:|---:|
| Clean | .4248 | .4275 | .4770 | .4797 |
| Crash5 | .4183 | .4192 | .4730 | .4729 |
| Crash10 | .4110 | .4122 | .4707 | .4714 |
| Compound | .3924 | .3910 | .4573 | .4566 |

这些指标只用于验证隔离，不构成性能筛选。

完整数据见 `prediction_invariance.csv` 和 `evidence_invariants.csv`。

## 守恒与 source mass

legacy 与 disabled 的四协议均满足：

- conservation violation count：0；
- unsupported growth count：0；
- source mass violation count：0；
- conservation residual abs max：不超过 `2.861023e-6`；
- source mass residual abs max：不超过 `2.861023e-6`。

关闭路径改变实际加入的证据，但守恒公式使用同一实际证据，因此不破坏
守恒；source ledger 也继续守恒。两条路径之间 residual 的逐 tensor 值并
不要求相等，因为其前态与加证据量已经不同。

## Checkpoint 与状态安全

S2.2 checkpoint 大小为 159,783,623 bytes，共 629 个 state-dict key。
审计结果：

- 旧 checkpoint 在 legacy/disabled 两条路径各完成四协议加载；
- correlation 配置 key：1；
- 新开关 state key：0；
- 场景运行态 key：0；
- S2.4 新运行态：0。

矩阵保留为 persistent 配置 buffer 是为了旧 checkpoint 严格兼容；真正的
ledger scene/batch/query 状态仍为 non-persistent。详见
`checkpoint_audit.csv`。

## 测试与运行安全

全量命令：

```text
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
```

首次运行发现 PyTorch 1.9 CPU Half 的 `clamp_min` 不支持，修复为仅在 CPU
FP16 内部用 FP32 计算并恢复输出 dtype。最终全量结果为：

```text
95 passed, 7 warnings in 6.61s
```

测试覆盖真正旁路、历史公式复现、旧 checkpoint 严格加载、运行态不进入
checkpoint、scene reset、batch/query/Top-K 变化，以及 CPU/GPU FP16。

全部全量 pytest、四协议推理和汇总均通过独立 tmux 会话运行，退出码均为
0；没有重复启动会话。日志与 metadata：

- `outputs/stage2/s2_4_isolation_audit/pytest_rerun1/`
- `outputs/stage2/s2_4_isolation_audit/inference_tmux/`
- `outputs/stage2/s2_4_isolation_audit/analysis_rerun1/`

## 是否允许下一步

不允许立即进入固定相关矩阵 smoke 或实验。原因不是关闭路径实现失败，而是
稳定锚点的语义已包含固定相关矩阵，无法同时满足“真正关闭”和“与该锚点
逐 tensor 相同”。

进入下一步前必须先做一个明确、可审计的基线决策：

1. 从 S2.2 代码建立真正无折扣的 canonical baseline，并重新完成其稳定性
   验收；或
2. 明确承认 `s2.2-stable` 已经是固定 correlation discount 实现，并重新
   定义 S2.4 的实验变量。

在该决策前继续固定矩阵实验无法形成有效的 S2.2 对照。
