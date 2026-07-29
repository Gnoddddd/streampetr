# S2.2 稳定锚点

## Git 锚点

- 准确可验证的仓库提交：
  `995836632255c637f2c89137bc868853f3d8a042`
- 稳定标签：`s2.2-stable`
- 提交标题：`baseline: preserve Evidence3D through S2.3 evaluation`

该仓库的可达历史从此初始快照开始，没有更早的 S2.2-only commit。
因此不能声称存在一个更早的纯 S2.2 提交。`9958366` 是唯一能够由 Git
历史、验收快照和后续 B0 报告共同验证的 S2.2 可执行锚点；后续方法必须
用 S2.2 配置并明确关闭 S2.3 模块。

## 验收证据

本地冻结验收记录：

```text
outputs/stage2_audit/s2_2_passed_20260728_200220/S2_2_ACCEPTANCE.txt
outputs/stage2_audit/s2_2_passed_20260728_200220/files.sha256
outputs/stage2/s2_2_source_ledger_debug_50/metrics/source_mass_conservation_summary.csv
```

这些 outputs 仅作为只读证据，不纳入 Git。

验收结果：

- 67 tests passed。
- 50/50 iter 完成，无 NaN、Inf、OOM、RuntimeError。
- source mass residual abs max `1.9073486e-6`。
- source mass violation ratio `0`。
- conservation violation ratio `0`。
- unsupported growth ratio `0`。
- checkpoint runtime source-ledger key hits `0`。
- source tracking on/off：results SHA256、action、score scale、write mask
  一致，prediction tensor max abs diff `0`。

协议指标：

| 协议 | mAP | NDS |
|---|---:|---:|
| Clean | 0.4248 | 0.4770 |
| Camera Crash 5f | 0.4183 | 0.4730 |
| Camera Crash 10f | 0.4110 | 0.4707 |
| Compound 10f | 0.3924 | 0.4573 |

原始 evaluator 数值分别为：

- Clean `0.4247724932 / 0.4770300703`
- Crash 5f `0.4183114887 / 0.4730280297`
- Crash 10f `0.4109787293 / 0.4706696455`
- Compound `0.3923952449 / 0.4573203851`

提交 `9958366` 中的三个 S2.2 配置、source-ledger 汇总脚本和三个核心
测试文件与验收快照的 SHA256 完全一致。后续 S2.3/R2 报告也把同一组
指标明确记录为 S2.2/B0。

## 使用规则

未来 S2.4 分支从 `s2.2-stable` 创建，不从 S2.3-R2 负结果分支继承。
checkpoint 仍使用已验收的 S2.2 50-iter checkpoint，但 checkpoint 和
outputs 不提交 Git。正式实现前先解决现有 `effective_count` 相关性脚手架
缺少显式 off-path 的语义隔离问题。
