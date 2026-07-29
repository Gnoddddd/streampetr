# S2.4 Canonical Baseline Disambiguation

## 实验结论

在冻结的单 seed、50 iter 公平对照中，固定相关矩阵 C1 没有改善整体性能。
C1 相对无折扣 C0：

- Clean：mAP `-0.002721`，NDS `-0.002518`；
- 三个故障协议平均：mAP `-0.000914`，NDS `-0.000447`；
- 仅 Compound 有极小正增量：mAP `+0.000805`，NDS `+0.000093`。

因此当前证据不支持保留 legacy fixed discount 作为性能方法，更不支持在此
结果上直接进入动态相关矩阵。下一阶段应采用
`canonical_no_discount`；如未来重新研究相关性，必须作为新的、重新定义的
方法从无折扣基线出发，而不是把 legacy 行为当作 S2.2 baseline。

## 公平性与初始化

C0、C1使用完全相同的：

- Stage1正式 `iter_200.pth` 初始化，SHA256
  `9f8f4ab9361bb3a880abbdb605f93929886aa242d1d665ea84500a7abd331a16`；
- seed `2026`、数据配置和数据顺序；
- optimizer、LR、batch size；
- dynamic FP16与 `max_norm=35` 梯度裁剪；
- 50 iter runner和冻结参数集合。

没有使用已经适应 legacy 路径的 S2.2 `iter_50.pth` 做正式训练对照。选择
Stage1 checkpoint 是因为它是现有的、早于 S2.1/S2.2账本训练的共同初始化
点。两组第1 iter的 loss `22.3750`、grad norm `10.5902`完全一致，说明
初始化、数据起点与训练条件对齐；后续分叉来自两条运行路径。

## 正式四协议结果

| 候选 | 协议 | mAP | NDS | C1-C0 mAP | C1-C0 NDS |
|---|---|---:|---:|---:|---:|
| C0 | Clean | .427493 | .479548 | — | — |
| C0 | Crash5 | .419369 | .473273 | — | — |
| C0 | Crash10 | .413468 | .471860 | — | — |
| C0 | Compound | .391590 | .457227 | — | — |
| C0 | 故障平均 | .408142 | .467454 | — | — |
| C1 | Clean | .424772 | .477030 | -.002721 | -.002518 |
| C1 | Crash5 | .418311 | .473028 | -.001057 | -.000245 |
| C1 | Crash10 | .410979 | .470670 | -.002489 | -.001191 |
| C1 | Compound | .392395 | .457320 | +.000805 | +.000093 |
| C1 | 故障平均 | .407228 | .467006 | -.000914 | -.000447 |

这些是单 seed、50 iter筛查结果，不外推为多seed结论。

## N_eff 与行为分布

C0真正旁路矩阵和 N_eff；用于诊断的 effective count恒为1。C1执行历史
固定矩阵公式：

| 协议 | C1 N_eff mean | median | p95 | max | zero ratio |
|---|---:|---:|---:|---:|---:|
| Clean | 1.070322 | .999992 | 1.481143 | 1.481732 | .000535 |
| Crash5 | 1.043459 | .999991 | 1.481135 | 1.481716 | .020878 |
| Crash10 | 1.016852 | .999989 | 1.481129 | 1.481719 | .040562 |
| Compound | 1.015903 | .999992 | 1.481034 | 1.481745 | .040658 |

当前公式相对C0的单位因子并非纯粹“折扣”：在多相机支持query上，
`N_eff`最高约1.482，会放大实际加入的证据；无支持时则为0。它使四协议
汇总 write ratio 从C0的 `.284582`升到C1的 `.393745`，keep ratio从
`.277435`升到`.386619`。recover ratio基本不变（C0 `.007147`、C1
`.007126`）。更多写入和keep没有转化为更好的Clean或故障平均性能。

完整分位数及逐协议action/write统计见 `neff_summary.csv` 和
`per_protocol_metrics.csv`。

## 训练稳定性与工程门

| 项目 | C0 | C1 |
|---|---:|---:|
| iter | 50 | 50 |
| first loss | 22.3750 | 22.3750 |
| final loss | 14.0058 | 13.4379 |
| max loss | 22.3750 | 22.3750 |
| max grad norm | 16.2765 | 16.2540 |
| peak logged GPU memory | 555 MB | 555 MB |
| train conservation residual abs max | 1.907349e-6 | 1.907349e-6 |
| train conservation violations | 0 | 0 |
| train unsupported growth | 0 | 0 |
| train source-mass violations | 0 | 0 |

两组2 iter smoke、Clean zero-shot、50 iter训练及四协议评测均退出0。没有
NaN、Inf、OOM或RuntimeError。评测阶段：

- conservation residual abs max不超过 `2.861023e-6`；
- conservation violation count为0；
- unsupported growth count为0；
- source-mass violation count为0。

全量测试结果为 `95 passed, 7 warnings in 6.73s`。

## 预测与历史语义

C1 50 iter在四协议上与现有S2.2正式预测逐tensor完全一致，
`max_abs_diff=0`，并精确复现：

- Clean `.424772/.477030`；
- Crash5 `.418311/.473028`；
- Crash10 `.410979/.470670`；
- Compound `.392395/.457320`。

因此 `s2.2-stable`不能定义为无折扣版本；它是**已含 fixed correlation
行为**的历史稳定锚点。C0与C1的classification、box、propagated query、
temporal memory和最终预测均发生分叉，最终预测最大绝对差
`102.39833068847656`。

## 方法定义回答

1. **fixed correlation是否真正改善性能？**  
   否。它降低Clean和故障平均，只在Compound产生不足千分之一的局部增量。

2. **当前s2.2-stable应如何定义？**  
   定义为“已含历史 fixed correlation/N_eff 行为”的稳定版本，不能称为
   canonical no-discount。

3. **S2.4还能否把fixed correlation作为新方法？**  
   不能。该行为已经存在于S2.2，且C1逐tensor复现S2.2；它不是相对S2.2的
   新增方法。

4. **下一步方向？**  
   采用C0 canonical no-discount作为清晰基线；不保留legacy作为性能方法，
   当前也不进入动态矩阵。动态相关性若以后重启，应重新预注册并相对C0
   证明增量，不应沿用被混淆的S2.2方法命名。

## 抗断线记录

全量pytest、smoke、zero-shot、50 iter训练、八项评测和汇总均由
`scripts/run_experiment_tmux.sh`启动。关键日志：

- `outputs/stage2/s2_4_baseline_disambiguation/pytest/`
- `outputs/stage2/s2_4_baseline_disambiguation/c0_smoke2/`
- `outputs/stage2/s2_4_baseline_disambiguation/c1_smoke2/`
- `outputs/stage2/s2_4_baseline_disambiguation/zero_shot/`
- `outputs/stage2/s2_4_baseline_disambiguation/c0_50/`
- `outputs/stage2/s2_4_baseline_disambiguation/c1_50/`
- `outputs/stage2/s2_4_baseline_disambiguation/eval_tmux/`
- `outputs/stage2/s2_4_baseline_disambiguation/analysis/`

所有对应 metadata 的exit status均为0，没有重复启动实验。
