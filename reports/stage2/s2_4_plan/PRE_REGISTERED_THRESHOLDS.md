# S2.4 预注册门限

状态：仅计划；未运行任何 S2.4 实验。

基线固定为 `s2.2-stable`：

| 协议 | mAP | NDS |
|---|---:|---:|
| Clean | 0.4248 | 0.4770 |
| Camera Crash 5f | 0.4183 | 0.4730 |
| Camera Crash 10f | 0.4110 | 0.4707 |
| Compound 10f | 0.3924 | 0.4573 |

## 工程硬门

- S2.4 disabled 对 S2.2 的 classification、box、最终预测、alpha/beta、
  added evidence、source、action、write mask、memory/Top-K 必须逐 tensor
  一致，`max_abs_diff=0`。
- 普通 `state_dict` 和 checkpoint 不得出现新的场景运行态。
- `N_eff` 有限且在 `[0,6]`。
- 单相机支持的 `N_eff` 接近 1；完全重复的同源证据不得线性增长；
  两个低相关相机共同支持的 `N_eff` 必须高于单相机。
- conservation residual absolute max `<=1e-5`，violation count `=0`。
- source-mass residual absolute max `<=1e-5`，violation count `=0`。
- unsupported growth count `=0`。
- CPU FP16、GPU FP16、scene/reset、batch/query/Top-K 与 checkpoint 测试
  全部通过。
- 无 NaN、Inf、OOM、RuntimeError 或状态泄漏。

## 性能硬门

采用总纲 3.1 的 mini Go/No-Go 标准，不以单协议偶然提升代替整体通过：

- Clean mAP 与 NDS 相对 S2.2 各下降不超过 `0.003`。
- Crash 5f mAP/NDS 均不低于 S2.2。
- Crash 10f mAP/NDS 均不低于 S2.2。
- Compound 10f mAP/NDS 均不低于 S2.2。
- public 100% recovery mean delay `<=5.0` 帧，max delay `<=7` 帧。

## 分阶段进入条件

1. **Smoke**：只有语义审计、全量单元测试和 disabled-path prediction
   invariance 全部通过后，才允许主候选与唯一消融各运行最小 smoke。
2. **50 iter**：对应 smoke 有限、checkpoint 安全且所有工程硬门通过后
   才允许；最多主候选和一个必要消融。
3. **200 iter**：50 iter 必须同时通过全部工程门和上述 Clean/三个故障
   协议性能门；只允许晋级候选运行，失败候选停止。
4. **多 seed**：200 iter 在冻结 dev 协议上仍通过全部门限，配置和方法
   参数冻结后才允许；不得以多 seed 搜索阈值。
5. **Holdout**：仅在多 seed 结果通过、候选选择和全部门限已用 dev 数据
   冻结、且得到新的明确授权后一次性使用。不得用 holdout 调参、回选
   `R` 或选择候选。

任何阶段失败即停止，不得自动放宽阈值。
