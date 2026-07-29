# S2.3-R2 Confirmed Reacquisition 前置诊断

## 结论

现有恢复延迟改善并不等于可靠的对象重获。主要问题是**确认条件不足**，其次是
错误触发后的 **memory write 缺少隔离**；候选选择本身也需要GT/类别一致性约束，
但仅替换候选排序不足以解决已观察到的错误写回。

## 必答问题

- 全部实际bonus触发的GT匹配精度为 **44/46 =
  95.65%**。
- B4 zero-shot 的23次故障bonus触发中，正确恢复 **22/23**；
  相对B0真正新增GT匹配 **0/23**。
- 全部被判为错误恢复的触发为 **240**，其中 **162**
  次实际进入memory；有可观测后续污染的为 **76** 次。
- 相对B0最明显的协议级mAP下降是 **b6_50iter/compound**
  （ΔmAP=-0.0039）。GT计数最差区间是
  **b4_50iter/camera_crash_10/
  post_fault**（Δmatched GT=
  -4）。
- source gate 几乎不触发的原因：source recovery 使用当前source向量与故障前累计source evidence做余弦相似度；相机失效/恢复会改变稀疏相机支持集合，而锚点还受source decay和Top-K重排影响，导致相似度经常为0或极低。它再与first-recovery、可靠性、motion、pre-gap presence及age相乘，使组合gate覆盖率近乎归零。

## 七项假设判断

1. **query与GT对齐错误：成立。** `gt_alignment_summary.csv` 中GT unmatched和
   class-wrong触发直接证明恢复事件并非稳定对象身份确认。
2. **类别或box不准确：成立。** `center_distance`、`class_correct`和
   `velocity_error`将空间对齐与类别错误分开统计。
3. **恢复时机过早：部分成立。** 首次可靠帧即触发，但1/3/5帧持续性不足的案例
   表明单帧确认过早。
4. **错误query写入memory：成立。** 见 `memory_write_summary.csv`。
5. **bonus覆盖率过低：成立。** B4仅23次；source-gated变体的乘法gate进一步
   将覆盖率压到接近0。
6. **source gate语义/阈值不合理：成立。** 当前余弦锚点衡量“相机组合相似”，
   并不能确认“同一GT对象”，且恢复时相机组合变化会错误拒绝。
7. **恢复窗改善但其他区间退化：成立。** `recovery_window_metrics.csv` 与
   `per_protocol_metrics.csv` 显示恢复延迟收益没有覆盖active-fault/late-post
   的GT损失。

## 下一方法方向

下一阶段应优先解决**确认条件**：把单帧bonus触发拆成候选与确认两阶段，要求
短时多帧类别、运动和空间一致后才确认。确认前必须禁止写入正式temporal
memory，或写入隔离的pending区；因此memory write是与确认条件绑定的第二优先级。
候选选择可保留motion召回，但不能直接授权bonus或正式写回。

这些证据足以支持实现 **Two-Phase Confirmed Reacquisition** 的设计论证；
尚不支持直接扩大bonus、放宽source阈值或进行200 iter训练。

## 范围与不变量

本报告只使用clean、camera_crash_5、camera_crash_10、compound和既有公开
w2_t100结果；未读取holdout，未增加seed，未训练新候选，未启用teacher，
未开始S2.4。`prediction_invariance.csv` 要求20/20逐tensor/逐字段完全一致。
