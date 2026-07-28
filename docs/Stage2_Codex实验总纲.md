# Evidence3D Stage2：来源感知证据守恒时序检测实验总纲

## 0.文档用途

本文档用于指导Codex继续开发Evidence3D Stage2，目标是在当前StreamPETR+nuScenes-mini工程中，实现完整的来源感知证据账本，并形成可以扩展至完整nuScenes和高水平会议实验的代码、配置、评测及消融体系。

本阶段不是简单增加几个损失项，而是要围绕以下核心科学问题建立完整证据链：

> 当多相机观测退化、缺失或相互矛盾时，时序3D检测器应当何时保留当前判断、何时依赖历史证据恢复、何时拒绝继续向时序记忆写入不可靠信息？

最终论文方法应由四部分共同构成：

1. 三维可观测性与三态目标监督；
2. 来源感知证据账本；
3. 证据新颖度与相机相关性折扣；
4. 证据守恒驱动的Keep/Recover/Defer时序门控。

不能保证仅凭当前工程即可发表CCF-A会议，但以下路线能够使方法、实验和论证逐步达到高水平论文所要求的完整度。

---

# 1.Stage1已完成状态

## 1.1工程基础

项目根目录：

```text
~/research/evidence3d
```

第三方代码：

```text
repos/StreamPETR
```

StreamPETR固定提交：

```text
95f64702306ccdb7a78889578b2a55b5deb35b2a
```

开发环境：

```text
WSL2 Ubuntu 22.04
Python 3.8.20
torch 1.9.0+cu111
mmcv 1.6.0
mmdet 2.28.2
mmseg 0.30.0
mmdet3d 1.0.0rc6
```

项目约束：

```text
第三方源码放repos/StreamPETR
Evidence3D方法代码放项目根目录models、datasets、hooks、evaluation等目录
不直接大规模重写第三方StreamPETR
通过Adapter、配置继承和小型补丁接入方法
```

StreamPETR本身通过目标查询逐帧传播历史信息，是当前Evidence3D账本和门控机制的合理承载基础。citeturn969953search0

## 1.2当前主要代码结构

```text
models/
├── observability_head.py
├── ternary_objectness.py
├── evidence_ledger.py
├── keep_recover_defer.py
├── temporal_update.py
└── streampetr_adapter.py

datasets/
└── corruption.py

hooks/
└── evidence_trace_hook.py

evaluation/
└── evidence_single_gpu_test.py

protocols/
├── presets/
└── ...

scripts/
├── eval_gt_recovery_predictions_*.sh
├── analyze_gt_recovery_delay_*.py
├── summarize_robust_gt_recovery_delay_*.py
├── summarize_stage1_r50_trace_phases.py
└── summarize_stage1_r50_trace_by_frame.py
```

已有账本状态包括：

```text
alpha
beta
provenance
age
effective_count
observability
novelty
action
```

已有三态动作编码：

```text
KEEP=0
RECOVER=1
DEFER=2
```

已有策略输出：

```text
keep_count
recover_count
defer_count
```

## 1.3Stage1模型

```text
B0：official_r50_900q_baseline
B1：classification_active_r50_900q
T1：stage1_active_r50_900q
```

其中：

- B0是官方StreamPETR参照；
- B1是普通二分类或分类式辅助分支；
- T1是三态Present/Absent/Unobserved监督模型。

## 1.4修复后的检测结果

| 模型 | Clean mAP/NDS | Crash 5f | Crash 10f | Compound 10f |
|---|---|---|---|---|
| B0 | 0.4292/0.4808 | 0.4186/0.4738 | 0.4123/0.4715 | 0.3892/0.4543 |
| B1 | 0.4182/0.4738 | 0.4100/0.4682 | 0.4094/0.4693 | 0.3917/0.4573 |
| T1 | 0.4272/0.4791 | 0.4196/0.4731 | 0.4113/0.4705 | 0.3915/0.4574 |

当前结论：

1. T1的Clean性能明显优于B1，并接近B0；
2. T1在Crash 5f下略优于B0；
3. T1在Compound条件下优于B0；
4. T1尚未在所有协议下稳定超过B0。

## 1.5严格恢复延迟结果

在连续2帧、100%GT支持恢复标准下：

| 模型 | 平均恢复延迟 | 最大恢复延迟 |
|---|---:|---:|
| B0 | 4.00帧 | 6帧 |
| B1 | 8.00帧 | 15帧 |
| T1 | 5.33帧 | 7帧 |

当前结论：

```text
B0恢复最快
T1明显优于B1
T1仍未超过B0
```

Stage2的首要目标是让完整证据账本进一步缩短T1的严格恢复延迟。

## 1.6三态轨迹验证

Stage1轨迹结果：

| 阶段 | KEEP | RECOVER | DEFER |
|---|---:|---:|---:|
| Clean | 99.91% | 0.01% | 0.07% |
| Crash 5f故障期 | 94.55% | 4.34% | 1.11% |
| Crash 10f故障期 | 94.86% | 3.70% | 1.44% |
| Compound 10f故障期 | 57.32% | 37.08% | 5.60% |
| Compound恢复后 | 99.84% | 0.02% | 0.14% |

逐帧结果已验证：

```text
第3帧故障开始时动作立即变化
第13帧观测恢复时KEEP立即回升到97%以上
第14帧基本恢复到100%KEEP
```

因此Stage1已经证明三态状态真正进入推理决策链，而不是只作为训练损失存在。

---

# 2.Stage2核心研究假设

## H1：证据来源必须被显式记录

来自同一相机、相邻帧或高度重叠视角的证据不能被视为完全独立。

否则模型会因为重复看到相似信息而错误提高置信度。

Stage2需要让每个时序查询保留相机来源向量：

```text
provenance ∈ R^C
```

其中C为相机数量。

## H2：重复证据不能无限增强账本

同一来源连续提供高度相似的观测时，新增证据量应受到新颖度折扣。

需要区分：

```text
新证据
重复证据
相关证据
无新观测
矛盾证据
```

## H3：没有新观测时证据强度只能衰减

如果当前帧没有可靠新观测，账本中的证据总强度不能增加。

目标行为：

```text
S_t ≤ γS_(t-1)
```

其中：

```text
S=alpha+beta
0<γ≤1
```

如果检测到可靠新证据，才允许：

```text
S_t=γS_(t-1)+有效新证据
```

## H4：DEFER必须阻止不可靠状态污染时序记忆

DEFER不应简单删除当前帧所有检测结果。

建议保持Stage1当前逻辑：

```text
当前帧仍可输出检测结果
但DEFER查询不得写入下一帧时序记忆
```

这样可以避免召回率突然崩溃，同时阻止不可靠查询持续污染后续帧。

---

# 3.Stage2目标指标

nuScenes-mini只能作为工程验证，不能作为最终论文主实验。以下目标首先作为mini阶段的Go/No-Go门槛。

## 3.1最低工程通过标准

```text
Clean mAP下降≤0.003
Clean NDS下降≤0.003

Crash 5f不低于T1 Stage1
Crash 10f不低于T1 Stage1
Compound 10f不低于T1 Stage1

100%恢复平均延迟≤5.0帧
100%恢复最大延迟≤7帧

所有协议无NaN、OOM和状态锁死
```

## 3.2强结果标准

```text
Clean性能基本保持T1水平
三种故障协议平均mAP和NDS均高于T1
Compound协议获得最明显提升

100%恢复平均延迟≤4.0帧
100%恢复最大延迟≤6帧

故障结束后第1帧KEEP≥97%
故障结束后第2帧KEEP≥99%

低观测且无新证据时不存在证据强度异常增长
```

## 3.3高水平论文伸展目标

```text
Clean性能接近或超过B0
平均故障性能超过B0和T1
严格恢复延迟达到或超过B0

多个故障类型和强度下稳定有效
随机相机、随机开始帧和随机持续时间下有效
完整nuScenes上有效
第二个数据集或公开腐蚀基准上有效

具有校准性、可解释性和计算效率实验
```

---

# 4.Stage2方法设计

## 4.1查询级证据账本状态

每个传播查询至少保存：

```text
alpha               正存在证据
beta                负存在证据
strength             alpha+beta
presence_probability alpha/(alpha+beta)
uncertainty          与证据总强度相关的不确定性

provenance           各相机来源贡献向量
age                  距离最后可靠观测的时间
last_observed        是否有当前可靠观测
novelty              当前证据新颖度
effective_count      有效独立证据数量

action               KEEP/RECOVER/DEFER
write_mask           是否写入下一帧记忆
```

不得在各模块中重复维护不一致的账本副本。

统一由：

```text
models/evidence_ledger.py
```

管理状态更新和场景重置。

## 4.2三态证据映射

三态输出：

```text
present
absent
unobserved
```

建议行为：

```text
present：
向alpha增加证据

absent：
仅在可观测性充分时向beta增加证据

unobserved：
不直接向alpha或beta增加强证据
主要影响observability、age和action
```

特别注意：

> 不可观测不等于目标不存在。

这是Stage2必须保持的基本语义。

证据理论在自动驾驶占用建模中已被用于显式表达未观测空间和冲突测量，说明“未知/未观测”不能被简单并入空闲或不存在类别。citeturn969953search1turn969953search9

## 4.3来源向量

当前帧每个查询计算：

```text
current_source ∈ R^6
```

来源可根据以下量构造：

```text
相机FOV覆盖
投影可见性
图像质量
注意力权重
有效采样点数量
遮挡或截断情况
```

推荐归一化：

```text
current_source=current_source/(sum(current_source)+eps)
```

账本来源更新：

```text
provenance_t=
normalize(
    γ_source×provenance_(t-1)
    +effective_new_evidence×current_source
)
```

必须保存未归一化强度或额外保存`source_strength`，避免只保留比例而丢失证据规模。

## 4.4证据新颖度

建议先实现一个稳定的组合形式：

```text
novelty=
λ_source×source_novelty
+λ_feature×feature_novelty
+λ_time×time_novelty
```

其中：

```text
source_novelty=
1-cosine(current_source,previous_source)

feature_novelty=
clamp((1-cosine(current_query,memory_query))/2,0,1)

time_novelty=
min(age/T_novelty,1)
```

第一版建议：

```text
λ_source=0.4
λ_feature=0.4
λ_time=0.2
```

必须通过配置文件控制，禁止硬编码在多个文件中。

新颖度应满足：

```text
完全相同来源和特征→接近0
新相机或明显变化特征→升高
长时间未观测后重新出现→升高
```

## 4.5相机相关性折扣

建议维护相机相关矩阵：

```text
R∈R^(C×C)
```

第一版可采用固定先验：

```text
同一相机相关性=1
相邻相机相关性较高
对向相机相关性较低
```

后续再加入基于特征或重叠区域的动态相关性。

有效独立证据数建议采用：

```text
N_eff=(sum(w))²/(wᵀRw+eps)
```

其中：

```text
w=current_source或来源强度
```

要求：

```text
重复同一相机证据不会使N_eff线性增长
多个低相关相机共同支持时N_eff增加
N_eff范围限制在[0,C]
```

## 4.6Beta证据守恒更新

设当前有效新证据为：

```text
e_pos
e_neg
```

更新：

```text
alpha_t=
1+gamma×(alpha_(t-1)-1)+e_pos

beta_t=
1+gamma×(beta_(t-1)-1)+e_neg
```

有效证据：

```text
e_pos=
raw_present_evidence
×observability
×novelty
×correlation_discount

e_neg=
raw_absent_evidence
×observability
×novelty
×correlation_discount
```

当：

```text
observability低
或novelty接近0
```

必须满足：

```text
e_pos≈0
e_neg≈0
```

无新观测时：

```text
alpha_t=1+gamma×(alpha_(t-1)-1)
beta_t=1+gamma×(beta_(t-1)-1)
```

不能出现：

```text
无新观测但strength持续增长
```

## 4.7守恒约束损失

建议新增：

```text
L_conservation
```

约束账本强度增长不得超过有效新证据：

```text
L_conservation=
ReLU(
    S_t
    -gamma×S_(t-1)
    -kappa×E_new
)
```

其中：

```text
E_new=e_pos+e_neg
```

第一版只对低可观测或无新观测查询启用，防止对正常学习产生过强限制。

## 4.8策略门控

保持已有策略输入：

```text
observability
presence_probability
uncertainty
age_since_observation
prior_strength
```

建议Stage2策略：

### KEEP

```text
observability充分
presence_probability充分
uncertainty较低
当前证据有效
```

### RECOVER

```text
当前observability不足
历史prior_strength仍充分
presence_probability较高
uncertainty未超过阈值
age未超过最大恢复年龄
```

### DEFER

```text
以上条件均不满足
或当前证据严重冲突
或历史证据已经过期
```

写回行为：

```text
KEEP：正常写回
RECOVER：折扣写回
DEFER：不写回
```

建议增加：

```text
recover_write_scale
```

让RECOVER写回强度小于KEEP，而不是二者完全相同。

---

# 5.损失函数

建议总损失：

```text
L_total=
L_detection
+λ_tri×L_ternary
+λ_obs×L_observability
+λ_evi×L_evidential
+λ_cons×L_conservation
+λ_cal×L_calibration
+λ_gate×L_policy_consistency
```

第一轮推荐权重仅作为起点：

```text
λ_tri=0.2
λ_obs=0.1
λ_evi=0.05
λ_cons=0.02
λ_cal=0.02
λ_gate=0.01
```

Codex不得一次性大范围调参。

先确保：

```text
所有损失有限
梯度正常
Clean性能没有立刻下降
```

然后逐项搜索权重。

---

# 6.代码修改边界

## 6.1models/evidence_ledger.py

主要实现：

```text
账本初始化
场景重置
alpha/beta守恒更新
来源向量更新
新颖度
有效独立证据数量
age更新
动作和write_mask
诊断信息输出
```

要求：

```text
不改变现有外部接口，除非确有必要
新增参数必须有默认值
支持CPU单元测试
支持FP16训练
所有状态随device和dtype正确迁移
场景切换必须清空账本
```

## 6.2models/streampetr_adapter.py

主要负责：

```text
从StreamPETR取得当前query、历史query和相机支持
调用observability_head
调用ternary_objectness
构造raw evidence
调用EvidenceLedger.update
应用动作分数缩放
应用write_mask
输出diagnostics
```

不得把完整账本更新逻辑重复写入Adapter。

## 6.3models/keep_recover_defer.py

保持策略模块独立。

新增配置建议：

```text
keep_min_novelty
recover_write_scale
defer_write_scale=0
conflict_defer_threshold
hard_gate_start_iter
```

## 6.4models/observability_head.py

优先保持当前结构。

只在确有需要时增加：

```text
每相机支持输出
质量置信度
可用相机数量
来源归一化结果
```

避免同时重写可观测性网络和证据账本。

## 6.5evaluation/evidence_single_gpu_test.py

扩展diagnostics字段：

```text
alpha
beta
strength
presence_probability
uncertainty
observability
novelty
effective_count
provenance
age
action
write_mask
conservation_residual
```

保持现有JSONL结构向后兼容。

---

# 7.必须先完成的单元测试

新建或补充：

```text
tests/test_evidence_ledger_stage2.py
tests/test_source_novelty.py
tests/test_correlation_discount.py
tests/test_conservation_update.py
tests/test_stage2_policy_gate.py
```

必须覆盖：

## 7.1无新观测守恒

```text
输入：
observability=0
new evidence=0

要求：
strength_t≤strength_(t-1)
alpha和beta只能向先验1衰减
```

## 7.2重复证据折扣

```text
连续输入完全相同的source和query feature

要求：
novelty逐渐降低
effective evidence不能线性增长
```

## 7.3独立相机支持

```text
两个低相关相机同时支持

要求：
N_eff高于单相机
证据增长高于重复同相机
```

## 7.4相机故障

```text
某相机支持突然变为0

要求：
provenance对应分量下降
observability下降
RECOVER或DEFER增加
```

## 7.5DEFER不写回

```text
action=DEFER

要求：
write_mask=False
下一帧memory中不出现该条新状态
当前帧输出接口仍有效
```

## 7.6场景重置

```text
scene_token变化

要求：
所有账本状态回到初始值
不存在跨场景污染
```

## 7.7FP16与CPU安全

禁止再次出现：

```text
clamp_min_cpu not implemented for Half
```

## 7.8保存与加载

```text
state_dict保存
重新加载
前向结果一致
```

---

# 8.Stage2实施顺序

## S2.0：冻结Stage1

创建标签或分支：

```text
stage1-r50-final-fixed-v2
```

保存：

```text
当前Git提交
配置
checkpoint
fixed_v2结果
恢复分析
动作轨迹
环境信息
SHA256
```

## S2.1：仅实现守恒账本，不启用硬门控

配置：

```text
mini_stage2_ledger_smoke.py
mini_stage2_ledger_debug.py
```

功能：

```text
alpha/beta更新
age
strength
conservation_residual
```

先不启用：

```text
相关性折扣
硬DEFER
复杂新颖度
```

验收：

```text
单元测试通过
24个原测试不回归
smoke前向成功
50iter无NaN
```

## S2.2：来源向量

增加：

```text
per-camera provenance
source_strength
source decay
```

验收：

```text
相机Crash后对应来源下降
Clean下来源分布稳定
```

## S2.3：新颖度

先实现：

```text
source novelty
feature novelty
time novelty
```

分别可通过配置关闭。

验收：

```text
重复证据新颖度下降
恢复时新颖度上升
```

## S2.4：相关性折扣

先实现固定相关矩阵，再尝试动态相关矩阵。

验收：

```text
同一来源重复证据增长受限
多相机独立支持仍能增强证据
```

## S2.5：软写回门控

```text
KEEP write_scale=1.0
RECOVER write_scale=0.5～0.8
DEFER write_scale=0.0～0.1
```

先不要立即使用完全硬门控。

## S2.6：严格门控

训练后期启用：

```text
DEFER write_scale=0
```

但仍保留当前帧检测输出。

## S2.7：完整mini训练与协议矩阵

通过后才运行完整固定协议评测。

---

# 9.训练课程设计

不建议只在固定“后相机第3帧故障”协议上训练。

训练协议应随机化：

```text
随机故障相机
随机单相机或多相机
随机开始帧
随机持续1/3/5/10/20帧
随机持续故障
随机恢复
随机Dark/Fog/Motion Blur
随机复合故障
```

第一版训练采样比例：

```text
Clean：40%
单相机Crash/Frame Lost：25%
图像质量退化：20%
多相机或复合退化：15%
```

训练应保持足够Clean比例，否则Clean性能容易下降。

建议课程：

```text
0%～20%：
Clean和轻度退化
软账本，无硬门控

20%～50%：
加入单相机故障
启用来源和守恒

50%～80%：
加入长故障和复合故障
启用新颖度与相关性折扣

80%～100%：
启用严格DEFER写回门控
```

长期记忆方法的性能不仅依赖结构，也依赖训练时如何组织时序数据和推理分布；近期工作同样强调记忆模块与训练调度需要联合设计。citeturn969953search2

---

# 10.配置与消融矩阵

必须建立以下配置，避免只比较Stage1和最终完整版。

```text
stage2_e0_stage1.py
stage2_e1_ledger_only.py
stage2_e2_plus_provenance.py
stage2_e3_plus_novelty.py
stage2_e4_plus_corr_discount.py
stage2_e5_plus_conservation.py
stage2_e6_soft_gate.py
stage2_e7_full.py
```

反向消融：

```text
stage2_full_no_provenance.py
stage2_full_no_novelty.py
stage2_full_no_corr_discount.py
stage2_full_no_conservation.py
stage2_full_no_defer.py
stage2_full_no_recover_write.py
```

每个实验只允许一个主要变量发生变化。

---

# 11.实验输出结构

统一使用：

```text
outputs/stage2/
└── <experiment_id>/
    ├── config.py
    ├── train.log
    ├── environment.txt
    ├── git_commit.txt
    ├── git_diff.patch
    ├── checkpoint/
    ├── eval/
    │   ├── clean/
    │   ├── camera_crash_5f/
    │   ├── camera_crash_10f/
    │   ├── compound_10f/
    │   └── full_protocol_matrix/
    ├── traces/
    ├── metrics/
    └── checksums/
```

所有Stage2新评测目录使用：

```text
fixed_v3_stage2_
```

前缀，禁止与Stage1的`fixed_v2_`混合。

---

# 12.评测指标体系

## 12.1检测指标

```text
mAP
NDS
mATE
mASE
mAOE
mAVE
mAAE
```

## 12.2鲁棒性指标

```text
绝对性能下降
相对性能保留率
平均故障性能
最差故障性能
最差场景性能
故障长度敏感性
```

## 12.3恢复指标

保留：

```text
w2_t090
w2_t095
w2_t100
w3_t095
```

增加：

```text
平均恢复延迟
最大恢复延迟
未恢复率
动作恢复延迟
证据强度恢复延迟
```

## 12.4证据守恒指标

新增：

```text
Conservation Violation Rate
```

定义为：

```text
在低可观测且无新颖证据时，
strength_t>gamma×strength_(t-1)+tolerance
的查询比例
```

目标：

```text
接近0
```

新增：

```text
Unsupported Evidence Growth
```

统计无可靠观测时的平均证据增长量。

## 12.5动作指标

```text
故障开始动作响应延迟
故障阶段KEEP/RECOVER/DEFER比例
恢复阶段动作回归延迟
错误DEFER率
长期RECOVER率
恢复后锁死率
```

## 12.6校准指标

建议增加：

```text
ECE
Brier Score
NLL
Risk-Coverage Curve
AURC
```

分别针对：

```text
存在概率
三态概率
可观测性
```

近期3D检测研究越来越重视语义不确定性与几何不确定性的联合分析，因此只报告mAP/NDS不足以形成完整的可靠性感知论文。citeturn969953search4

## 12.7效率指标

```text
参数量
FLOPs
显存
FPS
单帧延迟
账本额外内存
```

---

# 13.面向高水平论文的数据扩展

nuScenes-mini仅用于开发和排错。

正式论文至少需要：

```text
完整nuScenes train/val
多个随机种子
完整故障协议矩阵
不同相机故障组合
不同退化强度
不同故障持续时间
```

强烈建议再加入：

```text
第二个公开数据集
或公开鲁棒性基准
```

当前公开研究已经开始系统评估天气、传感器故障、缺失和组合腐蚀，说明论文必须覆盖多种退化，而不能只依赖单一后相机Crash。citeturn969953academia36turn969953academia37

建议最终协议覆盖：

```text
Camera Crash
Frame Lost
Persistent Failure
Intermittent Failure
Dark
Fog
Motion Blur
Multi-camera Failure
Fog+Crash
Dark+Frame Lost
Natural Recovery
```

---

# 14.论文贡献组织

论文不能写成：

```text
在StreamPETR上增加了几个不确定性模块。
```

应组织成以下贡献链。

## Contribution 1：三态时序目标认知

提出：

```text
Present/Absent/Unobserved
```

避免把不可观测目标错误监督为不存在。

## Contribution 2：来源感知证据守恒账本

每个目标查询维护：

```text
Beta证据状态
相机来源
证据年龄
新颖度
有效独立证据数量
```

## Contribution 3：证据驱动的时序动作

根据账本状态产生：

```text
Keep/Recover/Defer
```

并控制查询写回，而不是只改变最终检测分数。

## Contribution 4：PartialObs-3D协议与恢复指标

系统评估：

```text
故障开始
故障持续
自然恢复
复合退化
```

并报告：

```text
恢复延迟
动作响应
证据守恒
校准性
```

外部研究已经分别证明长期记忆、证据理论和腐蚀鲁棒性的重要性；Evidence3D真正需要形成差异化的部分，是把“可观测性、来源独立性、证据守恒和时序写回决策”统一到同一目标查询账本中。citeturn969953search1turn969953search2turn969953academia36

---

# 15.Codex执行原则

Codex必须遵守：

```text
先检查，后修改
先测试，后训练
先单模块，后组合
先mini，后完整数据
每次只改变一个核心变量
任何错误不得通过删除断言来掩盖
不得复用旧输出冒充新实验
不得覆盖Stage1正式结果
不得直接大规模修改第三方StreamPETR
```

每次修改后必须输出：

```text
1.修改文件列表
2.修改原因
3.核心逻辑
4.兼容性风险
5.运行命令
6.测试结果
7.输出目录
8.下一步建议
```

---

# 16.Codex首轮任务

将以下内容直接发送给Codex：

```text
请在/home/research/research/evidence3d中继续实现Evidence3D Stage2。

现有Stage1已经完成并验证：
1.三态Present/Absent/Unobserved监督；
2.Keep/Recover/Defer策略；
3.fixed_v2协议评测；
4.GT恢复延迟分析；
5.逐阶段和逐帧动作轨迹分析。

不得覆盖Stage1结果，不得大规模修改repos/StreamPETR。

第一轮只完成Stage2工程审计与S2.1证据守恒账本，不要立即实现所有模块。

请依次执行：

A.检查并总结以下文件的当前接口、张量形状、状态生命周期和调用关系：
- models/evidence_ledger.py
- models/keep_recover_defer.py
- models/streampetr_adapter.py
- models/observability_head.py
- models/ternary_objectness.py
- evaluation/evidence_single_gpu_test.py
- 当前Stage1 R50配置文件

B.绘制文字版数据流：
当前query和memory query
→observability
→ternary probabilities
→raw evidence
→ledger update
→action
→score scaling
→memory write

C.检查当前EvidenceLedger是否已经满足：
- alpha/beta守恒衰减
- scene reset
- prev_exists reset
- age更新
- provenance状态
- novelty状态
- effective_count状态
- DEFER不写回

D.在不改变现有外部接口的前提下，实现或修正S2.1：
- 无新观测时alpha/beta只能向先验1衰减；
- 有新观测时才能增加证据；
- 输出strength和conservation_residual；
- 所有新行为通过配置开关控制；
- 默认关闭时必须与Stage1行为兼容。

E.新增tests/test_evidence_ledger_stage2.py，至少覆盖：
1.无新观测强度不增长；
2.可靠正证据增加alpha；
3.可靠负证据增加beta；
4.unobserved不产生强正负证据；
5.场景切换重置；
6.DEFER write_mask为False；
7.CPU和FP16安全；
8.状态保存加载。

F.运行：
- 新增单元测试；
- 现有全部单元测试；
- py_compile；
- 一个最小前向smoke测试。

G.不要开始完整训练。完成后报告：
- 当前账本存在的问题；
- 修改文件；
-关键公式；
-测试结果；
-仍未实现的Stage2功能；
-S2.2来源向量的具体接入点。

所有新配置、输出和测试名称使用stage2或fixed_v3前缀。
```

---

# 17.Stage2首轮完成判据

Codex第一轮完成后，必须满足：

```text
所有原单元测试通过
新增守恒测试通过
Stage1默认配置结果不变
EvidenceLedger无新观测时强度不增长
diagnostics能够输出strength和conservation_residual
没有开始不可控的大规模训练
```

完成S2.1后再依次进入：

```text
S2.2来源向量
S2.3证据新颖度
S2.4相关性折扣
S2.5软写回
S2.6硬门控
S2.7完整mini评测
S2.8完整nuScenes实验
```

---

# 18.最终成功标准

Stage2是否成功，不能只看某一个mAP。

必须同时满足：

```text
Clean性能基本不损失
故障性能提升
严格恢复延迟缩短
证据守恒违规接近0
Compound故障动作响应合理
恢复后不锁死
不确定性校准改善
计算开销可控
多协议和多随机种子下稳定
```

只有形成以下闭环，才具有高水平论文潜力：

```text
问题定义
→方法原则
→可验证的守恒性质
→故障和恢复协议
→检测与校准指标
→动作和证据可解释性
→完整消融
→跨数据集泛化
```