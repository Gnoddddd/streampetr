# Evidence3D

**部分可观测条件下的证据守恒多相机三维目标检测**的 StreamPETR + nuScenes-mini 训练工程。

本工程严格采用以下边界：

- Windows 11 只负责 VS Code 图形界面；代码、CUDA、数据处理和训练均在 WSL2 Ubuntu 22.04 中执行。
- 项目根目录默认是 `/home/research/research/evidence3d`，简写为 `~/research/evidence3d`。
- `nusc-tools` 只用于 nuScenes 数据检查与可视化；模型训练使用独立的 `streampetr` Conda 环境。
- 官方第三方源码只放在 `repos/StreamPETR`；本文方法放在根目录的 `models/`、`datasets/`、`protocols/`。
- 本地 RTX 3080 Ti Laptop 以 nuScenes-mini 的单批次、少量迭代、可视化和错误排查为目标；完整 nuScenes 与正式多随机种子实验应迁移到 Linux 服务器。

## 1. 已实现内容

### 1.1 三维可观测性场

`models/observability_head.py` 将每个三维查询中心投影到六个相机视锥，并计算：

- 几何覆盖 `O_geo`；
- 相机在线状态 `O_cam`；
- 图像质量 `O_quality`；
- 当前帧新鲜度；
- 归一化相机来源向量；
- 相关性折扣后的有效独立证据数 `N_eff`。

mini 首版按研究方案只将几何覆盖、在线状态和图像质量作为核心变量，不把预测置信度误当成可观测性。

### 1.2 存在—不存在—不可观测三态监督

`models/ternary_objectness.py` 为每个查询输出：

```text
present / absent / unobserved
```

Hungarian 匹配后的正查询监督为 `present`。未匹配查询采用软目标：

```text
O * absent + (1 - O) * unobserved
```

同时，原始类别分支中的背景损失会按可观测性缩放，避免低可观测查询被强制训练成高置信度背景。

### 1.3 证据来源账本与新颖度

`models/evidence_ledger.py` 为 StreamPETR 的传播查询维护：

- Beta 参数 `alpha / beta`；
- 相机来源向量；
- 距离最近有效观测的年龄；
- 当前可观测性；
- 证据新颖度；
- 有效独立证据数量；
- Keep / Recover / Defer 状态。

账本只保存紧凑状态，不保存历史图像。

### 1.4 证据守恒时序更新

`models/temporal_update.py` 实现：

```text
alpha_t = 1 + gamma * (alpha_{t-1} - 1) + O * novelty * N_eff * e_pos
beta_t  = 1 + gamma * (beta_{t-1}  - 1) + O * novelty * N_eff * e_neg
```

当没有新观测时，新增证据门控为零，历史证据强度只能按 `gamma` 衰减。训练日志还记录无新证据查询上的最大守恒比率与守恒违例值。

### 1.5 Keep / Recover / Defer

`models/keep_recover_defer.py` 根据可观测性、存在概率、未知度、观测年龄和负证据直接作出动作，不使用额外多专家路由器：

- `Keep`：当前可观测且证据充分；
- `Recover`：当前观测弱，但新鲜历史先验仍有效；
- `Defer`：未知度过高或历史已过期；降分并停止高置信度记忆写入。

### 1.6 PartialObs-3D 退化协议

已实现：

- Camera Crash；
- Frame Lost；
- Dark；
- Fog；
- Motion Blur；
- 持续故障与自然恢复；
- 多相机联合故障；
- 复合退化；
- JSON 可复现协议调度。

训练退化按样本 token 和帧号产生确定性随机种子。测试默认不随机退化，只读取显式协议文件。

## 2. 项目结构

```text
evidence3d
├── .vscode
├── checkpoints
├── configs
│   ├── base
│   ├── streampetr
│   └── evidence_conserving
├── data
│   └── nuscenes-mini
├── datasets
├── docs
├── downloads
├── evaluation
├── evidence3d_plugin
├── hooks
├── models
├── outputs
├── protocols
├── repos
│   └── StreamPETR              # 由脚本克隆，压缩包不内置第三方源码
├── scripts
├── tests
└── tools
```

## 3. WSL 中安装

所有命令均在 Ubuntu 22.04 的 WSL 终端执行，不要在 Windows PowerShell 中执行。

```bash
cd ~/research/evidence3d
code .
```

确认终端：

```bash
pwd
echo "$WSL_DISTRO_NAME"
uname -s
```

期望：

```text
/home/research/research/evidence3d
Ubuntu-22.04
Linux
```

### 3.1 创建正式训练环境并安装官方兼容栈

```bash
cd ~/research/evidence3d
bash scripts/install_streampetr_env.sh
conda activate streampetr
```

安装脚本会：

1. 创建 Python 3.8 的 `streampetr` 环境；
2. 安装 PyTorch 1.9.0 + CUDA 11.1；
3. 安装 MMCV 1.6.0、MMDetection 2.28.2、MMSegmentation 0.30.0；
4. 克隆并固定 StreamPETR 到提交 `95f64702306ccdb7a78889578b2a55b5deb35b2a`；
5. 在 `repos/StreamPETR/mmdetection3d` 安装 MMDetection3D `v1.0.0rc6`；
6. 写入本项目、StreamPETR 和 MMDetection3D 的 `.pth` 搜索路径；
7. 默认跳过旧版 FlashAttention，改用官方 `PETRMultiheadAttention`。

检查：

```bash
which python
python --version
nvidia-smi
python tools/diagnose.py --strict
```

### 3.2 `nusc-tools` 环境边界

`nusc-tools` 只运行数据检查与原始数据可视化：

```bash
conda activate nusc-tools
python scripts/check_nuscenes.py
python scripts/render_nuscenes.py
```

不要在 `nusc-tools` 中安装 PyTorch、MMCV、MMDetection、MMDetection3D 或 StreamPETR。

## 4. 准备 nuScenes-mini

原始数据应位于：

```text
~/research/evidence3d/data/nuscenes-mini
├── LICENSE
├── maps
├── samples
├── sweeps
└── v1.0-mini
```

先在 `nusc-tools` 中检查原始数据，再在 `streampetr` 中生成 StreamPETR 的 2D 标注与时序索引：

```bash
conda activate nusc-tools
python scripts/check_nuscenes.py

conda activate streampetr
bash scripts/prepare_nuscenes_mini.sh
```

成功后应生成：

```text
data/nuscenes-mini/nuscenes2d_temporal_infos_train.pkl
data/nuscenes-mini/nuscenes2d_temporal_infos_val.pkl
```

## 5. 测试顺序

### 5.1 纯模块测试

```bash
conda activate streampetr
bash scripts/smoke_test.sh
```

该命令运行单元测试和证据守恒诊断，不要求先启动完整训练。

### 5.2 两次迭代的真实模型烟雾测试

完成依赖和数据索引后：

```bash
conda activate streampetr
RUN_MODEL_SMOKE=1 bash scripts/smoke_test.sh
```

该测试使用 `configs/evidence_conserving/mini_smoke.py`，检查真实数据读取、Forward、Backward、优化器和 checkpoint 链路。

## 6. 训练

### 6.1 官方 StreamPETR 干净数据基线

```bash
conda activate streampetr
bash scripts/train_baseline_mini.sh \
  --work-dir outputs/exp_001_baseline
```

### 6.2 官方 StreamPETR + 相同退化分布

```bash
conda activate streampetr
bash scripts/train_baseline_corrupted_mini.sh \
  --work-dir outputs/exp_002_baseline_corrupted
```

### 6.3 证据守恒 mini 调试训练

```bash
conda activate streampetr
bash scripts/train_mini.sh \
  --work-dir outputs/exp_003_evidence_conserving
```

`mini_debug.py` 默认 400 次迭代；`mini_train.py` 默认 4000 次迭代：

```bash
python tools/train.py \
  --config configs/evidence_conserving/mini_train.py \
  --work-dir outputs/exp_004_evidence_long
```

独立消融配置：

```text
configs/evidence_conserving/ablation_ternary_only.py
configs/evidence_conserving/ablation_decay_only.py
```

断点续训：

```bash
python tools/train.py \
  --config configs/evidence_conserving/mini_debug.py \
  --work-dir outputs/exp_003_evidence_conserving \
  --resume-from outputs/exp_003_evidence_conserving/latest.pth
```

## 7. 评测退化协议

示例协议：

```text
protocols/example_mini_protocol.json
protocols/presets/*.json
```

重新生成 1/3/5/10/20 帧 Camera Crash、Frame Lost、多相机故障、恢复和复合退化预设：

```bash
python scripts/generate_protocol_presets.py
```

协议事件使用闭区间帧号。例如：

```json
{
  "version": 1,
  "scenes": {
    "*": [
      {
        "start_frame": 3,
        "end_frame": 7,
        "failed_cameras": ["CAM_BACK"],
        "lost_cameras": [],
        "dark": {},
        "fog": {},
        "motion_blur": {}
      }
    ]
  }
}
```

单 GPU 评测：

```bash
conda activate streampetr
bash scripts/evaluate_mini.sh \
  outputs/exp_003_evidence_conserving/latest.pth \
  protocols/example_mini_protocol.json
```

`tools/evaluate.py` 会自动使用单进程分布式启动，因为固定版本的 StreamPETR `tools/test.py` 不支持普通单 GPU 分支。

## 8. 日志与输出

每个实验使用独立目录：

```text
outputs/exp_xxx
├── mini_debug.py
├── *.log
├── evidence_trace.jsonl
├── iter_*.pth
└── latest.pth
```

绘制证据状态：

```bash
python tools/visualize.py \
  outputs/exp_003_evidence_conserving/evidence_trace.jsonl
```

日志包含：

- 平均可观测性；
- 平均证据新颖度；
- 平均存在概率；
- 平均未知度；
- Keep / Recover / Defer 数量；
- 无新观测下的最大证据比率；
- 守恒违例值。

`evaluation/metrics.py` 提供 UFPR、SOP、EIR、RD 和 Risk–Coverage 的独立计算函数，可用于诊断实验汇总。示例输入与命令：

```bash
python tools/partialobs_report.py evaluation/example_metric_input.json
```

## 9. 关键配置

mini 默认参数：

```text
输入分辨率       256 x 704
batch size       1
基础查询数       256
传播查询数       64
记忆长度         256
Top-K 写入       64
gamma            0.90
证据 warm-up      200 帧
```

16 GB 显存仍不足时，按以下顺序调整：

1. 将 `num_query` 从 256 降到 192；
2. 将 `memory_len` 从 256 降到 192；
3. 将 `topk_proposals` 与 `num_propagated` 从 64 降到 48；
4. 保持 `samples_per_gpu=1`；
5. 不要开启旧版 FlashAttention；
6. 保留 backbone checkpointing。

## 10. VS Code 调试

`.vscode/launch.json` 提供：

- Evidence3D 单元测试；
- 环境严格检查；
- 两次迭代 mini 模型烟雾测试；
- 400 次迭代 mini 训练。

调试第三方代码已设置：

```json
"justMyCode": false
```

训练阶段默认解释器：

```text
/home/research/miniconda3/envs/streampetr/bin/python
```

运行 `scripts/check_nuscenes.py` 或 `scripts/render_nuscenes.py` 时，临时切换到：

```text
/home/research/miniconda3/envs/nusc-tools/bin/python
```

## 11. 设计限制

- nuScenes-mini 只适合代码调试，不能据此报告正式论文性能。
- 当前可观测性首版采用相机在线状态、几何视锥和退化质量；尚未加入显式深度遮挡估计。
- mini 实现用 Beta 未知度和历史年龄驱动 Keep/Recover/Defer，尚未增加独立的三维框协方差或运动预测协方差 head。
- Frame Lost 在训练 pipeline 中使用黑色占位图并显式设置 `fresh=0`，避免多 worker 随机访问下错误地把缓存帧当作新证据。离线协议工具仍提供 hold-last-frame 模拟。
- 证据账本与三态分支是研究实现，需要通过文档规定的两个诊断实验确认失败现象，再开展完整消融。
- 本压缩包不包含 nuScenes 数据、预训练权重或 StreamPETR 第三方源码；安装脚本会在 WSL 中按固定提交下载。

## 12. 验证状态

交付前已执行：

```text
24 个纯 PyTorch/工程单元测试
Python compileall
全部 Shell 脚本 bash -n
StreamPETR 适配层的桩环境形状测试
```

由于交付环境不含用户本机的 WSL GPU、nuScenes-mini 数据和固定版 OpenMMLab 栈，完整两次迭代 GPU 烟雾测试需要在本机执行第 5.2 节命令。该边界不会被隐藏为“已经完整训练通过”。

更多实现说明见：

- `docs/ARCHITECTURE.md`
- `docs/WSL_RUNBOOK.md`
- `docs/EXPERIMENTS.md`
