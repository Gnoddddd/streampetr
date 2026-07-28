# 交付验证记录

验证日期：2026-07-13

## 已执行

```text
pytest: 24 passed
Python compileall: passed
Python 3.8 grammar parse: passed
Shell bash -n: passed
JSON parse: passed
StreamPETR adapter stub shape integration: passed
FP16 observability correlation regression: passed
```

测试覆盖：

- 三维投影与六/多相机可观测性；
- 相机相关性折扣与 `N_eff`；
- 三态软目标与不可观测背景抑制；
- 无新观测时 Beta 证据严格衰减；
- 新观测可增加证据；
- Recover 必须具有历史证据；
- Recover 无新观测时保留历史来源；
- Camera Crash、Frame Lost 与图像退化；
- 协议调度；
- UFPR、SOP、EIR、RD、Risk–Coverage；
- 自定义 StreamPETR Top-K 记忆写入张量形状；
- 关闭证据记忆的三态-only 消融路径。

## 当前环境无法执行的验证

交付容器没有用户本机的：

- WSL2 Ubuntu 22.04 NVIDIA GPU；
- nuScenes-mini 原始数据；
- Python 3.8 旧版 OpenMMLab 完整栈；
- StreamPETR 官方仓库本地 checkout。

因此没有声称已在交付容器完成真实 StreamPETR GPU Forward/Backward。用户本机的最终链路验证命令是：

```bash
conda activate streampetr
bash scripts/prepare_nuscenes_mini.sh
RUN_MODEL_SMOKE=1 bash scripts/smoke_test.sh
```
