# WSL 日常运行手册

## 每天开始

```bash
cd ~/research/evidence3d
code .
```

检查 VS Code 左下角为 `WSL: Ubuntu-22.04`，新终端执行：

```bash
pwd
echo "$WSL_DISTRO_NAME"
uname -s
```

## 数据工具

```bash
conda activate nusc-tools
which python
python scripts/check_nuscenes.py
python scripts/render_nuscenes.py
```

## 模型开发

```bash
conda activate streampetr
which python
nvidia-smi
python tools/diagnose.py --strict
python -m pytest -q tests
```

## 模型烟雾测试

```bash
RUN_MODEL_SMOKE=1 bash scripts/smoke_test.sh
```

## GPU 与系统监控

```bash
watch -n 1 nvidia-smi
htop
```

## 输出目录从 Windows 打开

```bash
cd ~/research/evidence3d/outputs
explorer.exe .
```

## 常见错误

### 终端显示 `PS C:\...`

当前是 Windows PowerShell。重新打开 VS Code 的 WSL 窗口或在 Ubuntu 终端执行 `code .`。

### `StreamPETR not found`

```bash
bash scripts/clone_streampetr.sh
```

### 找不到 temporal infos

```bash
conda activate streampetr
bash scripts/prepare_nuscenes_mini.sh
```

### CUDA 可见但 PyTorch 不可用

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

确认命令在 WSL 的 `streampetr` 环境中执行，并重新运行安装脚本。

### 显存不足

先使用 `mini_smoke.py`，再按 README 第 9 节缩减查询数和记忆长度。不要把 batch size 提高到 2。
