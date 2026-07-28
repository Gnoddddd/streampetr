#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${EVIDENCE3D_ROOT:-$HOME/research/evidence3d}"
ENV_NAME="${1:-streampetr}"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
  echo "Miniconda initialization script not found: $CONDA_SH" >&2
  exit 2
fi
source "$CONDA_SH"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.8 pip -y
fi
conda activate "$ENV_NAME"
python -m pip install --upgrade "pip==23.3.2" "setuptools<70" wheel

# Exact upstream-compatible legacy stack. NVIDIA drivers in WSL are normally
# backward-compatible with the bundled CUDA 11.1 runtime.
python -m pip install \
  torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
  -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install mmcv-full==1.6.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
python -m pip install mmdet==2.28.2 mmsegmentation==0.30.0
python -m pip install \
  "numpy==1.23.5" "numba==0.56.4" "nuscenes-devkit==1.1.10" \
  "pyquaternion==0.9.9" "shapely==1.8.5.post1" \
  "networkx==2.8.8" "scikit-image==0.19.3" "pandas==1.5.3" \
  "matplotlib<3.8" pytest

EVIDENCE3D_ROOT="$PROJECT_ROOT" "$PROJECT_ROOT/scripts/clone_streampetr.sh"
MMDET3D_DIR="$PROJECT_ROOT/repos/StreamPETR/mmdetection3d"
if [[ ! -d "$MMDET3D_DIR/.git" ]]; then
  git clone https://github.com/open-mmlab/mmdetection3d.git "$MMDET3D_DIR"
fi
git -C "$MMDET3D_DIR" fetch --all --tags
git -C "$MMDET3D_DIR" checkout v1.0.0rc6
python -m pip install -v -e "$MMDET3D_DIR"

python - <<PY
from pathlib import Path
import site
root = Path(r"$PROJECT_ROOT").resolve()
stream = root / "repos/StreamPETR"
mmdet3d = stream / "mmdetection3d"
site_dir = Path(site.getsitepackages()[0])
(site_dir / "evidence3d_local.pth").write_text(
    f"{root}\n{stream}\n{mmdet3d}\n", encoding="utf-8"
)
print("Wrote", site_dir / "evidence3d_local.pth")
PY

python - <<'PY'
import torch, mmcv, mmdet, mmseg, mmdet3d
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
print("mmcv:", mmcv.__version__)
print("mmdet:", mmdet.__version__)
print("mmseg:", mmseg.__version__)
print("mmdet3d:", mmdet3d.__version__)
PY

echo "Environment '$ENV_NAME' is ready. FlashAttention was intentionally skipped."
