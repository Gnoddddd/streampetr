#!/usr/bin/env bash
set -euo pipefail

# Evidence3D StreamPETR Python path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STREAM_PETR_ROOT="${PROJECT_ROOT}/repos/StreamPETR"

export PYTHONPATH="${STREAM_PETR_ROOT}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

PROJECT_ROOT="${EVIDENCE3D_ROOT:-$HOME/research/evidence3d}"
cd "$PROJECT_ROOT"

echo "project=$PWD"
echo "wsl_distro=${WSL_DISTRO_NAME:-<unset>}"
echo "kernel=$(uname -s)"
echo "conda_env=${CONDA_DEFAULT_ENV:-<unset>}"
echo "python=$(command -v python || true)"
python --version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "nvidia-smi not found" >&2
fi
python tools/diagnose.py "${@}"
