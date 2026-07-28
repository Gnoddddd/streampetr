#!/usr/bin/env bash

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate streampetr

cd "$HOME/research/evidence3d" || exit 1

export EVIDENCE3D_PROJECT_ROOT="$PWD"
export EVIDENCE3D_ROOT="$PWD"
export PYTHONPATH="$PWD/repos/StreamPETR:$PWD${PYTHONPATH:+:$PYTHONPATH}"

unset EVIDENCE3D_PROTOCOL
unset EVIDENCE3D_PROTOCOL_DEBUG
unset EVIDENCE3D_EVAL_TRACE

echo "[环境已激活]"
echo "项目：$PWD"
echo "Python：$(which python)"
python --version
