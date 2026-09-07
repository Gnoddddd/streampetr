#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for SEED in 42 2027 2028
do
  echo
  echo "=================================================="
  echo "START CARE-3D P1 DEPLOYMENT-SHAPE SEED=${SEED}"
  echo "=================================================="
  date

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python scripts/train_care3d_p1_deployment_shape.py \
    --seed "${SEED}" \
    --device cuda:0

  echo
  echo "=================================================="
  echo "FINISHED CARE-3D P1 DEPLOYMENT-SHAPE SEED=${SEED}"
  echo "=================================================="
  date
done

echo
echo "ALL THREE CARE-3D P1 DEPLOYMENT-SHAPE SEEDS FINISHED"
