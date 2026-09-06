#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

python scripts/prepare_care3d_p1.py

pytest \
  tests/test_care3d.py \
  tests/test_care3d_p0_pipeline.py \
  tests/test_care3d_cross_severity.py \
  tests/test_care3d_p1.py \
  -v

python scripts/export_care3d_p1_supervision.py \
  --engineering-scene \
  --device cuda:0

python scripts/check_care3d_p1_engineering_smoke.py

cat reports/care3d/p1_sparse_evidence_router/engineering_smoke_gate.json
cat reports/care3d/p1_sparse_evidence_router/progress_manifest.json
