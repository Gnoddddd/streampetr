# CARE-3D P2-A0 Execution Runbook

P2-A0 removes the P1 oracle target-query input without changing P0/P1.  Run the
stages in order.  Do not open any P2 probe-test artifact during this gate.

## 1. Engineering smoke only

```bash
cd ~/research/evidence3d
conda activate streampetr
CUDA_VISIBLE_DEVICES=0 bash scripts/run_care3d_p2a_smoke.sh
```

Required final status:

`P2A_ENGINEERING_SMOKE_PASSED`

The smoke uses one previously excluded discovery scene.  It cannot be used for
parameter selection or scientific claims.

## 2. Probe-train extraction

Only after smoke passes:

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/export_care3d_p2a_association.py \
  --split probe_train \
  --device cuda:0
```

This evaluates all 15 preregistered global association configurations on the
419 probe-train scenes.  The detector/datasets are initialized once and the
900-query cost matrix is torch-vectorized.

Expected terminal progress state after complete extraction:

`P2A_TRAIN_EXTRACTION_COMPLETE_SELECTION_ELIGIBLE`

## 3. Freeze the unique train-only global config

```bash
python scripts/select_care3d_p2a_train_config.py
```

This writes `selection.json`.  It does not read probe-val outcomes before the
winner is frozen and it never reads probe-test.

Expected status:

`P2A_CONFIG_FROZEN_VAL_EXTRACTION_ELIGIBLE`

## 4. Probe-val extraction

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/export_care3d_p2a_association.py \
  --split probe_val \
  --device cuda:0
```

Probe-val evaluates only the frozen selected full association plus the three
frozen baselines.  No config or threshold can be changed after this step begins.

Expected state:

`P2A_VAL_EXTRACTION_COMPLETE_ANALYSIS_ELIGIBLE`

## 5. Frozen 5000-bootstrap Go/No-Go analysis

```bash
python scripts/analyze_care3d_p2a_val.py
```

Formal outcomes are exactly one of:

- `GO_CARE3D_P2A_ONLINE_QUERY_ASSOCIATION`
- `NO_GO_CARE3D_P2A_ONLINE_QUERY_ASSOCIATION`

P2-A1 online router evaluation is eligible only after GO.

## Resume behavior

Train and val extraction are per-scene resume-safe.  Re-running the same split
skips valid completion markers.  SIGINT/SIGTERM is honored after the current
scene is atomically saved.

## Forbidden during P2-A0

Do not:

- run any P2 probe-test evaluation;
- reuse P1 probe-test results to select association weights or thresholds;
- change P0/P1 checkpoints or gates;
- train a neural association model;
- use GT, clean target-frame output, or oracle query identity in association
  cost construction;
- alter the StreamPETR classifier execution shape for throughput.
