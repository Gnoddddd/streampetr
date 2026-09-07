# CARE-3D P1 deployment-shape classifier runbook

Use this runbook only after the pre-test replay diagnosis has completed and
`probe_test_read=false`.

## Validation

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_care3d_p1_deployment_shape_smoke.sh
```

The smoke must show:

- all selected tests pass;
- standalone trainer-style replay remains above the frozen `5e-4` tolerance;
- packed `[1,900,256]` replay is below `5e-4`;
- shape/index-matched `[1,900,256]` replay is below `5e-4`;
- probe-test remains unread.

## Training

Archive any incomplete pre-repair P1 training directory. Restart all three seeds
from initialization:

```bash
for SEED in 42 2027 2028; do
  CUDA_VISIBLE_DEVICES=0 python scripts/train_care3d_p1_deployment_shape.py \
    --seed "$SEED" --device cuda:0
done
```

Do not resume pre-repair checkpoints.

## Formal test

Only after all three manifests are frozen and probe-test is unlocked:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_care3d_p1_deployment_shape.py \
  --formal-test --device cuda:0
```

Do not run the legacy `train_care3d_p1.py` or `evaluate_care3d_p1.py` entrypoints
for the confirmatory repaired P1 run.
