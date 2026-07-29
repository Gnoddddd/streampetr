# Mini configuration and pretrained-weight audit

Audit baseline: `s2.2-stable` (`9958366`).

## Dataset and budget

| split | scenes | samples |
|---|---:|---:|
| train | 8 | 323 |
| validation | 2 | 81 |

All three groups use one GPU and `samples_per_gpu=1`, hence the effective
batch is 1 and the measured mini-equivalent epoch length is 323 iterations.
The inherited `num_iters_per_epoch=1758` is a full-dataset constant and is not
used.  The frozen 12-epoch budget is 3,876 iterations, with checkpoints at
323, 969, 1,938, and 3,876 iterations (epochs 1/3/6/12).

## Pretraining

- Model initialization checkpoint:
  `checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth`
- Checkpoint SHA256:
  `e6323ae5c31adf1eedd46d6dd4fd3c73d95aa26f18cc8aa23c196494b7de3451`
- Backbone declaration: `torchvision://resnet50`
- Resolved local backbone cache:
  `/home/research/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth`
- Backbone cache SHA256:
  `0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a`

The full StreamPETR initialization checkpoint is loaded after module
construction and overwrites every B0 model tensor, including the backbone.

## Load and shared-initialization result

| group | state keys | missing | unexpected |
|---|---:|---:|---:|
| B0 | 591 | 0 | 0 |
| M1 | 629 | 38 | 0 |

The 38 M1-only missing keys are the expected new Evidence3D tensors:
`evidence_step`, the fixed `camera_correlation` buffer, and six ternary
branches (six parameters per branch).  No original StreamPETR key is missing.

B0 and M1 have 591 shared state keys.  After loading the same checkpoint with
seed 2026, all 591 shared tensors are exactly equal:

- unequal shared tensors: 0
- maximum absolute difference: 0
- shape mismatches: 0

Therefore pretraining loaded successfully and the common detector
initialization is tensor-identical.  M1's additional tensors are initialized
deterministically from the same seed.

## Fairness controls

All groups use seed 2026, the same mini annotations and sampler, AdamW
(`lr=2.5e-6`, weight decay 0.01), batch 1, dynamic FP16, max gradient norm 35,
and the same 3,876-iteration budget.  Official StreamPETR DN is disabled in
all groups, so DN loss keys are absent.  M1-Ramp differs from M1 only through
a stateless hook on the existing ternary auxiliary-loss return value.
