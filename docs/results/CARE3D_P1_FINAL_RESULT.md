# CARE-3D P1 Final Confirmatory Result

## Decision

GO_CARE3D_P1_SPARSE_EVIDENCE_ROUTER

P2_STATUS: ELIGIBLE

## Frozen formal setting

- Dataset: full nuScenes
- P1 probe-test scenes: 132 / 132
- Seeds: 42, 2027, 2028
- Bootstrap repetitions: 5000
- Source bank:
  - CAM_BACK_LEFT
  - CAM_BACK_RIGHT
  - TEMPORAL_ANCHOR
- Sparse routing Top-K: 2
- Classification only: true
- Regression unchanged: true
- Clean identity: true
- Probe-test opened only after all three training checkpoints were frozen: true

## Protocol decision

- blur_back: PASS
- crash_back: FAIL
- dark_back: PASS

Required passing fault families: 2

Passing fault families:
- blur_back
- dark_back

Therefore the preregistered P1 gate is passed.

## Main result

CARE-3D sparse evidence routing consistently recovers fault-lost targets under
Motion Blur and Dark while satisfying the frozen recovery, net-TP, target-score,
retained-object no-harm, false-positive control, and clean-identity gates across
all three seeds.

Camera Crash shows strong target recovery and positive net TP, but fails the
frozen false-positive control point-estimate gate. Its FP inflation is
approximately 1.21%-1.23%, above the preregistered 1% point-estimate threshold,
while its bootstrap CI upper bound remains below the frozen 2% limit.

No P1 threshold is retuned after probe-test observation.

## Frozen artifact SHA256

- decision.json
  d0172c9de8d17c4359224a90cee11312d260b45f595fcf662d5d8ccbd2405da3

- p1_metrics.csv
  ea02ee1bcf9cc09b89fee14a96e28f033ebf93ec405aef2b8d4aea4ce8fe08a4

- p1_gate_summary.csv
  40cbb861e853b802b1c0cce81d286c05e45c541187f6259d940ab0cb77f94b5d

- p1_cluster_ci.csv
  61428c9d518fe8277b453c5bbe63443dcb10c453506d1fc3a1399c7872ef20ea

- p1_fp_ci.csv
  af7d89f932d369040c897875f37f59436807af6e6d8e553cccc155b7f54b6c08
