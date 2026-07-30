# S3-R1 Observability-Guided Temporal Distillation

## Frozen baseline and groups

The reproducible B0 anchor is commit `20b0c93`, config
`configs/stage3/mini_convergence_b0.py`, and initialization checkpoint
`checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth` with SHA256
`e6323ae5c31adf1eedd46d6dd4fd3c73d95aa26f18cc8aa23c196494b7de3451`.
This screening uses exactly B0 clean-only, R0 with 50% clean / 50% uniformly
sampled corruption, and R1 with the same input distribution plus distillation.

The six mutually exclusive corruptions are Camera Crash, Frame Lost, Dark,
Fog, Motion Blur and Compound. Compound is fixed to one crashed view plus fog
on a different view. Sampling is deterministic from seed 2026 and sample token.

## Training-only method

R1 owns an EMA (momentum 0.999) copy of B0 outside the registered PyTorch
module tree. It sees the geometrically identical clean six-camera frame,
never receives gradients, and keeps an independent temporal memory. Student
and teacher each run the existing Hungarian assigner. Positive queries are
joined by the nuScenes GT instance token, never by query index.

For paired positives, R1 uses fixed weights 1.0 for logit MSE, 1.0 for L1 on
normalized 3D boxes and 0.1 for cosine query-embedding loss. There is no weight
search. Present GT keeps normal detection supervision. Negative queries receive
background supervision only when their predicted center projects inside at
least one online, fresh camera. Unobserved negative queries have classification
weight zero; positive GT and teacher temporal targets remain active.

The deployment state, inference path, query count, prediction head, Top-K and
memory write logic are B0. Disabling the module dispatches directly to B0.
Teacher parameters and all diagnostic tensors are excluded from `state_dict`.

## Preregistered decision rule

Checkpoints are selected independently per group using Clean NDS only among
epochs 1, 3 and 6. R1 advances only if Clean is within 0.001 of B0, fault-average
NDS is at least B0 + 0.003, at least two fault protocols improve, no protocol
regresses over 0.002, GT recall does not fall, false memory writes fall at least
10%, R1 beats R0, and training has no NaN, Inf or OOM. Failure stops the method
without tuning distillation weights.
