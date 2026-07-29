# S2.4 Baseline Confirmation Pre-registered Gate

This gate is frozen before either 200-iteration result is run or inspected.

## Candidates and fairness

- C0: `canonical_no_discount`; bypass correlation matrix and `N_eff`.
- C1: `legacy_fixed_discount`; reproduce historical S2.2 semantics.
- Initialization:
  `outputs/final_snapshots/stage1_ternary_r50_200/checkpoint/iter_200.pth`.
- Seed: `2026`.
- Both candidates use identical data, order, optimizer, learning rate, batch
  size, dynamic FP16, gradient clipping (`max_norm=35`), frozen parameter
  patterns and 200-iteration runner.
- The only model-semantic difference is
  `enable_correlation_discount=False/True`.

No threshold, matrix or implementation from the 50-iteration experiment may
be changed.

## Provisional canonical-baseline gate

C0 passes only if all conditions hold:

1. Clean mAP and Clean NDS are each greater than or equal to C1.
2. The arithmetic mean over Crash5, Crash10 and Compound is greater than or
   equal to C1 for both mAP and NDS.
3. No individual fault protocol has an absolute mAP or NDS regression greater
   than `0.0010` relative to C1. This value is the pre-registered definition of
   “no obvious single-protocol regression.”
4. Conservation violation count, unsupported-growth count and source-mass
   violation count are zero for both candidates.
5. Full tests pass; training and evaluation complete without NaN, Inf, OOM,
   RuntimeError or scene/runtime-state leakage into checkpoints.

If every condition passes, C0 is marked **provisional canonical baseline** and
may proceed to a separately authorized additional-seed confirmation. If any
condition fails, C0 must not replace legacy S2.2.
