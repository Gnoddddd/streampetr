# S2.3 N1 performance-collapse root cause

All comparisons use the same S2.2 checkpoint, identical 81-frame protocol
order, and exact `(sample_idx, scene_token, frame_idx)` alignment.

## Answers

1. **Clean positive evidence is systematically suppressed.** Mean
   `actual_positive/base_positive` is 0.6077 for zero-shot N1
   and 0.5848 after 50 iterations. Training does not restore
   the S2.2 evidence budget.
2. **Negative evidence also changes.** Mean `actual_negative/base_negative`
   is 0.5944 zero-shot and 0.6293 debug50.
   The legacy multiplicative strategy therefore changes both directions,
   rather than isolating recovery-positive evidence.
3. **Lower strength changes policy and writeback.** Zero-shot transitions:
   {'keep_to_recover': 688, 'keep_to_defer': 76658, 'high_to_low_score_scale': 76658, 'write_mask_lost': 77942}. Debug50 transitions:
   {'keep_to_recover': 738, 'keep_to_defer': 72859, 'high_to_low_score_scale': 72859, 'write_mask_lost': 74095}.
4. **Both same-frame scaling and later propagation contribute.** The
   innovation factor changes alpha/beta before policy/score scaling in the
   current frame; the resulting action/write-mask losses then reduce valid
   propagated memory in subsequent frames. `high_to_low_score_scale` is a
   policy-scale proxy because raw detector logits are not present in legacy
   traces.
5. **Recovery signals exist at frame 13.**
- n1_zero_shot crash_10f: source mean/p90=0.8152/1.0000, time mean/p90=0.0376/0.0000
- n1_debug50 crash_10f: source mean/p90=0.8180/1.0000, time mean/p90=0.0371/0.0000
- n1_zero_shot crash_10f: source mean/p90=0.7774/1.0000, time mean/p90=0.0062/0.0000
- n1_debug50 crash_10f: source mean/p90=0.7759/1.0000, time mean/p90=0.0033/0.0000
- n1_zero_shot compound_10f: source mean/p90=0.8092/1.0000, time mean/p90=0.0323/0.0000
- n1_debug50 compound_10f: source mean/p90=0.8084/1.0000, time mean/p90=0.0317/0.0000
- n1_zero_shot compound_10f: source mean/p90=0.7646/1.0000, time mean/p90=0.0030/0.0000
- n1_debug50 compound_10f: source mean/p90=0.7676/1.0000, time mean/p90=0.0040/0.0000
6. **Why 50 iterations fail:** the detector is initialized from S2.2, but
   legacy N1 structurally multiplies every ordinary evidence increment by an
   innovation gain. The short run only updates the permitted lightweight
   branches and cannot reconstruct the removed evidence budget; action and
   memory feedback amplify the persistent suppression.

## Implication

The rescue must preserve S2.2 positive and negative base evidence tensor
exactly during continuous observation. Innovation may only add a one-shot,
bounded positive bonus after a verified observation gap. It must not scale
Clean base evidence or negative evidence.
