# S2.3-R2 Formal Experiment: Pre-registered thresholds

Recorded before implementing R2-A/R2-B and before running either new
50-iteration experiment.

## Method parameters

Both candidates inherit the B4 motion-only, evidence-budgeted reacquisition
configuration and use the same S2.2 50-iteration checkpoint.

| Parameter | Fixed value | Rationale fixed from existing development traces |
|---|---:|---|
| `enable_memory_isolation` | `True` | Directly tests H1. |
| `enable_two_phase_reacquisition` | R2-A `False`; R2-B `True` | The only intended difference between candidates. |
| `confirmation_frames` | `2` | Required minimum and matches the public `w2` recovery window. |
| `pending_max_age` | `3` frames | Matches the existing Recover policy horizon and avoids long-lived identities. |
| `class_consistency_required` | `True` | A confirmed identity must retain its predicted class. |
| `center_distance_threshold` | `2.0 m` | Same as the existing nuScenes-mini GT alignment audit; all prior true-positive triggers were below 1.95 m from GT. |
| `motion_distance_threshold` | `2.0 m` | Allows normal 0.5 s motion while rejecting discontinuous identity jumps. |
| `minimum_confirmation_score` | `0.075` | Prior B4 true-trigger 10th percentile was 0.0747; this retains about 90% of known-correct events without selecting a result-dependent threshold. |
| `minimum_confirmation_reliability` | `0.65` | Prior correct-trigger minimum was 0.6495; reliability did not separate TP/FP, so it is only a numerical floor. |
| `allow_pending_memory_write` | `False` | Mandatory safety default. |

No GT, protocol identity, teacher output, new loss, new network, or S2.4
feature may participate in the runtime confirmation decision.

## Hard engineering gates

- Full pytest suite passes.
- Disabled R2 path is tensorwise identical to commit `0bdaf1f`.
- Conservation and source-mass violation counts are zero.
- Unsupported growth is zero.
- Maximum absolute conservation residual is at most `1e-5`.
- Pending runtime state is absent from ordinary `state_dict` and checkpoints.
- No NaN, Inf, OOM, or RuntimeError.

## Performance gates fixed before results

The reference is B0: Clean `0.4248/0.4770` mAP/NDS and fault mean
`0.407233/0.467000`. Existing short-run/ablation variation is approximately
`0.0006-0.0010` on Clean and up to `0.0012` on fault mean.

- Clean mAP and NDS may not fall by more than `0.0010` versus B0.
- Fault-mean mAP and NDS must each be at least B0 minus `0.0001` (rounding
  allowance only).
- At least two of the three fault protocols must be non-degraded within
  `0.0001` in both mAP and NDS; a single-protocol gain is insufficient.
- Wrong formal-memory writes must fall by at least 50% from the diagnosed
  `32` per-candidate baseline, so at most `16` are allowed.
- False-confirmed rate must be at most 25%, versus the prior ordinary
  reacquisition false rate of about 51.6%.
- Public `w2_t100` mean recovery delay may not exceed B0 by more than
  2 frames, and maximum delay may not exceed B0 by more than 3 frames.
- New GT matches are reported but are not made a post-hoc pass requirement.

These values are frozen for the two allowed 50-iteration candidates.
