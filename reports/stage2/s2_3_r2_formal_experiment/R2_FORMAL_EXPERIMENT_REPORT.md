# S2.3-R2 Formal Experiment Report

## Scope and provenance

This report uses only the declared mini development protocols: Clean,
camera_crash_5, camera_crash_10, Compound, and public `w2_t100`. No holdout,
private/hidden set, extra seed, 200-iteration run, teacher, S2.4 component, or
third candidate was accessed. Thresholds were frozen in
`PRE_REGISTERED_THRESHOLDS.md` before both 50-iteration runs.

## Candidate results

| candidate | Clean mAP/NDS | fault mean mAP/NDS | confirmed | precision | false recovery | wrong writes | w2 mean/max |
|---|---:|---:|---:|---:|---:|---:|---:|
| r2_a_zero | 0.424772/0.477030 | 0.406094/0.466285 | 0 | 0.00% | 51.61% | 0 | 6.667/10 |
| r2_b_zero | 0.424772/0.477030 | 0.406259/0.466628 | 4 | 100.00% | 51.61% | 0 | 6.500/12 |
| r2_a_50iter | 0.427985/0.478752 | 0.406705/0.466260 | 0 | 0.00% | 51.09% | 0 | 5.500/8 |
| r2_b_50iter | 0.427985/0.478752 | 0.406562/0.465821 | 5 | 100.00% | 51.09% | 0 | 5.833/8 |

The fixed B0 reference is Clean **0.424800/0.477000** and fault mean
**0.407233/0.467000**.

Historical reused comparisons (Clean; fault mean mAP/NDS) are: S2.2/B0
**0.424800/0.477000; 0.407233/0.467000**, B4 zero-shot
**0.424800/0.477000; 0.407167/0.467100**, B6 zero-shot
**0.424000/0.476400; 0.407000/0.466567**, B4 50iter
**0.425500/0.477200; 0.406800/0.466133**, and B6 50iter
**0.426600/0.477800; 0.405833/0.465467**.

Both R2 candidates use confirmation frames=2, pending max age=3, class
consistency=true, center/motion thresholds=2.0 m, minimum score=0.075,
minimum reliability=0.65, and pending memory write=false. R2-A disables
confirmation; R2-B enables it. All other model, optimizer, FP16, data, seed,
and training settings are identical.

## Pre-registered hard gates

- **r2_a_50iter**: clean=PASS, fault_mean=FAIL, protocol_breadth=FAIL, wrong_writes=PASS, false_confirmed=PASS, recovery=PASS, evidence=PASS; fault-protocol breadth=1/3.
- **r2_b_50iter**: clean=PASS, fault_mean=FAIL, protocol_breadth=FAIL, wrong_writes=PASS, false_confirmed=PASS, recovery=PASS, evidence=PASS; fault-protocol breadth=1/3.

Neither 50-iteration candidate passes the full performance gate. R2-A is the
better formal candidate because its fault mean is higher than R2-B and it
eliminates unconfirmed memory writes without confirmation delay, but it still
falls below B0 and passes only one of three fault protocols. R2-B sharply
reduces false formal writes, yet its stricter two-frame admission does not
directly recover any B0-missed object (confirmed-event delta=0) and loses more
Compound performance.

## GT, memory, and interval interpretation

The largest matched-GT improvement is
**r2_a_50iter/camera_crash_5/first_recovery**
(+1); the largest regression is
**r2_a_50iter/compound/post_fault**
(-11). See `per_interval_metrics.csv` for
pre-fault, in-fault, first-recovery, post-fault, and 1/3/5-frame windows.

Memory isolation is mechanically successful (`pre_confirmation_wrong_writes=0`
and confirmed/write agreement is exact), but the fault-average failure shows
that memory write was not the only bottleneck. The remaining primary bottleneck
is **candidate selection**: two-frame confirmation filters writes but does not
create new correct GT matches. Confirmation condition is secondary; loosening
it would trade precision for the original contamination failure.

## Engineering result

Full pytest: **141 passed, 7 warnings**. Disabled-path replay: **12/12 exact**,
each with 243 tensors and 81 box objects (`max_abs_diff=0`). Both smoke and
both 50-iteration runs completed without NaN, Inf, OOM, or RuntimeError.
Runtime pending buffers are non-persistent and absent from all inspected
checkpoints. Across formal traces, conservation/source-mass violations and
unsupported growth are all zero; see `evidence_summary.csv`.

## Decision

This is a pre-registered **Case C / negative screening result**: both R2-A and
R2-B fail the fault-performance gate. S2.2 remains the stable version. Do not
run 200 iterations or extra seeds for these candidates. Stop S2.3-R2 here; a
future task may reconsider candidate selection, but this task does not start
S2.4.
