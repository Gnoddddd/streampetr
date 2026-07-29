# S2.4 C0/C1 200-iteration Baseline Confirmation

## Decision

C0 does **not** pass the pre-registered provisional canonical-baseline gate.
It retains a small fault-average advantage, but it is below C1 on both Clean
metrics and has a Compound regression larger than the frozen `0.0010`
single-protocol tolerance.

Consequently:

- C0 is not marked provisional canonical baseline.
- C0 must not replace legacy `s2.2-stable`.
- No additional-seed confirmation is recommended from this result.
- Existing `s2.2-stable` tags are unchanged.
- This experiment does not authorize a dynamic matrix or any new method.

## Pre-registered gate

The gate was committed as `5e0db32` before either 200-iteration run:

1. C0 Clean mAP and NDS must each be at least C1.
2. C0 fault-average mAP and NDS must each be at least C1.
3. No fault protocol may regress by more than `0.0010` in mAP or NDS.
4. Conservation, unsupported-growth, source-mass and engineering checks must
   pass.

Observed gate:

| Gate | Result |
|---|---|
| Clean mAP non-inferior | FAIL |
| Clean NDS non-inferior | FAIL |
| Fault-average mAP non-inferior | PASS |
| Fault-average NDS non-inferior | PASS |
| Individual fault tolerance | FAIL |
| Engineering invariants | PASS |
| Overall | **FAIL** |

## Fairness

Both candidates used:

- the same Stage1 formal `iter_200.pth`, SHA256
  `9f8f4ab9361bb3a880abbdb605f93929886aa242d1d665ea84500a7abd331a16`;
- seed `2026`;
- identical data configuration/order, optimizer, LR, batch size, dynamic FP16,
  `max_norm=35` clipping, frozen parameters and 200-iteration runner.

Programmatic comparison of all frozen fairness fields returned `True`. The
only intended semantic difference was
`enable_correlation_discount=False/True`. Neither candidate resumed from a
50-iteration checkpoint.

## Training

| Metric | C0 no-discount | C1 fixed-discount |
|---|---:|---:|
| iterations | 200 | 200 |
| first loss | 22.3750 | 22.3750 |
| final loss | 16.7909 | 16.5846 |
| mean loss | 20.4177 | 20.3581 |
| max loss | 40.4662 | 40.4662 |
| max grad norm | 17.4290 | 17.4544 |
| logged peak GPU memory | 555 MB | 555 MB |
| train residual abs max | 1.907349e-6 | 3.814697e-6 |
| conservation violations | 0 | 0 |
| unsupported growth | 0 | 0 |
| source-mass violations | 0 | 0 |

Both runs exited `0`, saved `iter_200.pth`, and had no NaN, Inf, OOM or
RuntimeError.

## Formal protocol metrics

The deltas below are C0 minus C1.

| Protocol | C0 mAP/NDS | C1 mAP/NDS | C0-C1 mAP/NDS |
|---|---|---|---|
| Clean | .424837/.476760 | .427993/.479573 | -.003156/-.002813 |
| Crash5 | .418599/.471713 | .418227/.471671 | +.000372/+.000042 |
| Crash10 | .413006/.471748 | .408842/.468460 | +.004165/+.003288 |
| Compound | .388993/.455022 | .392175/.457286 | -.003181/-.002264 |
| Fault average | .406866/.466161 | .406415/.465805 | +.000452/+.000355 |

C0 therefore improves Crash10 and the arithmetic fault average, but the Clean
and Compound regressions are larger and fail the frozen joint gate.

## N_eff and action/write behavior

C0 bypassed the matrix and N_eff computation; its diagnostic effective count
is exactly `1` for every query.

C1 N_eff means were:

| Protocol | Mean | p95 | Max | Zero ratio |
|---|---:|---:|---:|---:|
| Clean | 1.070317 | 1.481143 | 1.481728 | .000535 |
| Crash5 | 1.043300 | 1.481133 | 1.481720 | .020878 |
| Crash10 | 1.016799 | 1.481122 | 1.481730 | .040466 |
| Compound | 1.015911 | 1.481031 | 1.481740 | .040576 |

Across all four protocols, C1 increased keep ratio from C0 `.303495` to
`.413700` and write ratio from `.309818` to `.420031`. Recover ratios remained
essentially unchanged (`.006324` versus `.006331`). As in the 50-iteration
screen, the fixed path mainly causes more keep/write decisions.

## Evaluation invariants

Across both candidates and all four protocols:

- conservation residual absolute maximum was at most `2.861023e-6`;
- conservation violation count was `0`;
- unsupported-growth count was `0`;
- source-mass violation count was `0`.

The full test suite passed before training: `95 passed, 7 warnings in 8.17s`.

## Checkpoint audit

Both 200-iteration checkpoints contain 629 model state keys and one persistent
fixed-correlation configuration buffer. Each has:

- runtime ledger state keys: `0`;
- switch state keys: `0`;
- scene/batch/query state leakage: none;
- checkpoint audit status: safe.

The persistent matrix is model configuration retained for compatibility, not
scene runtime state.

## tmux execution

All long work used `scripts/run_experiment_tmux.sh`:

- `outputs/stage2/s2_4_baseline_confirmation/pytest/`
- `outputs/stage2/s2_4_baseline_confirmation/c0_200/`
- `outputs/stage2/s2_4_baseline_confirmation/c1_200/`
- `outputs/stage2/s2_4_baseline_confirmation/eval_tmux/`
- `outputs/stage2/s2_4_baseline_confirmation/analysis/`

Every recorded exit status is `0`; no duplicate session was launched.
