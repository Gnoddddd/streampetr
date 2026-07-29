# S2.5 Pre-registered Thresholds

Baseline: `s2.2-stable` / `9958366`, with inherited legacy fixed-correlation
semantics explicitly acknowledged.

## Engineering gates

- `enable_soft_write_gate=False` is tensor-exact to S2.2:
  `max_abs_diff=0` for classification, boxes, final predictions, alpha/beta,
  positive/negative/source evidence, action, write mask, Top-K, propagated
  query and temporal memory.
- Disabled mode bypasses all S2.5 scale calculation/application.
- Action remains unchanged by the S2.5 module.
- Main scales are exactly KEEP `1.0`, RECOVER `0.75`, DEFER `0.05`.
- `write_scale` is finite, in `[0,1]`, correctly aligned after Top-K, and safe
  for scene reset, batch/query/Top-K changes and CPU/GPU FP16.
- Soft memory state and committed ledger state remain aligned.
- No scene runtime state enters ordinary `state_dict` or checkpoint.
- conservation residual abs max `<=1e-5`, violations `=0`.
- source-mass residual abs max `<=1e-5`, violations `=0`.
- unsupported growth count `=0`.
- No NaN, Inf, OOM, RuntimeError or state leakage.

## Performance gates

Use the Stage2 mini Go/No-Go gates without post-result relaxation:

- Clean mAP and NDS each decline by at most `0.003` versus S2.2.
- Crash5 mAP/NDS are each not below S2.2.
- Crash10 mAP/NDS are each not below S2.2.
- Compound mAP/NDS are each not below S2.2.
- Arithmetic fault-average mAP and NDS are each not below S2.2.
- public `w2_t100` recovery mean delay `<=5.0` frames and max delay `<=7`.
- no increase in recovery lock-up or cross-scene memory contamination.

## Entry sequence

1. **Engineering/unit**
   - full pytest passes;
   - disabled-path exact replay passes;
   - synthetic KEEP/RECOVER/DEFER scale and memory-write tests pass;
   - checkpoint and FP16 gates pass.
2. **Smoke**
   - only the main candidate and `no_recover_discount` ablation;
   - enter only after all engineering gates;
   - exit on finite forward/backward, correct scales, safe checkpoint and no
     invariant violation.
3. **50 iter**
   - each candidate enters only after its smoke passes;
   - evaluate Clean, Crash5, Crash10, Compound and public recovery;
   - a candidate must pass every engineering and performance gate.
4. **200 iter**
   - only one frozen 50-iter winner may enter;
   - all gates must remain satisfied with no scale/threshold changes.
5. **Holdout**
   - forbidden until the 200-iter winner also passes a separately authorized
     multi-seed confirmation;
   - method, scales, protocols and selection must be frozen;
   - requires explicit new authorization and one-shot reporting;
   - holdout may not select scales or candidates.

Any failed stage stops that candidate. Do not add a third candidate or relax a
threshold after observing results.
