# CARE-3D P2-A2-R1 Disagreement-Gated Identity Arbiter

## Frozen scope

R1 arbitrates only the two already-frozen online identity hypotheses: the
P2-A0 query `A` and explicit memory-lineage query `L`. It does not create a
third candidate. Both heads use exactly `MODEL_FEATURE_COLUMNS` and
`finite_model_matrix()` from `analysis/care3d_p2a2_r1_features.py`.

The two heads are independently fitted `StandardScaler` plus binary
`LogisticRegression` models with `solver="lbfgs"`, `penalty="l2"`, `C=1.0`,
`max_iter=2000`, `class_weight="balanced"`, and `random_state=314159`.
Preference is trained only on decisive disagreement rows and predicts
`lineage_wins`. Ambiguity is trained on every disagreement row and predicts
`both_wrong`. Protocol is never a model input and all protocols share both
model definitions and one threshold pair.

## Runtime

Agreement is an exact bypass: when `A == L`, the selected query is `A`, the
decision is `AGREEMENT`, neither head is called, and defer is impossible.
For disagreement, runtime first applies ambiguity and then preference:

```text
if p_both_wrong >= tau_defer: DEFER, selected_query = -1
elif p_lineage >= tau_preference: LINEAGE, selected_query = L
else: P2A0, selected_query = A
```

`FrozenIdentityArbiter` replays scaler and logistic parameters in NumPy and
does not use sklearn for deployment inference.

## Nested development evaluation

All 419 `probe_train` manifest scenes, including the legal zero-row scene, are
assigned to an outer 5-fold `GroupKFold` by `scene_token`. Each outer-training
cohort receives an independent inner 4-fold scene-grouped OOF prediction for
both heads. Inner OOF predictions and the corresponding outer-training full
population are the only inputs to threshold selection. The frozen outer-test
cohort is not accepted by the selector and is used only after thresholds are
fixed and both heads are refit on all outer-training disagreement data.

Preference thresholds are 0.05 through 0.95 in 0.01 steps plus 1.01. Defer
thresholds are 0.50 through 0.99 in 0.01 steps plus 1.01. A pair is feasible
only when every protocol has unmatched rate at most 0.01 and non-negative
exact gain. Feasible pairs are ordered by: minimum maximum protocol wrong
rate; maximum minimum protocol exact gain; minimum mean wrong rate; maximum
mean exact gain; minimum maximum unmatched rate; minimum mean unmatched rate;
maximum defer threshold; minimum distance of preference threshold from 0.5;
then maximum preference threshold.

R0's 275,985 rows are retained. The key `(scene_token, instance_token,
anchor_frame_idx, target_frame_idx, protocol)` must be unique, and the R0
disagreement key set and three identity columns must exactly match all 34,342
R1-F0 feature rows. Agreement rows keep NaN head probabilities and receive a
complete bypass decision.

## Metrics, bootstrap, and Gate

Point metrics are evaluated separately for `blur_back`, `crash_back`, and
`dark_back` on the full population. Repair accounting distinguishes P2-A0
wrong repaired, correct broken to wrong, correct broken to unmatched, wrong to
unmatched, and wrong remaining wrong.

After nested predictions are frozen, 5,000 paired percentile bootstrap
replicates use seed 314159. Scene bootstrap samples from the complete 419-scene
manifest, including zero-row scenes. Instance-trajectory bootstrap samples
`(scene_token, instance_token)` clusters within each protocol. Baseline and
arbiter always share the same sampled cluster multiplicities.

The final development Gate requires all three protocols, with no Crash
exception, to pass G1--G10: wrong rate at most 0.10, unmatched rate at most
0.01, positive exact gain, positive scene and instance exact-gain CI lower
bounds, negative wrong-rate delta, negative scene and instance wrong-delta CI
upper bounds, exact agreement invariance, and all leakage/search discipline
flags remaining false.

## Entrypoint boundary

`scripts/analyze_care3d_p2a2_r1_arbiter.py` performs source validation before
formal nested evaluation. It reads only frozen `probe_train` R0/R1-F0 reports.
`scripts/freeze_care3d_p2a2_r1_arbiter.py` is a separate future action and
refuses to fit unless the nested development decision is GO. The freeze step
selects final thresholds from fresh five-fold scene-grouped development OOF
probabilities, fits both heads on all development disagreement rows, and emits
explicit scaler/logistic arrays in `frozen_arbiter.json`.

Neither entrypoint accesses `probe_val` or `probe_test`; both audit flags remain
false. Formal nested evaluation, bootstrap, Gate evaluation, and final freeze
are operational steps and are not run as part of source implementation review.
