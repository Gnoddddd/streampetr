# CARE-3D P0 Cross-Severity Transfer Preregistration

## Status and purpose

The main CARE-3D counterfactual P0 is already frozen as
`GO_CARE3D_COUNTERFACTUAL_P0` on Blur 0.9, Camera Crash, and Dark 0.9.  This
follow-up is confirmatory only.  It asks whether the already-trained
object-level vulnerability predictor transfers to *lighter* camera corruption
without retraining, recalibration, threshold tuning, seed selection, or feature
changes.

This experiment does **not** reopen the main P0 decision.  Its purpose is to
strengthen or weaken the claim that the learned signal reflects conditional
object susceptibility rather than memorization of one severe corruption level.

## Frozen predictor

The three main-P0 seed checkpoints are reused exactly:

- seed `42`;
- seed `2027`;
- seed `2028`.

The exporter records SHA256 for all three checkpoints before the first
cross-severity forward.  Any later checkpoint change is a hard failure.

No parameter update is allowed.  No validation split is used.  No calibration
or threshold fitting is allowed.  All three frozen seeds are reported.

## Frozen cohort

Only the already-frozen `probe_test` scenes from main P0 are used:

- 132 scenes;
- the same scene-manifest SHA256 as main P0;
- the exact same clean-TP object cohort and sample order per scene.

The lighter-severity exporter reruns the clean trajectory only to obtain the
identical one-step counterfactual branch state.  For every scene it then
requires:

1. exact `sample_id` order equality with the main P0 export;
2. bitwise equality of all CARE predictor inputs with the main P0 export:
   - `object_features`;
   - `temporal_features`;
   - `decision_features`;
   - `camera_support`;
   - `camera_quality`.

If either cohort alignment or predictor-input identity fails, the scene is not
accepted.

## Transfer interventions

Main P0 trained/evaluated the relevant heads at severity `0.9`.  The frozen
transfer interventions are:

- Blur: `protocols/presets/motion_blur_back_10f_s03.json`;
- Dark: `protocols/presets/dark_back_10f_s03.json`.

Both apply severity `0.3` to `CAM_BACK` for frames 3--12.  Camera Crash is not
assigned an artificial continuous severity and is excluded from this transfer
experiment.

As in main P0, every target-frame branch starts from the same clean post-state
`H_t`, lasts exactly one step, and is then discarded.  Fault history remains
prohibited.

## Labels

For the clean `t+1` TP, the clean matched query/class `(q,c)` remains fixed.
For lighter-severity protocol `p`:

```text
E_clean(i)   = sigmoid(logit_clean[q,c])
E_fault(i,p) = sigmoid(logit_fault,p[q,c])
drop(i,p)    = max(E_clean(i) - E_fault(i,p), 0)
```

The same flattened query/class Top-K rule (`K=100`) is used:

```text
cross_topk(i,p) = 1
  iff clean (q,c) is in Top-K and the same (q,c) leaves Top-K under p.
```

`tp_to_fn` is retained as an auxiliary outcome but is not the primary transfer
gate target.

## Predictor mapping

The frozen three-output main-P0 predictor is not modified:

- `blur_s03` uses the existing `blur_back` output dimension (index 0);
- `dark_s03` uses the existing `dark_back` output dimension (index 2).

No new output head is created.

## Required metrics

For every frozen seed and both transfer protocols, report on all 132 probe-test
scenes:

Evidence-drop prediction:

- Spearman correlation;
- Pearson correlation;
- MAE;
- actual-drop difference between predicted top and bottom vulnerability deciles.

Boundary prediction:

- AUROC;
- AUPRC;
- positive base rate;
- AUPRC minus base rate;
- Brier score;
- ECE-10.

Stability:

- 5,000 scene-cluster bootstrap replicates;
- 5,000 instance/trajectory-cluster bootstrap replicates.

## Frozen transfer rule

The transfer experiment is `PASS_CARE3D_P0_CROSS_SEVERITY` only if **both**
`blur_s03` and `dark_s03` pass for **all three** frozen seeds.

A seed/protocol pair passes only when the same main-P0 within-test stability
criteria are satisfied:

1. point Spearman > 0;
2. scene- and instance-cluster Spearman 95% CI lower bounds > 0;
3. point top-vs-bottom decile actual-drop separation > 0;
4. scene- and instance-cluster separation CI lower bounds > 0;
5. point boundary AUROC >= 0.65;
6. scene- and instance-cluster AUROC CI lower bounds > 0.50;
7. scene- and instance-cluster `(AUPRC - base rate)` CI lower bounds > 0.

There is intentionally no validation/test direction condition here because no
lighter-severity validation set is opened and no parameter is selected from
lighter-severity data.

## Engineering gate

Before the 132-scene transfer run, one already-excluded engineering scene is
used.  It must satisfy both:

- passive-hook clean equivalence;
- exact clean-anchor predictor-input identity against its main-P0 export.

Only then is formal probe-test extraction eligible.

## Interpretation

A pass supports the narrower statement that the frozen severe-fault CARE P0
predictor retains object-specific vulnerability ranking and boundary-risk
information under a lighter Blur/Dark intervention on the same camera.

A pass does not establish cross-camera, cross-architecture, unseen-fault, or
zero-shot generality.  Those remain separate later experiments.
