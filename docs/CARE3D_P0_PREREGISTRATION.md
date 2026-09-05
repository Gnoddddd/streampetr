# CARE-3D P0 Preregistration

## Scope

This document freezes the formal P0 experiment for CARE-3D
(Counterfactual Adaptive Redundant Evidence Routing).  P0 asks whether a
currently correct object contains enough information to predict its
**conditional susceptibility** to a future camera-fault intervention.

P0 is not a detector-training experiment and is not allowed to modify baseline
predictions.  The official StreamPETR R50 90e detector is frozen throughout.
Routing remains disabled.

## Frozen causal question

For a clean anchor object `i` at frame `t`, define one clean continuation and
three one-step counterfactual interventions at `t+1`:

```text
clean post-state H_t
  |-- clean(t+1)
  |-- blur_back(t+1)
  |-- crash_back(t+1)
  `-- dark_back(t+1)
```

Every branch starts from the exact same `H_t`.  Fault branches are discarded
after one forward.  The main clean continuation becomes the next anchor state.
Fault history is therefore prohibited in P0.

The input `Z_i^t` is built only from the clean anchor frame.  No `t+1` tensor,
fault image, fault output, fault-derived quality, future GT attribute, or
outcome label may enter the predictor.

## Frozen cohort

The formal scene split is inherited byte-for-byte from the validated Stage5
prospective experiment:

- `probe_train`: 419 scenes;
- `probe_val`: 133 scenes;
- `probe_test`: 132 scenes;
- 16 earlier mechanism-discovery scenes remain excluded.

The expected Stage5 scene-manifest SHA256 is:

`83637205c930611ccdc6879eb233f72a9b0a5997248f4b5b5edf3242182d6da1`

The 16-scene discovery-list SHA256 is:

`7bbd389f8ec1f02e75d3d8a7feb773f109fdf24d5824b65db89fc7544e425fb3`

An object is eligible at anchor `t` only when:

1. it is a deployed clean TP at frame `t`;
2. the same instance exists at `t+1`;
3. it is again a deployed clean TP at `t+1`.

GT is used only to define/evaluate the frozen cohort and matching.  GT is never
part of the CARE predictor input.

## Frozen detector and sources

- checkpoint: `checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth`
- checkpoint SHA256: `e6323ae5c31adf1eedd46d6dd4fd3c73d95aa26f18cc8aa23c196494b7de3451`
- detector config: `configs/full_nuscenes/stream_petr_r50_90e_ctep_train_audit.py`
- config SHA256: `927ba2518a4ca460d2f7f6b3ba74dab620ac8e2995ee7e9aadbcbebf2d7c64a6`
- train-info SHA256: `dc5e5e611badbdb1c0270a3583e022cf14a9af7b3ff8f02370434b8ec50b493d`
- StreamPETR commit: `95f64702306ccdb7a78889578b2a55b5deb35b2a`

Main protocols:

- `blur_back`: `motion_blur_back_10f_s09.json`
- `crash_back`: `camera_crash_back_10f.json`
- `dark_back`: `dark_back_10f_s09.json`

Any source-hash change is a hard failure.

## Counterfactual labels

For the clean `t+1` TP, let `(q,c)` be its clean matched query and class.  The
same `(q,c)` is read in every fault branch.  Fault-side query reassignment is
not allowed when measuring target evidence.

```text
E_clean(i)   = sigmoid(logit_clean[q,c])
E_fault(i,p) = sigmoid(logit_fault,p[q,c])
drop(i,p)    = max(E_clean(i) - E_fault(i,p), 0)
```

For the flattened query-class deployment ranking with `K=100`:

```text
cross_topk(i,p) = 1
  iff the clean (q,c) is in Top-K and the same (q,c) is outside Top-K under p.
```

A second auxiliary label is retained:

```text
tp_to_fn(i,p) = 1
  iff clean(t+1) is a deployed TP and fault_p(t+1) is a deployed FN.
```

The primary CARE boundary target is `cross_topk`; `tp_to_fn` is an auxiliary
end-task failure label.

## Predictor input

The canonical clean-anchor input is:

- `object_features`: `final_decoder_pre_cls_query`, 256-D;
- `temporal_features`: `decoder_layer5_temporal_self_attn_output`, 256-D;
- `decision_features`: the existing 21-D deployed-observable vector;
- `camera_support`: six-camera geometry support derived from the current
  predicted 3D center and calibration only;
- `camera_quality`: all ones in main P0 because the anchor is clean.

The two representation taps above are frozen before this P0 because the Stage5
prospective gate independently showed stable signal across Blur, Crash and Dark.
This prior result is motivation only; its `y_tp_to_fn` label is not reused as
CARE supervision.

## Model and training

Only `CARE3DStateEncoder` and `CounterfactualVulnerabilityHead` are trained.
StreamPETR is never in the optimizer and all detector parameters must have
`requires_grad=False` during export.

Frozen initial dimensions:

```text
object_dim=256
num_cameras=6
hidden_dim=256
state_dim=128
decision_dim=21
use_temporal=True
num_protocols=3
dropout=0.0
```

Seeds: `42`, `2027`, `2028`.

`probe_train` is the only fit split.  `probe_val` selects the best epoch.
`probe_test` is not read by the training script and is opened only after the
seed checkpoint has been frozen by validation loss.

Boundary class imbalance is handled with a protocol-specific positive weight
computed **only from `probe_train`** as `N_negative / N_positive`.  Neither
`probe_val` nor `probe_test` may contribute to this weight.  A protocol whose
`probe_train` boundary labels do not contain both classes is a hard failure.
The same train-derived weights are used when computing validation loss for
early stopping/model selection.

No hyperparameter, class weight, threshold, or seed may be chosen from
probe-test performance.

## Required evaluation

For each protocol and each seed:

Evidence-drop regression:

- Spearman correlation;
- Pearson correlation;
- MAE;
- actual-drop difference between predicted top and bottom vulnerability deciles.

Boundary prediction:

- AUROC;
- AUPRC;
- positive base rate;
- Brier score;
- ECE-10.

Uncertainty/stability:

- 5,000 scene-cluster bootstrap replicates;
- 5,000 instance/trajectory-cluster bootstrap replicates;
- probe-val and probe-test direction agreement;
- all three seeds reported separately.

## Hard GO / NO-GO rule

A protocol passes only when the result is stable across the three frozen seeds
and the following conditions hold:

1. evidence-drop Spearman is positive and both scene-cluster and
   instance/trajectory-cluster bootstrap lower confidence bounds are above zero;
2. predicted top-vulnerability objects have larger actual drop than the bottom
   group and both clustered lower bounds are above zero;
3. boundary-crossing AUROC is at least `0.65` on the frozen test split;
4. both scene-cluster and instance/trajectory-cluster AUROC lower confidence
   bounds are above `0.50`;
5. AUPRC is above the protocol positive base rate and both clustered lower
   confidence bounds of `AUPRC - positive_base_rate` are above zero;
6. validation and test directions agree;
7. disabled/clean detector identity checks pass.

A protocol must satisfy the full rule for **all three frozen seeds**.  Main P0
is `GO` only if at least two qualitatively different fault families pass.
Otherwise P1 routing stays locked.

## Cross-severity transfer

Cross-severity is locked until the main P0 decision is frozen.  If P0 is GO,
a lighter Blur and a lighter Dark protocol may be evaluated with the already
trained predictor.  No retraining is allowed for transfer.  Crash is not given
an artificial continuous severity.

## Engineering smoke and formal execution

Before formal extraction, one of the 16 excluded discovery scenes may be used
for an engineering-only smoke test.  It must write under an engineering output
folder and must never enter the 684-scene formal manifest or any metric.

Formal extraction is resume-safe per scene.  Ctrl+C/SIGTERM requests a stop
after the current scene has been atomically saved.  Large CSV/NPZ outputs stay
under `reports/care3d/p0_counterfactual_vulnerability/` and are not committed.

P1 `SparseEvidenceRouter` remains disabled until the P0 decision is GO.
