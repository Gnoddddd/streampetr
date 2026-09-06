# CARE-3D P1 Preregistration: Sparse Evidence Router

## Decision being tested

P0 established that a still-correct target's clean object state contains stable,
protocol-conditioned information about its one-step counterfactual evidence
collapse. The confirmatory severity-transfer experiment further showed that
this vulnerability ordering transfers from Blur/Dark severity 0.9 to 0.3
without retraining or recalibration.

P1 asks a different causal question:

> When P0 says an object is vulnerable, can a sparse backup-evidence route
> improve the actual StreamPETR classification decision under a camera fault,
> while preserving retained objects, clean behavior and the false-positive
> budget?

P1 is an activation gate, not yet the final deployment system.

## Frozen upstream state

P1 is allowed to start only if:

- `reports/care3d/p0_counterfactual_vulnerability/decision.json` is
  `GO_CARE3D_COUNTERFACTUAL_P0`;
- `reports/care3d/p0_cross_severity/decision.json` is
  `PASS_CARE3D_P0_CROSS_SEVERITY`;
- all P0 seeds `42`, `2027`, `2028` have frozen checkpoints with
  `probe_test_read=false`, routing disabled and zero detector parameters in the
  optimizer;
- the 684-scene manifest remains the frozen `419 / 133 / 132`
  `probe_train / probe_val / probe_test` split.

The official full-nuScenes StreamPETR 90-epoch checkpoint remains frozen.
No detector parameter is optimized in P1.

## Experimental unit and limitation

The paired unit remains the P0 clean-TP object trajectory. At `t+1`, Clean and
Fault branches start from the exact same clean post-state `H_t`; only the Clean
branch advances the trajectory.

For this P1 activation experiment, the target is evaluated at the same frozen
clean `t+1` matched `(query, class)` used by P0. This is an intentional causal
experimental unit that avoids fault-side query reassignment confounding.
It is **not** claimed to be a deployment-ready identity association mechanism.
A later system-level stage must replace this oracle pairing with an online
object/query association rule before any end-to-end deployment claim.

GT identity and GT geometry are never router inputs. GT is used only for cohort
construction and evaluation, exactly as in P0.

## Implementation amendment: shared target-query collisions

This amendment was frozen on 2026-09-06 after the formal train/val supervision
export encountered a structural ambiguity and **before any P1 router training or
probe-test evaluation**. No P1 outcome metric, validation score, test result or
gate threshold was inspected or changed to define this rule.

The P0 cohort is constructed from flattened `(query, class)` deployment
predictions. Consequently, in a small number of frames, two or more P0 object
rows can share the same clean `t+1` detector query while referring to different
object/class matches. P0 can represent those rows independently because its
label is read from a fixed `(query, class)` entry. P1 cannot apply two different
object-specific routed query vectors to the same detector query in one frame:
the detector has only one query state and one full class vector at that index.
Allowing sequential writes would create arbitrary last-write semantics.

The frozen, outcome-blind P1 eligibility rule is therefore:

- form groups by `(scene_token, target_frame_idx, target_clean_query_index)`;
- if a group has multiplicity `1`, the row is P1-eligible;
- if a group has multiplicity `>1`, **exclude every row in that collision
  group** from P1 object-level supervision and object-level formal evaluation;
- never select one colliding row using class, score, evidence drop,
  `cross_topk`, `tp_to_fn`, vulnerability prediction, future outcome or GT
  geometry;
- apply exactly the same rule to `probe_train`, `probe_val` and the still-locked
  `probe_test` split;
- in full-frame deployed TP/FP evaluation, excluded collision queries remain at
  their unmodified Fault baseline values rather than being removed from the
  detector output.

The frozen implementation identifier is
`exclude_all_rows_in_shared_target_query_frame`. Every new P1 scene completion
marker records the original P0 row count, eligible row count, excluded row count
and number of collision groups. Pre-amendment P1 supervision markers do not
contain this identifier and are therefore invalidated and re-extracted. The
engineering smoke must also be rerun under this policy.

This amendment changes only the technical eligibility of an intervention whose
unit was otherwise undefined. It does not change the P0 cohort definition, P1
source bank, loss, optimizer, seeds, train/val/test split, test lock, bootstrap
procedure or any Go/No-Go threshold.

## Frozen source bank

The failed `CAM_BACK` source is never offered to the router. Each object has
exactly three candidate backup sources:

1. `CAM_BACK_LEFT`: current `t+1` Fault FPN-P0 token, bilinearly sampled at the
   fixed target query's **predicted** 3D center when that projection is in view;
2. `CAM_BACK_RIGHT`: same construction;
3. `TEMPORAL_ANCHOR`: the target's clean `t` final-decoder pre-classification
   query already used by the frozen P0 predictor.

Camera source reliability is binary geometric visibility from the detector's
predicted center and transformed `lidar2img` matrices. The temporal source is
always available. The router may select at most two sources (`top_k=2`).

The source choice is therefore local, geometry-constrained and source-aware.
It does not reconstruct the failed camera.

## Frozen vulnerability control signal

For each P1 seed, the corresponding frozen P0 seed checkpoint is loaded.
Only clean-anchor P0 inputs are provided to it. The active protocol's frozen
P0 boundary-crossing probability gates the magnitude of the P1 residual.
No P1 label, future clean representation, fault outcome or GT information is
used to compute this risk gate.

## Intervention location

The main P1 intervention is deliberately narrow:

- input object state: the Fault `t+1` `final_decoder_pre_cls_query`;
- routed residual: `SparseEvidenceRouter`, zero-initialized residual scale;
- risk gate: frozen P0 protocol-specific crossing probability;
- output: routed final decoder query;
- the frozen final StreamPETR classification branch is replayed on the routed
  query;
- the Fault box-regression output is left unchanged.

This isolates the already-established target-score-collapse mechanism. P1 does
not claim geometry repair.

Clean inference is an explicit bypass. Therefore the P1 code path must be
bitwise identity on Clean when routing is disabled/no fault is active.

## Training protocol

Only P1 router parameters are trainable. StreamPETR and P0 parameters are
frozen and excluded from the optimizer.

Three fixed seeds are used: `42`, `2027`, `2028`.

- fit: `probe_train` only;
- early stopping/model selection: `probe_val` only;
- `probe_test`: physically locked until all three P1 checkpoints are frozen.

Training sampling is balanced over `(protocol, cross_topk label)` using counts
computed from `probe_train` only. Validation is evaluated at its natural class
frequency.

The frozen loss is:

- positive clean-target score restoration, weight `1.0`;
- positive Top-K boundary hinge, weight `1.0`, score margin `0.01`;
- positive clean-query Smooth-L1 restoration, weight `0.25`;
- retained-object full-class logit drift, weight `0.5`;
- positive non-target logit-increase penalty, weight `0.25`.

No P1 hyperparameter may be changed after probe-test extraction/evaluation is
unlocked.

## Engineering smoke gate

Before formal train/val extraction, one frozen discovery scene must satisfy:

1. passive query capture versus unhooked B0 is exact;
2. P0 sample IDs and clean-anchor predictor inputs are identical to the frozen
   P0 source after applying the frozen outcome-blind query-collision eligibility
   rule;
3. recomputed clean/fault same-query scores agree with the P0 labels;
4. the source bank has the frozen three-source layout;
5. the failed `CAM_BACK` is absent;
6. every row has the temporal backup source available;
7. the frozen P0 seed-42 checkpoint accepts the exact single-object input
   interface used by P1 and returns finite `[B,1,3]` vulnerability/crossing
   outputs;
8. saved clean/fault final-query tensors replay the frozen StreamPETR final
   classifier with maximum target-score absolute error <= `5e-4`;
9. a newly initialized P1 router is exact identity because its residual scale is
   zero;
10. the explicit clean bypass is exact identity after arbitrary router weights.

Failure locks formal P1 extraction. Formal train/val export is invoked through a
wrapper that requires the completed smoke-gate JSON to contain `passed=true`.

## Formal test outcomes

The formal 132-scene `probe_test` evaluation patches the full Fault
classification tensor at each P1-eligible cohort object's frozen query. The full
routed class vector is substituted, not only the GT class, so new class
competition and FP inflation remain observable. Fault boxes are unchanged.
Shared-query collision groups excluded by the frozen amendment remain unpatched
in the full-frame Fault baseline and are not part of object-level recovery/no-
harm denominators.

For each protocol and seed we report:

- base fault-lost objects and `lost_recovery_rate`;
- retained objects and `retained_damage_rate`;
- paired object-level `net_tp_delta = patched_tp - base_tp`;
- P0 `cross_topk` events and routed Top-K recovery rate;
- fixed-target score delta on P0 crossing events;
- full-frame deployed TP/FP counts and relative FP inflation;
- risk probability and selected source IDs.

All object-level uncertainty is bootstrapped 5000 times by both `scene_token`
and `instance_token`. FP inflation is bootstrapped 5000 times by scene.

## Hard gate

A protocol/seed passes only when all conditions hold:

1. lost-target recovery point estimate > 0 and both scene/instance 95% CI lower
   bounds > 0;
2. paired net TP delta > 0 and both scene/instance CI lower bounds > 0;
3. P0 crossing recovery > 0 and both scene/instance CI lower bounds > 0;
4. target-score delta on P0 crossing events > 0 and both scene/instance CI lower
   bounds > 0;
5. retained damage rate <= `0.5%` and both scene/instance CI upper bounds <=
   `1.0%`;
6. deployed FP inflation <= `1.0%` and scene-bootstrap 95% CI upper bound <=
   `2.0%`;
7. clean identity passes.

A fault family passes only if all three frozen seeds pass. P1 is
`GO_CARE3D_P1_SPARSE_EVIDENCE_ROUTER` only if at least two qualitatively
different fault families pass. Otherwise the decision is
`NO_GO_CARE3D_P1_SPARSE_EVIDENCE_ROUTER` and downstream mechanism expansion is
locked.

## Stop rules

- Do not choose the best seed.
- Do not change thresholds after test unlock.
- Do not inspect partial probe-test metrics to retune P1.
- Do not add more source cameras or increase `top_k` after a failed formal gate.
- A No-Go is retained as the formal result and must be explained before a new
  preregistered P1 variant is created.
