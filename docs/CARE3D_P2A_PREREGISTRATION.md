# CARE-3D P2-A0 Online Query Association Preregistration

## Scope

P2-A0 is the deployment-gap pre-gate that follows the frozen P1 decision
`GO_CARE3D_P1_SPARSE_EVIDENCE_ROUTER`.  P1 proved that sparse redundant
routing can recover fault-lost targets when the paired target query is supplied
as an oracle experimental unit.  P2-A0 asks a narrower question:

> while the object is still a valid current clean detection at frame `t`, can
> deployment-visible information associate that object to the P1 oracle query
> among all 900 fault-frame decoder queries at `t+1`?

P2-A0 does **not** train or evaluate the P1 router.  It does not modify
StreamPETR, P0, P1, source-bank definitions, P1 gates, or any frozen P1 result.
P2-A1 online routing remains locked until P2-A0 passes.

## Frozen prerequisites

- P1 decision: `GO_CARE3D_P1_SPARSE_EVIDENCE_ROUTER`;
- P1 `P2_status`: `ELIGIBLE`;
- frozen P1 decision SHA256:
  `d0172c9de8d17c4359224a90cee11312d260b45f595fcf662d5d8ccbd2405da3`;
- full-nuScenes detector checkpoint:
  `checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth`;
- detector checkpoint SHA256:
  `e6323ae5c31adf1eedd46d6dd4fd3c73d95aa26f18cc8aa23c196494b7de3451`;
- detector config:
  `configs/full_nuscenes/stream_petr_r50_90e_ctep_train_audit.py`;
- detector config SHA256:
  `927ba2518a4ca460d2f7f6b3ba74dab620ac8e2995ee7e9aadbcbebf2d7c64a6`;
- train info:
  `data/nuscenes/nuscenes2d_temporal_infos_train.pkl`;
- train-info SHA256:
  `dc5e5e611badbdb1c0270a3583e022cf14a9af7b3ff8f02370434b8ec50b493d`;
- official val info exists for later P2 system confirmation but is **not** a
  P2-A0 tuning source;
- P0/P1 scene-manifest SHA256:
  `83637205c930611ccdc6879eb233f72a9b0a5997248f4b5b5edf3242182d6da1`.

The frozen internal scene split is retained byte-for-byte:

- `probe_train`: 419 scenes;
- `probe_val`: 133 scenes;
- `probe_test`: 132 scenes;
- 16 earlier mechanism-discovery scenes remain excluded and only one of them
  may be used for engineering smoke.

P2-A0 configuration selection uses **probe_train only**.  Probe-val is a
confirmation split and cannot retune weights or thresholds.  Probe-test remains
locked throughout P2-A0 and cannot be opened to repair a failed P2-A0 gate.

## Experimental unit and eligibility

The base object cohort is the P1-eligible P0 paired cohort.  P1 already excludes
all rows in a shared `(target_frame_idx, target_clean_query_index)` collision
group because one detector query cannot receive two object-specific routed
vectors in the same frame.

P2-A0 adds the outcome-blind policy:

`exclude_all_rows_in_shared_anchor_or_target_query_frame`

Within a frame transition, all rows are excluded when either the current
`anchor_query_index` or the oracle `target_clean_query_index` is shared by more
than one cohort object.  This is an implementation eligibility rule, not an
outcome-dependent filter.  Collision counts are reported for train and val.

GT is permitted only to define the already-frozen current-TP cohort and to
check the offline oracle target query.  GT coordinates, labels, identities, or
future outcomes are never passed to the association cost.

## Online information contract

For each eligible current object at clean anchor frame `t`, association receives
only:

1. the 256-D `final_decoder_pre_cls_query` at the current anchor query;
2. the current anchor prediction class;
3. the current anchor predicted 3D center;
4. the target-frame ego pose/calibration needed to transform that predicted
   center from anchor lidar coordinates into target lidar coordinates;
5. at fault frame `t+1`, all 900 256-D final pre-classification query features;
6. all fault-frame classification logits;
7. all fault-frame predicted 3D centers.

Forbidden association inputs:

- `target_clean_query_index`;
- clean `t+1` tensors, logits, boxes, or query features;
- GT center, GT class, GT token, or instance token;
- fault outcome labels such as TP/FN, `cross_topk`, or evidence drop;
- any future frame.

The oracle `target_clean_query_index` is used only after assignment for offline
metric computation.

## Temporal/counterfactual execution

The formal execution preserves the existing one-step counterfactual state rule.
For every target frame:

```text
clean post-state H_t
  |-- clean(t+1)       -> advances the next anchor only
  |-- blur_back(t+1)   -> discarded after association capture
  |-- crash_back(t+1)  -> discarded after association capture
  `-- dark_back(t+1)   -> discarded after association capture
```

Every branch starts from an exact clone of the same clean `H_t`.  Fault history
is prohibited.  The clean `t+1` branch may become the anchor for the following
transition, but its output is not an input to the association being evaluated
for the preceding `t -> t+1` transition.

The StreamPETR query layout is frozen to `644 + 256 = 900` queries.

## Deterministic association

No new association neural network is trained in P2-A0.

For current object `i` and fault-frame query `j`:

```text
C_ij = lambda_geo * C_geo
     + lambda_emb * C_emb
     + lambda_cls * C_cls
```

where

```text
C_geo = min(||xy_i_to_target - xy_j||_2 / 12 m, 1)
C_emb = (1 - cosine(z_i, z_j)) / 2
C_cls = 1 - sigmoid(logit_j[c_i])
```

`c_i` is the current anchor prediction class.  A candidate with predicted XY
center distance greater than 12 m is prohibited regardless of its other costs.

Association is one-to-one Hungarian matching.  Each current track receives one
private dummy column with cost just above the frozen maximum accepted real cost,
so a track may remain unmatched rather than being forced onto a prohibited or
high-cost real query.

The cost matrices are computed with vectorized torch operations.  The Hungarian
solver may consume the resulting small matrix on CPU.  No Python loop over the
900 candidate queries is permitted in cost construction.

## Frozen train-only parameter grid

A single global configuration is selected; protocol-specific parameters are
prohibited.

Weights `(lambda_geo, lambda_emb, lambda_cls)`:

```text
(0.5, 0.3, 0.2)
(0.4, 0.4, 0.2)
(0.6, 0.2, 0.2)
(0.4, 0.3, 0.3)
(0.5, 0.2, 0.3)
```

Maximum accepted association cost:

```text
0.35
0.45
0.55
```

This yields exactly 15 candidates.

Selection uses all 419 probe-train scenes and all three fault protocols.  The
primary criterion is the macro mean across protocols of exact oracle-query
Recall@1.  Ties are broken, in order, by:

1. lower protocol-macro wrong-match rate;
2. lower protocol-macro mean cost among accepted real matches;
3. lexicographically smaller frozen configuration ID.

After a unique winner is written to `selection.json`, the configuration is
frozen.  Probe-val cannot change it.

Definitions use all eligible rows as denominator:

```text
exact Recall@1 = exact oracle-query assignments / eligible rows
wrong-match rate = accepted wrong-query assignments / eligible rows
unmatched rate = dummy assignments / eligible rows
```

These three outcomes are exhaustive.

## Baselines

Probe-val reports the frozen selected full association together with three
non-tuned baselines.  Every baseline uses the selected full-association
max-cost threshold so that only the cost components change:

- geometry only: `(1.0, 0.0, 0.0)`;
- embedding only: `(0.0, 1.0, 0.0)`;
- class + geometry: `(0.5, 0.0, 0.5)`.

Baselines do not affect the Go/No-Go decision.

## Required metrics

For each protocol and each method on probe-val:

- exact oracle-query Recall@1;
- wrong-match rate;
- unmatched rate;
- mean accepted association cost;
- oracle-query cost rank;
- correct-vs-best-wrong cost margin;
- oracle geometry-eligibility rate.

For the selected full association, uncertainty is reported with:

- 5,000 scene-cluster bootstrap replicates;
- 5,000 instance/trajectory-cluster bootstrap replicates.

The bootstrap primary variable is per-object exact-match indicator.  Wrong and
unmatched rates are also reported descriptively.

## Frozen P2-A0 Go / No-Go gate

A fault protocol passes only when the selected full association satisfies all
of the following on frozen probe-val:

1. exact oracle-query Recall@1 >= `0.70`;
2. 95% scene-cluster bootstrap lower bound for exact Recall@1 > `0.50`;
3. 95% instance/trajectory-cluster bootstrap lower bound > `0.50`;
4. wrong-match rate <= `0.10`.

P2-A0 is `GO` only if at least two qualitatively different fault protocols pass.
All three protocols are reported regardless of outcome.

If fewer than two protocols pass:

`NO_GO_CARE3D_P2A_ONLINE_QUERY_ASSOCIATION`

and P2-A1 remains locked.  Probe-test must not be opened to repair the result.

If at least two pass:

`GO_CARE3D_P2A_ONLINE_QUERY_ASSOCIATION`

and P2-A1 online router evaluation becomes eligible.  The selected association
configuration, P0, P1, detector, source bank, and all P1 gates remain frozen.

## Engineering smoke

Before any formal train extraction, exactly one of the 16 excluded discovery
scenes may be used.  The smoke must verify:

- passive tap capture preserves frozen B0 output/state;
- query layout remains 644 current + 256 propagated = 900;
- clean/fault branches begin from identical clean `H_t`;
- the association cost API has no oracle-query, GT, or clean-future input;
- all cost matrices are finite before geometry masking and have shape `N x 900`;
- Hungarian real-query assignments are one-to-one and unmatched is supported;
- no formal probe-test artifact exists or is read.

Smoke results cannot contribute to parameter selection or any formal metric.

## Performance constraints

Detector and datasets are initialized once per extraction process.  Association
cost matrices are torch-vectorized.  Fault protocols remain sequential in the
formal path because the earlier classifier replay audit demonstrated
execution-shape sensitivity; the detector classifier execution shape is not
changed for throughput.  Safe input prefetching or storage optimizations may be
introduced only if an engineering equivalence check proves detector outputs,
query taps, boxes, and association components unchanged within the declared
numerical tolerance.
