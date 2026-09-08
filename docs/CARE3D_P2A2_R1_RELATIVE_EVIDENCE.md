# CARE-3D P2-A2-R1-F0 Relative Candidate Evidence Audit

## Scope

R1-F0 is a `probe_train`-only feature-sufficiency diagnostic. It does not train
or define the final R1 association model. P2-A0, P2-A2-R0, the detector,
checkpoint, fault protocols, collision policy, and all frozen gates remain
unchanged. `probe_val` and `probe_test` are inaccessible.

Rows are exported only when frozen P2-A0 candidate `A` differs from explicit
one-step lineage candidate `L`. Before any feature row is accepted, the rerun
must exactly reproduce the stored R0 P2-A0 selection, lineage child, hybrid
assignment, and selected P2-A0 cost.

## Feature definition

The exporter calls the existing `association_cost_components()` and
`weighted_cost()` with `g0.4_e0.4_c0.2_t0.45`. For both A and L it exports:

- geometry, embedding, anchor-class cost, total cost, and distance;
- anchor-class probability (`1 - class_cost`);
- fault-query top-1 probability and predicted class;
- whether the predicted class equals the online anchor class.

Every scalar pair also has the prescribed `L - A` delta. Row ranks are
one-based over finite geometry-eligible queries with query-index tie-breaking.
Row margin is the best other eligible cost minus the candidate cost.

Column ownership uses the complete frame-local online-anchor by 900-query cost
matrix. `mutual` means the current row is the minimum-cost anchor for that
candidate. Its column margin is second-best minus current when owned, otherwise
best minus current, so the sign records local ownership. Row/column ranks and
margins never use the oracle. An ineligible candidate receives rank zero and a
NaN margin, which the analysis rejects.

When a candidate has no second finite anchor, the exact raw column margin is
`+inf`. Raw CSVs retain that value. The fixed model-input conversion maps only
this no-competitor column-margin case to the positive boundary sentinel `1.0`.
The model delta is then recomputed from the encoded L and A margins, including
one-sided no-competitor cases. Any invalid A/L margin or other non-finite model
feature is rejected. This conversion is fixed, label-free, protocol-free, and
preserves every disagreement row.

Retained context is limited to:

- `anchor_is_propagated`
- `p2a0_is_propagated`
- `lineage_position_norm = lineage_position / 255`
- `target_frame_norm = (target_frame_idx - 3) / 9`

Protocol is metadata for reporting and is never a model feature.

## Leakage boundary

`relative_candidate_features()` has no GT, oracle, clean-future, or protocol
argument. Only after it returns does the exporter read the stored R0 oracle
query and call a separate labeling function. The labels are mutually exclusive
and exhaustive: `P2A0_WINS`, `LINEAGE_WINS`, or `BOTH_WRONG`.

## Fixed F0 diagnostics

All diagnostics use five-fold `GroupKFold(group=scene_token)` over the pooled
protocol rows. Consequently every protocol view of a scene remains in the same
fold. Raw CSV columns remain frozen in `EVIDENCE_FEATURE_COLUMNS`. Model inputs
use `MODEL_FEATURE_COLUMNS`, which excludes the categorical identifiers
`A_predicted_class` and `L_predicted_class`; protocol is also excluded.

The decisive-preference diagnostic excludes both-wrong rows, targets lineage
wins, and uses `StandardScaler` plus balanced logistic regression with
`max_iter=2000` and fixed random state. It reports per-protocol AUROC, AUPRC,
accuracy at 0.5, and lineage choice rate. Both-wrong detection uses the same
fixed binary pipeline on all disagreement rows and reports AUROC/AUPRC.

One fixed balanced multinomial logistic regression additionally reports
per-protocol macro one-vs-rest AUROC, balanced accuracy, and confusion matrices.
There is no model-family, hyperparameter, or threshold search.

`PASS_R1_RELATIVE_EVIDENCE` requires at least two protocols with decisive AUROC
at least 0.85 and every protocol strictly above its frozen cheap-feature AUROC.
Otherwise the status is `STOP_RELATIVE_EVIDENCE_INSUFFICIENT`. Both-wrong AUROC
is auxiliary and does not gate the status.

## Execution

Run focused tests and the excluded engineering smoke:

```bash
DEVICE=cuda:0 bash scripts/run_care3d_p2a2_r1_feature_smoke.sh
```

After review, the formal two-worker command is:

```bash
DEVICE=cuda:0 P2A2_R1_WORKERS=2 \
  bash scripts/run_care3d_p2a2_r1_feature_parallel.sh
```

Generated artifacts live under
`reports/care3d/p2a2_r1_relative_evidence/` and are gitignored.
