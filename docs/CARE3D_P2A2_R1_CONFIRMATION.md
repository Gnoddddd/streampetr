# CARE-3D P2-A2-R1-C0 Frozen Arbiter Confirmation

## Scope and state

P2-A2-R1-C0 is an independent confirmatory evaluation of the already-frozen
identity arbiter. The source baseline is
`d499dbb34c44fc91ece1444580ff37ef40ce939e`. Until a clean cohort is approved
and the official validation extraction is separately authorized, the state is:

`SOURCE_REPAIRED_TESTED_AWAITING_CONFIRMATORY_REREVIEW`

This source pass does not read official nuScenes validation detector data, run
feature extraction, open the original 133-scene probe-val cache, or open the
original 132-scene probe-test cache.

## Immutable inputs

The runtime loads
`reports/care3d/p2a2_r1_identity_arbiter/frozen_arbiter.json` directly and
requires SHA256
`cdfc0b1e6f967498b2d394a3698bf062d93653448b177698ca1f3b9653c64e69`.
It also verifies the exact 40-column `MODEL_FEATURE_COLUMNS` order and the two
frozen thresholds:

- `tau_preference = 0.52`
- `tau_defer = 0.88`

There is no fit, rescaling, calibration, threshold search, feature search, or
model search interface in the confirmation module. P2-A0 remains
`geo=0.4`, `embedding=0.4`, `class=0.2`, `max_cost=0.45` through the frozen
upstream evidence contract.

## Cohort and lineage audit

Before any official-val evidence can be exported, the audit entry point
resolves the official 150-scene nuScenes val split from
`create_splits_scenes()["val"]` plus NuScenes scene metadata. The candidate
identity manifest must contain both `scene_token` and `split`, contain exactly
150 unique `official_val` rows, and match that official token set exactly.
Missing, extra, or substituted scenes stop confirmation before historical
exclusion. The audit then requires a complete registry
of every scene manifest used for P2 fitting, tuning, model/feature selection,
or failure-driven redesign. The registry must include at least the 419
P2/R1 probe-train scenes and original 133 probe-val scene identities. Additional
P2 development manifests are explicit required sources; an omitted or
unregistered source produces `STOP_CONFIRMATORY_COHORT_NOT_CLEAN`.
Before authorization, human source review must freeze the complete registry.
The two minimum sources may serve as a union-level provenance source only when
that review proves they are a superset of every P2 fitting, tuning, selection,
and redesign scene. Supplying only those two CLI entries does not itself prove
registry completeness.

Only `scene_token` and optional `split` provenance are accepted. Protocol,
severity, detection result, association result, oracle identity, and outcome
columns are rejected. Each used-scene intersection is recorded, intersecting
official-val scenes are excluded, and the remaining cohort is accepted only if
its union intersection is exactly zero. Probe-val access here means identity
provenance only, never its raw feature/outcome rows. Probe-test names and paths
are rejected entirely and do not participate in selection.

An authorized future identity-only audit uses the following shape (all actual
manifest paths and all additional lineage sources must be reviewed first):

```bash
python scripts/export_care3d_p2a2_r1_confirmation.py --audit-only \
  --official-val-manifest <official_val_identity_manifest.csv> \
  --lineage-manifest p2_r1_probe_train_419=<train_identity_manifest.csv> \
  --lineage-manifest original_probe_val_133=<val_identity_manifest.csv> \
  --required-source p2_r1_probe_train_419 \
  --required-source original_probe_val_133
```

The audit writes `confirmatory_manifest.csv`,
`heldout_lineage_audit.json`, and the initial `progress_manifest.json` under
`reports/care3d/p2a2_r1_confirmation_c0/`.

## Frozen runtime and full population

The staged per-scene evidence input must contain the complete eligible-object
population for each of exactly `blur_back`, `crash_back`, and `dark_back`.
Protocol groups must have identical eligible identity keys. The eligibility
definition and upstream P2-A0 association are frozen; all upstream audit flags
must prove that GT, oracle query, and clean future were absent from association
and feature computation.

Only the ordered 40-column model frame is passed to the arbiter. Protocol and
all offline labels are excluded. Agreement rows bypass both heads, retain A,
cannot defer, and retain `NaN` probabilities. Disagreements run the frozen
Preference and Ambiguity heads with this order:

```text
if A == L:                       AGREEMENT -> A
elif p_both_wrong >= 0.88:       DEFER -> -1
elif p_lineage >= 0.52:          LINEAGE -> L
else:                            P2A0 -> A
```

Oracle identity is attached only afterward by the offline outcome function.
Both P2-A0 and R1 rows must satisfy `exact + wrong + unmatched == 1`.

The authorized future exporter performs two passes per audited scene. A
clean-only pass freezes the unchanged P1/P2-A0 eligible-object population and
offline target identity. A fresh replay then computes A/L online evidence for
all three protocols before advancing the clean `t+1` state. Clean future and
oracle identity are never passed to association or either arbiter head.

```bash
python scripts/export_care3d_p2a2_r1_confirmation.py --extract-official-val \
  --device cuda:0
```

Formal extraction is frozen to
`configs/full_nuscenes/stream_petr_r50_90e_mechanism_val.py`. Immediately after
configuration parsing and before any dataset construction, the exporter
requires `data.test.ann_file` to resolve exactly to
`nuscenes2d_temporal_infos_val.pkl`; the train pickle is a hard failure.

The first pass uses the same target-query and anchor-query collision exclusions
as frozen P1/P2-A0. It deterministically uses frames 0 through 12 from each
audited official-val scene. The replay retains P2-A0
`0.4/0.4/0.2/max_cost=0.45`, explicit one-step memory lineage, and the frozen
relative-evidence helper. A staged `--export --input-dir ...` mode exists for
reviewed or synthetic online-evidence tables and applies the same validation.

Outputs are atomic
`incremental/official_val/<scene_token>.rows.csv` and
`<scene_token>.complete.json` pairs. A completion marker binds the source rows,
output rows, frozen artifact SHA, thresholds, discipline flags, and scene
identity. A verified pair is a resume no-op; incomplete or mismatched pairs are
recomputed atomically.

## Metrics, bootstrap, and Gate

Each protocol is evaluated on Agreement plus Disagreement rows. Point output
contains row counts, P2-A0 and frozen-R1 exact/wrong/unmatched rates, exact and
wrong deltas, repair/break counts, lineage switches, deferrals, and agreement
modifications.

The fixed-prediction paired percentile bootstrap uses 5,000 replicates and seed
314159. Baseline and R1 share each sampled cluster multiplicity. Separate
summaries use `scene_token` and the exact instance-trajectory key
`(scene_token, instance_token)` for `delta_exact`, `delta_wrong`,
`arbiter_exact`, `arbiter_wrong`, and `arbiter_unmatched`.

The analyzer applies C1-C10 exactly: wrong at most 0.10, unmatched at most
0.01, positive exact delta and strictly positive lower bounds under both
bootstraps, negative wrong delta and strictly negative upper bounds under both
bootstraps, zero agreement changes, and the full frozen/no-search/no-leakage
discipline. All three protocols must pass; Crash is mandatory and there is no
2-of-3 concession.

```bash
python scripts/analyze_care3d_p2a2_r1_confirmation.py
```

Analysis writes `protocol_metrics.csv`, `repair_breakdown.csv`, both bootstrap
summaries, `gate_summary.csv`, and `decision.json`. The only overall outcomes
are `CONFIRMED_P2A2_R1_FROZEN_ARBITER` and
`NOT_CONFIRMED_P2A2_R1_FROZEN_ARBITER`. A non-confirmation ends the experiment;
only a separate read-only failure decomposition is permitted.

C10 is not accepted from analyzer defaults. Every scene completion marker must
contain the exact frozen artifact SHA, thresholds, search/refit flags, protocol
discipline, and probe locks. Analysis hard-fails on any invalid marker, derives
the discipline mapping from all validated markers, and only then passes that
mapping to the Gate. A written decision records
`discipline_verified_from_all_markers=true`.
