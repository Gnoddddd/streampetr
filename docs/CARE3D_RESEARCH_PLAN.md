# CARE-3D Research Plan

## Goal

CARE-3D (Counterfactual Adaptive Redundant Evidence Routing) treats StreamPETR
as a discovery host rather than the final target architecture. The research
question is detector-agnostic:

> Which evidence sources does a currently correct object depend on, how much
> evidence would it lose under a source/fault intervention, and which reliable
> redundant source should be routed to the object before its evidence crosses
> the detector's decision boundary?

The target mechanism is deliberately written without StreamPETR-specific
terms:

```text
sensor / imaging fault
  -> object evidence degradation
  -> architecture-specific decision-boundary crossing
  -> miss
```

For StreamPETR, positive-query score collapse and Top-K crossing are one
instance of this general mechanism.

## Design principles

1. **Object-centric, not frame-global.** A fault does not affect every object
   equally; vulnerability is represented per object.
2. **Source-aware, not feature-only.** Camera and temporal sources remain
   distinguishable instead of being collapsed into one scalar reliability.
3. **Counterfactual supervision, not post-hoc failure classification.** P0
   predicts the response `E_clean - E_fault`, not merely TP/FN.
4. **Geometry/reliability constrain routing.** Redundant evidence is selected
   from physically plausible and currently reliable sources.
5. **No direct score boosting.** Vulnerability may decide where to retrieve
   evidence, but it must not directly increase a detector score.
6. **Architecture-generalizable interface.** Detector-specific code only
   exposes canonical object/source tensors; CARE-3D core remains unchanged.

## Canonical interface

Each detector adapter should expose:

```text
object_features      [B,Q,C]
camera_support       [B,Q,N_cam]
camera_quality       [B,N_cam] or [B,Q,N_cam]
temporal_features    [B,Q,C]          optional
decision_features    [B,Q,D]          optional
source_features      [B,Q,S,Cs]       P1 only
source_reliability   [B,Q,S]          P1 only
source_valid         [B,Q,S]          P1 only
```

No StreamPETR memory-bank field is part of the core API.

## P0: Counterfactual vulnerability decodability

### Frozen detector

Do not change detection predictions. Run the frozen baseline on paired clean
and corrupted data. Build a frozen cohort from objects that are correctly
matched in the clean reference frame.

For protocol `p`, define detector-agnostic object evidence `E` and target:

```text
drop_i,p = max(E_clean_i - E_fault_i,p, 0)
cross_i,p = 1 if the same object crosses the architecture-specific decision
            boundary under p, otherwise 0
```

For StreamPETR, the current validated implementation may use the matched
positive-query evidence and Top-K boundary. Other architectures should map
these to their own object-evidence and decision-boundary definitions.

### Predictor

`models/care3d.py` implements:

```text
CARE3DStateEncoder
  -> CounterfactualVulnerabilityHead
       -> protocol-conditioned evidence-drop vector
       -> protocol-conditioned boundary-crossing logits
```

P0 uses **no routing**. `CARE3DCore(enable_routing=False)` returns the detector
object features unchanged.

### Required P0 gates

Do not promote to P1 from a single AUROC. At minimum check:

- scene/trajectory-cluster bootstrap stability;
- rank correlation between predicted and actual evidence drop;
- high-vulnerability vs low-vulnerability separation in actual drop;
- boundary-crossing AUROC/AUPRC under class imbalance;
- cross-severity transfer;
- leave-one-scene-out or frozen-scene holdout;
- protocol-wise results rather than forcing Blur/Crash/Dark into one scalar;
- strict leakage audit: no `t+1`/fault outcome information in the predictor
  input when the experiment is prospective.

Suggested hard stop: if the vulnerability ranking is not cluster-stable for at
least two qualitatively different fault families, do not add a more complex
router to rescue the hypothesis.

## P1: Sparse evidence routing

Only after P0 passes, enable `SparseEvidenceRouter`.

The router receives per-object source tokens and reliability. It selects only
Top-K reliable sources and produces a residual object correction:

```text
object token
  + vulnerability-conditioned query
  -> reliable source ranking
  -> Top-K sparse source attention
  -> residual evidence correction
  -> original detector head
```

Important invariants implemented in the current core:

- zero-initialized residual scale -> exact baseline identity at initialization;
- zero-reliability sources receive zero routing mass;
- no reliable source -> exact no-op, not hallucinated recovery;
- vulnerability never directly rescales class score.

Start with Top-1/Top-2 source routing. Do not use a large all-camera transformer
until the sparse version proves target-specific causal recovery.

## P2: Geometry-constrained source construction

After P1 establishes benefit, replace generic source tokens with explicit
object evidence sources:

- current-camera local object token;
- adjacent-camera token only when the object projects into a valid overlap;
- motion-aligned temporal object token;
- optional local BEV token for dense-BEV detectors.

Camera/object edges should be gated by calibration-derived projection,
visibility, image-boundary distance, projected area, and FOV overlap. Temporal
edges should be motion/ego-motion aligned.

## Cross-architecture validation

Do not claim generality from StreamPETR + a near-identical query detector alone.
Recommended validation order:

1. StreamPETR: mechanism discovery and full ablation.
2. One different query/BEV transformer detector: adapter-level replication.
3. One LSS/center-based BEV detector such as BEVDepth/BEVStereo: cross-paradigm
   test with ROI/bilinear object-token extraction and local BEV scatter-back.

The CARE-3D core, losses, and vulnerability definition should remain unchanged;
only the thin detector adapter may differ.

## Strong generalization experiments

Once the main model works:

- leave-one-fault-out;
- leave-one-camera-out;
- cross-severity;
- clean-performance preservation;
- fault recovery vs false-positive tradeoff;
- latency/FLOPs/active-object ratio;
- source-routing interpretability against actual counterfactual dependence.

These tests are important to show the method learns evidence dependency and
redundancy rather than memorizing three corruption labels.

## Repository integration policy

The repository's official StreamPETR dependency under `repos/StreamPETR` should
remain untouched. New research code belongs in the project root and adapter
layer. P0 must remain disabled by default until its data/metrics gate passes.

Current implementation on `care3d/p0-counterfactual-vulnerability`:

- `models/care3d.py`: detector-agnostic state encoder, vulnerability head,
  vulnerability loss, sparse router, and composite core;
- `tests/test_care3d.py`: tensor-contract and disabled/identity invariants;
- no change to the existing StreamPETR inference path yet.
