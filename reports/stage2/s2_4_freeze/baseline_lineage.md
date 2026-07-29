# Evidence3D Stage2 Baseline Lineage

| Stage | Git anchor | Meaning | Status |
|---|---|---|---|
| S2.2 | `s2.2-stable` / `9958366` | Source-ledger stable baseline; executable code already includes legacy fixed correlation | formal stable |
| S2.3 | frozen negative branches/reports | Novelty/reacquisition candidates did not pass performance gates | stopped; not inherited |
| S2.4 audit | `de2abb6` | Proved true no-discount diverges from S2.2 and legacy path reproduces it | completed |
| S2.4 50 iter | `e5f3bbe` lineage | C0 led Clean and fault average at 50 iter | screening only |
| S2.4 200 iter | `fda2324` | C0 failed Clean and Compound pre-registered gates | negative confirmation |
| S2.4 freeze | `s2.4-fixed-correlation-no-independent-gain` | Fixed correlation has no independent new contribution | frozen |

## Baseline rule

The current formal stable baseline remains `s2.2-stable`. Its name is a Git
stage anchor, not a claim that correlation discount is absent. Any new
independent method must:

1. branch from the stable S2.2 lineage rather than S2.3/S2.4 negative
   experiment branches;
2. explicitly describe the inherited fixed-correlation behavior;
3. use a feature-specific off switch with exact disabled-path invariance;
4. preserve existing conservation, source-mass and checkpoint-state
   invariants.

C0 no-discount is not promoted, and neither `s2.2-stable` nor any pushed tag is
rewritten.
