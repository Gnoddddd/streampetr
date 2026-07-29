# S2.5 Baseline Anchor

## Git anchor

- stable tag: `s2.2-stable`
- peeled commit: `995836632255c637f2c89137bc868853f3d8a042`
- inherited semantics:
  - source-aware evidence ledger;
  - evidence conservation and source-mass checks;
  - KEEP/RECOVER/DEFER action;
  - binary temporal-memory write behavior;
  - legacy fixed-correlation `N_eff`.

S2.5 must start from a fresh branch rooted in this stable lineage. It must not
inherit S2.3 rescue code, C0 no-discount promotion, or experimental S2.4
branches.

## Baseline protocol metrics

| Protocol | mAP | NDS |
|---|---:|---:|
| Clean | 0.4247724932 | 0.4770300703 |
| Crash5 | 0.4183114887 | 0.4730280297 |
| Crash10 | 0.4109787293 | 0.4706696455 |
| Compound | 0.3923952449 | 0.4573203851 |

## Dependency statement

S2.5 is downstream of the action policy and memory writer, both already
available in S2.2. It is independent of whether S2.3 novelty/reacquisition or
S2.4 fixed correlation produced a new performance gain. The inherited fixed
correlation is held constant in both baseline and S2.5 candidates and is not
claimed as an S2.5 contribution.

## Branch and artifact rules

- create the future implementation branch from `s2.2-stable`, not from the
  current freeze branch;
- do not move existing tags;
- use a new `outputs/stage2/s2_5_*` directory;
- do not commit outputs, checkpoints, data, weights, caches or third-party
  repository changes;
- implementation and training require a new explicit task.
