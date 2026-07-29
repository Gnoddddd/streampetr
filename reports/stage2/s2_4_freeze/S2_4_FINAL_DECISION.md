# S2.4 Final Decision

## Frozen status

- Final experiment commit:
  `fda2324c49c72e7c19929c74cc71dee9f8f015fd`
- Final tag: `s2.4-fixed-correlation-no-independent-gain`
- Stable baseline remains: `s2.2-stable`
  (`995836632255c637f2c89137bc868853f3d8a042`)
- No S2.2 tag was moved or rewritten.

S2.4 is stopped as an independent contribution. No additional seed, dynamic
correlation or holdout experiment is authorized.

## Semantic finding

The executable S2.2 anchor already calculates the fixed six-camera correlation
matrix and uses its `N_eff` in evidence accumulation. The explicit legacy
path reproduces all four historical S2.2 predictions tensor-for-tensor.
Therefore fixed correlation is an existing S2.2 behavior, not a new S2.4
module.

## 50-iteration disambiguation

Both C0 and C1 started from the same pre-ledger Stage1 checkpoint and seed
2026.

| Metric | C0 no-discount | C1 legacy fixed | C0-C1 |
|---|---:|---:|---:|
| Clean mAP | .427493 | .424772 | +.002721 |
| Clean NDS | .479548 | .477030 | +.002518 |
| Fault-average mAP | .408142 | .407228 | +.000914 |
| Fault-average NDS | .467454 | .467006 | +.000447 |

The screen motivated a frozen 200-iteration confirmation; it did not redefine
the stable baseline.

## 200-iteration confirmation

The non-inferiority gate was committed before training. C0 had to preserve
Clean, preserve the fault average, and avoid any individual fault regression
larger than `0.0010`.

| Protocol | C0 mAP/NDS | C1 mAP/NDS | C0-C1 |
|---|---|---|---|
| Clean | .424837/.476760 | .427993/.479573 | -.003156/-.002813 |
| Crash5 | .418599/.471713 | .418227/.471671 | +.000372/+.000042 |
| Crash10 | .413006/.471748 | .408842/.468460 | +.004165/+.003288 |
| Compound | .388993/.455022 | .392175/.457286 | -.003181/-.002264 |
| Fault average | .406866/.466161 | .406415/.465805 | +.000452/+.000355 |

C0 failed the Clean gate and the Compound single-protocol tolerance. It is not
a provisional canonical baseline and must not replace legacy S2.2.

## Final interpretation

1. `s2.2-stable` is the formal stable baseline and already contains legacy
   fixed-correlation semantics.
2. Fixed correlation has no independently attributable S2.4 contribution.
3. No-discount C0 is a useful diagnostic counterfactual, not a stable
   replacement.
4. S2.4 ends here. Dynamic correlation is not the next authorized step.

All checkpoints and outputs remain local evidence and are not committed.
