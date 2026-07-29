# S2.5 Method Specification: Soft Write-back Gating

## Authoritative definition

The unique next stage in `docs/Stage2_Codex实验总纲.md` sections 4.8, 8 and 17
is:

- stage: **S2.5**
- formal name: **软写回门控（Soft Write-back Gating）**
- documented scales:
  - KEEP `write_scale=1.0`
  - RECOVER `write_scale=0.5–0.8`
  - DEFER `write_scale=0.0–0.1`
- hard DEFER gating belongs to S2.6, not S2.5.

## Goal

Replace the current binary memory-write decision with an action-conditioned
soft write strength. Reliable KEEP states enter temporal memory normally;
RECOVER states enter with reduced strength; uncertain DEFER states have only
a small soft write during S2.5. Current-frame detections remain available.

The research question is whether controlled write strength reduces temporal
memory pollution without sacrificing Clean detection or recovery.

## Dependency

S2.5 depends on:

- the existing KEEP/RECOVER/DEFER action;
- the existing Top-K temporal-memory write path;
- scene-safe S2.2 ledger state.

Those capabilities are already present in `s2.2-stable`. S2.5 does not require
S2.3 novelty/reacquisition success or an independently successful S2.4
correlation method. It may start directly from `s2.2-stable`, while explicitly
inheriting that anchor's legacy fixed-correlation behavior.

## Frozen first candidate

Main candidate:

```text
KEEP    write_scale = 1.00
RECOVER write_scale = 0.75
DEFER   write_scale = 0.05
```

The values are inside the ranges specified by the Stage2 plan. They are frozen
before implementation; this stage is not a scale sweep.

At most one necessary ablation:

```text
KEEP    write_scale = 1.00
RECOVER write_scale = 1.00
DEFER   write_scale = 0.05
```

This `no_recover_discount` ablation isolates the documented contribution of
reduced RECOVER write strength. The existing S2.2 binary behavior is the
baseline, not a third candidate.

## Inputs and outputs

Inputs:

- action (`KEEP`, `RECOVER`, `DEFER`);
- bootstrap/warm-up state already used by the memory writer;
- Top-K selected query state.

New output:

- `write_scale`, aligned with batch/query/Top-K dimensions.

The action policy, score calibration and current-frame detection outputs are
not redefined by S2.5.

## Modules in scope

- `models/keep_recover_defer.py`
  - produce an action-aligned `write_scale`;
  - retain the existing action and score-scale semantics.
- `models/streampetr_adapter.py`
  - apply soft scale at the temporal-memory write point;
  - keep memory embedding, reference state, velocity, pose and ledger commit
    aligned;
  - preserve bootstrap and warm-up semantics.
- `models/evidence_ledger.py`
  - only if needed to keep the committed ledger state consistent with the
    scaled memory state; no new evidence formula.
- diagnostics/evaluation hooks
  - export `write_scale` and per-action write statistics.
- tests and S2.5 configs
  - disabled invariance, shape/reset/Top-K/checkpoint and FP16 coverage.

No new loss, teacher, dynamic correlation, S2.3 rescue or S2.6 hard gate is in
scope.

## Required isolation

Add one explicit switch, default off:

```text
enable_soft_write_gate = False
```

When off, the model must execute the exact S2.2 memory-write path. It must not
calculate or apply S2.5 scales. Off-path classification, boxes, predictions,
ledger evidence, action, write mask, Top-K and memory must have
`max_abs_diff=0`.
