# CARE-3D P2-A2-R0 Memory-Lineage Runbook

## Allowed sequence

Run the focused tests and excluded engineering smoke first:

```bash
DEVICE=cuda:0 bash scripts/run_care3d_p2a2_lineage_smoke.sh
```

The smoke writes
`reports/care3d/p2a2_memory_lineage_r0/engineering_smoke.json` and must report
`P2A2_R0_LINEAGE_SMOKE_PASSED`. Formal extraction is locked by the parallel
runner until this status exists.

After explicitly reviewing the smoke, launch the frozen two-worker
`probe_train` extraction:

```bash
DEVICE=cuda:0 P2A2_WORKERS=2 \
  bash scripts/run_care3d_p2a2_lineage_parallel.sh
```

The runner uses deterministic strided scene shards and shared immutable dataset
metadata. It resumes from per-scene completion markers. Only after all 419
markers exist does it refresh the progress manifest and run the development
analysis.

For a non-formal execution pilot, limit each shard:

```bash
DEVICE=cuda:0 P2A2_WORKERS=2 P2A2_MAX_SCENES_PER_SHARD=1 \
  bash scripts/run_care3d_p2a2_lineage_parallel.sh
```

A pilot intentionally does not analyze or claim completion.

## Outputs

Generated outputs are ignored under
`reports/care3d/p2a2_memory_lineage_r0/`. The formal analysis produces:

- `p2a2_r0_protocol_metrics.csv`
- `p2a2_r0_paired_cluster_ci.csv`
- `p2a2_r0_diagnostic_strata.csv`
- `p2a2_r0_gate_summary.csv`
- `decision.json`

The decision is a development pre-gate only. It must not be described as a
confirmatory result.

## Split discipline

The extraction entry point accepts exactly one of `--engineering-scene` or
`--split probe_train`. `probe_val` and `probe_test` are rejected during argument
parsing. Preparation emits a train-only manifest, and every report records both
held-out read flags as false.
