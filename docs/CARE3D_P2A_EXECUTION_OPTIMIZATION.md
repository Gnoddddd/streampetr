# CARE-3D P2-A execution-only acceleration

This note documents an implementation-only acceleration added after the P2-A engineering smoke passed and before any formal 419-scene probe-train extraction.

It does **not** change the preregistered scientific definition.  The following remain frozen and identical: cohort, protocols, detector checkpoint, clean/fault state construction, 900-query tensors, association cost, 12 m geometry gate, Hungarian matching, unmatched threshold, 15-config train-only grid, train-only selection rule, probe-val gate, bootstrap count, and probe-test lock.

## Execution policy

`p2a_train_parallel_shards_fast_diagnostics_v1`

Two independent accelerations are permitted for probe-train only.

1. **Deterministic scene sharding.** The ordered 419-scene probe-train manifest is partitioned by strided index (`shard_index::num_shards`). Shards are pairwise disjoint and exhaustive. Each worker owns unique scene output paths. Shared progress-manifest writes are deferred until all workers exit successfully, after which the canonical exporter performs a zero-forward progress refresh.

2. **Omit unused train-only rank/margin sorting.** Formal train configuration selection consumes selected query, selected cost, exact/wrong/unmatched counts and accepted-cost sums. It does not consume oracle-query rank or correct-vs-best-wrong margin. The accelerated train wrapper therefore gathers oracle cost/geometry eligibility in O(N) and emits sentinel rank/margin values only inside the temporary in-memory structure required by `assignment_rows`. Selection-relevant outputs are bit-for-bit identical to the full diagnostic path; unit tests compare them directly. Probe-val continues to use the complete oracle diagnostics and is not accelerated by this omission.

No classifier execution shape, detector tensor shape, precision policy, model parameter, batch size, or fault transformation is changed.

## Recommended hardware use

Start with two concurrent workers on the single RTX 3080 Ti:

```bash
P2A_WORKERS=2 CUDA_VISIBLE_DEVICES=0 bash scripts/run_care3d_p2a_train_parallel.sh
```

Do not increase worker count merely to fill VRAM. Observe `nvidia-smi dmon -s pucvmt -d 1`; increase to three workers only if two workers remain clearly underutilized and memory headroom is comfortably above the observed per-worker peak. If throughput falls because the mounted F: drive becomes the bottleneck, reduce the worker count.

## Preflight

Before launching all 419 scenes, an optional execution-only pilot can process one formal train scene per shard. Do not inspect association efficacy from these rows; inspect only runtime/invariants/log errors. The later full run resumes from the written complete markers.

```bash
P2A_WORKERS=2 P2A_MAX_SCENES_PER_SHARD=1 \
  CUDA_VISIBLE_DEVICES=0 bash scripts/run_care3d_p2a_train_parallel.sh
```

The pilot deliberately does not update the shared progress manifest or run train configuration selection.

## Formal sequence

After the execution tests/pilot pass:

```bash
P2A_WORKERS=2 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_care3d_p2a_train_parallel.sh
python scripts/select_care3d_p2a_train_config.py
python scripts/export_care3d_p2a_association.py --split probe_val --device cuda:0
python scripts/analyze_care3d_p2a_val.py
```

The last three commands must only be reached in order. `probe_test` remains unavailable throughout P2-A0.
