# CARE-3D P2-A execution-only acceleration

This note documents implementation-only acceleration added after the P2-A engineering smoke passed and before any formal 419-scene probe-train extraction.

It does **not** change the preregistered scientific definition. The following remain frozen and identical: cohort, protocols, detector checkpoint, clean/fault state construction, 900-query tensors, association cost, 12 m geometry gate, Hungarian matching, unmatched threshold, 15-config train-only grid, train-only selection rule, probe-val gate, bootstrap count, and probe-test lock.

## Execution policy

`p2a_train_parallel_shared_infos_fast_diagnostics_v2`

Three independent execution accelerations are permitted for probe-train only.

1. **Deterministic scene sharding.** The ordered 419-scene probe-train manifest is partitioned by strided index (`shard_index::num_shards`). Shards are pairwise disjoint and exhaustive. Each worker owns unique scene output paths. Shared progress-manifest writes are deferred until all workers exit successfully, after which the canonical exporter performs a zero-forward progress refresh.

2. **Share immutable full-nuScenes dataset metadata inside each worker.** The canonical builder independently loads the 599 MiB annotation pickle for clean plus each of the three fault protocol datasets. With multiple workers this creates a much larger duplicated Python object graph and can exhaust host RAM even while GPU VRAM remains almost empty. The v2 train-only wrapper therefore builds the clean dataset canonically once, shallow-clones the dataset object for each fault protocol, reuses the exact same `data_infos` object, and rebuilds only the independent protocol pipeline with its frozen `ApplyPartialObservation.schedule_file`. This is valid only after the excluded-scene model-input equivalence check passes.

3. **Omit unused train-only rank/margin sorting.** Formal train configuration selection consumes selected query, selected cost, exact/wrong/unmatched counts and accepted-cost sums. It does not consume oracle-query rank or correct-vs-best-wrong margin. The accelerated train wrapper therefore gathers oracle cost/geometry eligibility in O(N) and emits sentinel rank/margin values only inside the temporary in-memory structure required by `assignment_rows`. Selection-relevant outputs are bit-for-bit identical to the full diagnostic path; unit tests compare them directly. Probe-val continues to use the complete oracle diagnostics and is not accelerated by this omission.

No classifier execution shape, detector tensor shape, precision policy, model parameter, batch size, fault transformation, association formula, or scientific threshold is changed.

## Runtime evidence motivating v2

The first two-worker pilot completed successfully but showed very low sustained GPU utilization. A subsequent four-worker pilot failed in one worker with `OSError: [Errno 12] Cannot allocate memory`. During that failed four-worker pilot the GPU still averaged only about 7.7% SM utilization and roughly 1 GiB framebuffer allocation, demonstrating that host-memory duplication rather than GPU capacity was the immediate scaling limit. The partial pilot wrote only probe-train development markers; probe-test remained locked.

Do not solve this failure by adding swap or blindly increasing worker count. Swap can hide an allocation failure while making the mounted-dataset path substantially slower and does not address duplicated dataset metadata.

## Required v2 equivalence check

Before any further formal train extraction, run:

```bash
pytest -q tests/test_care3d_p2a_association.py tests/test_care3d_p2a_speed.py
python scripts/check_care3d_p2a_shared_dataset_equivalence.py
```

The second command uses only the excluded engineering scene and compares canonical independently built fault datasets against the shared-info views on frozen frames 3 and 12 for Blur, Crash, and Dark. Image tensors, tensor-valued model inputs, and metadata must be exactly equal. Required status:

`P2A_SHARED_DATASET_EQUIVALENCE_PASSED`

`probe_test_read` must remain false.

## Worker selection

After v2 equivalence passes, benchmark workers based on **throughput and host-memory safety**, not VRAM occupancy. Start with three workers. Record host RAM/swap and GPU SM utilization. Test four workers again only if three workers complete without memory pressure and sufficient host-memory headroom remains.

The mounted F: drive can become the next bottleneck. If increasing workers does not improve scenes/minute, stop increasing concurrency even when GPU utilization is low.

## Pilot outputs and implementation-policy changes

Execution pilots are probe-train development runs only. When changing from v1 to v2, do not mix v1 pilot scene outputs into the later formal v2 extraction. Preserve them for audit by moving the existing per-scene pilot artifacts to an `outputs/` archive before the v2 formal run. Do not inspect efficacy metrics from pilot outputs.

## Formal sequence

Only after v2 unit tests, excluded-scene equivalence, and a memory-safe worker benchmark pass:

```bash
P2A_WORKERS=<frozen_worker_count> CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_care3d_p2a_train_parallel.sh
python scripts/select_care3d_p2a_train_config.py
python scripts/export_care3d_p2a_association.py --split probe_val --device cuda:0
python scripts/analyze_care3d_p2a_val.py
```

The last three commands must only be reached in order. `probe_test` remains unavailable throughout P2-A0.
