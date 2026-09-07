# CARE-3D P1 implementation note: FP32 query replay cache

## Trigger

During the first formal P1 router-training attempt (seed 42), training stopped at epoch 2 before any P1 checkpoint was finalized and before probe-test was opened. The frozen classifier replay guard observed a maximum target-score discrepancy of `0.0005164146423339844`, slightly above the preregistered `5e-4` tolerance.

## Root cause

The P1 supervision exporter computed `clean_score` / `fault_score` from the live full-precision StreamPETR forward pass, but cached `clean_query` / `fault_query` through float16 arrays. Training later reloaded those quantized queries, cast them to float32, and replayed the frozen classifier. The resulting comparison therefore measured float16 serialization error rather than a change in detector or classifier behavior.

The engineering smoke scene had passed because its worst-case float16 round-trip error (`0.000164031982421875`) happened to be below `5e-4`; it was not a proof that every train/val row satisfied the same tolerance.

## Frozen repair

The replay tolerance is **not relaxed**. Instead, formal P1 router-supervision caches are regenerated with FP32 storage under policy id:

`fp32_router_supervision_v1`

The compatibility wrapper `scripts/export_care3d_p1_supervision_fp32.py` changes only exporter cache precision. It leaves the following frozen components unchanged:

- official StreamPETR checkpoint and detector parameters;
- frozen P0 predictors and P0 inputs;
- 419 / 133 / 132 scene split;
- shared-query collision policy;
- source bank and top-k routing rule;
- labels, loss weights, optimizer settings and seeds;
- `5e-4` classifier replay tolerance;
- all P1 Go / No-Go thresholds;
- probe-test lock.

Old supervision markers are stale for this wrapper unless they carry `storage_precision_policy=fp32_router_supervision_v1`, forcing engineering smoke and all 552 train/val scenes to be regenerated before training restarts.

## Restart rule

The incomplete seed-42 attempt must not be resumed. Any partial P1 training artifacts from the float16-cache attempt are archived or removed, and all three seeds `42 / 2027 / 2028` restart from initialization only after the FP32 engineering smoke and 552-scene supervision export pass.

No probe-test result was inspected in defining this repair.
