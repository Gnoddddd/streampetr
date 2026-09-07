# CARE-3D P1 implementation amendment: deployment-shape classifier replay

## Trigger

The first two formal P1 router-training attempts stopped before any P1 seed
checkpoint was frozen and before probe-test was opened. The frozen replay guard
observed the same maximum target-score discrepancy:

`0.0005164146423339844 > 5e-4`.

Regenerating all 552 train/val supervision scenes with FP32 query storage did not
change this maximum, proving that float16 query serialization was not the full
cause.

## Pre-test diagnosis

A test-blind diagnostic read only the frozen 419/133 train/val supervision cohort
and compared three classifier execution paths across 356,568 protocol-object
rows:

- original P1 trainer standalone batch `[512, 256]`:
  `max_abs_diff = 0.0005164146423339844`;
- packed deployed-shape replay `[1, 900, 256]`:
  `max_abs_diff = 1.1920928955078125e-07`;
- scene/frame/query-index-matched deployed-shape replay `[1, 900, 256]`:
  `max_abs_diff = 1.1920928955078125e-07`.

The classifier parameters are FP32 and CUDA matmul TF32 is enabled on the formal
stack. The diagnosis therefore localizes the failed invariant to classifier
execution shape / CUDA math-path drift, not to corrupted supervision labels or
query-cache precision.

`probe_test_read` remained `false` throughout diagnosis.

## Frozen repair

The replay tolerance remains `5e-4`; it is not relaxed.

The formal classifier execution policy is:

`packed_deployment_shape_900_v1`.

For a standalone P1 batch `[B, 256]`, `B <= 900`, queries are concatenated with
zero padding to form `[1, 900, 256]`. The frozen StreamPETR final classifier is
executed once and the first `B` output rows are returned. Concatenation preserves
gradients from routed queries to P1 router parameters.

Native detector calls already shaped `[1, 900, 256]` pass through unchanged. The
shim is installed on the existing classifier instance instead of replacing the
module, preserving checkpoint keys, module identity and forward hooks used by
StreamPETR tap capture.

Both fault-query replay and routed-query classification use the same deployed
shape during P1 training. Formal probe-test evaluation uses the same policy for
standalone routed-query classification while leaving native full-detector calls
unchanged.

## Unchanged frozen components

This amendment changes no supervision row, label, P0 model, StreamPETR weight,
source bank, loss term, loss weight, optimizer, learning rate, seed, train/val/test
split, query-collision rule, FP32 query-storage policy, early-stopping rule,
replay tolerance, bootstrap procedure or P1 Go/No-Go threshold.

The existing 552-scene FP32 supervision cache remains valid and must not be
regenerated again for this repair.

## Restart rule

Any incomplete P1 training artifacts produced before this execution policy are
archived. Seeds `42`, `2027`, `2028` restart from initialization using
`scripts/train_care3d_p1_deployment_shape.py`.

Probe-test remains locked until all three new training manifests report
`P1_TRAINING_COMPLETE_TEST_UNSEEN`, `probe_test_read=false`, and
`classifier_execution_policy=packed_deployment_shape_900_v1`.

Only then may formal probe-test evaluation run through
`scripts/evaluate_care3d_p1_deployment_shape.py`.
