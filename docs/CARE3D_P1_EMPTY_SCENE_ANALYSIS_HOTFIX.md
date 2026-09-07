# CARE-3D P1 zero-object-scene analysis I/O hotfix

## Trigger

The formal 132-scene P1 probe-test completed under the frozen
`packed_deployment_shape_900_v1` classifier execution policy.  The first formal
analysis invocation then stopped in `load_evaluation()` with
`pandas.errors.EmptyDataError: No columns to parse from file` while reading one
`*.objects.csv` file.

No P1 metric table, bootstrap confidence interval, protocol pass/fail flag or
final Go/No-Go decision was produced before this stop.  The frozen gate and all
132 probe-test outputs therefore remain untouched.

## Root cause

The evaluator accumulates object-level rows per scene and writes them with
`pd.DataFrame(object_rows).to_csv(...)`.  A frozen probe-test scene may
legitimately contain zero P1-eligible object rows while still producing the
full frame-level deployment statistics.  When `object_rows == 0`, pandas writes
a headerless empty object CSV.  The original analyzer unconditionally calls
`pd.read_csv()` on every object CSV, so a legitimate zero-row scene is parsed as
an I/O error before any formal metric is computed.

## Frozen repair

The analysis wrapper `scripts/analyze_care3d_p1_empty_scene_safe.py` changes only
file-loading semantics:

- a headerless empty object CSV is accepted only when the scene completion
  marker declares `object_rows == 0`;
- nonzero marker counts paired with an empty CSV remain a hard failure;
- all nonempty object and frame CSVs must match their marker-declared row counts;
- frame-level rows from zero-object scenes remain included, preserving the full
  132-scene FP-inflation analysis;
- every scene marker must retain the frozen scene-manifest hash and
  `packed_deployment_shape_900_v1` execution policy;
- the existing P1 metric definitions, bootstrap seeds, 5000 repetitions, gate
  thresholds, protocol aggregation and Go/No-Go logic are reused unchanged from
  `scripts/analyze_care3d_p1.py`.

This is a post-test, pre-result I/O-integrity repair.  It does not retrain P1,
rerun or alter probe-test inference, drop object-bearing scenes, inspect partial
formal metrics, or retune any threshold.

## Validation and formal rerun

Run the focused I/O tests before the formal analysis rerun:

```bash
pytest -q tests/test_care3d_p1_analysis_empty_scene.py
```

Then rerun the same frozen 5000-bootstrap decision through the strict wrapper:

```bash
python scripts/analyze_care3d_p1_empty_scene_safe.py
```

Do not pass `--bootstraps` for the confirmatory rerun.
