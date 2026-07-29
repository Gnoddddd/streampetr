# RayDN nuScenes-mini screening report

The four pre-registered 50-iteration groups completed. RayDN did not pass the 200-iteration gate for either corresponding baseline.

## Four-protocol metrics

| Experiment | Protocol | mAP | NDS |
|---|---|---:|---:|
| B0 | clean_no_corruption | 0.433410 | 0.482380 |
| B0 | camera_crash_back_5f | 0.421787 | 0.474954 |
| B0 | camera_crash_back_10f | 0.417788 | 0.473632 |
| B0 | compound_fog_crash_10f | 0.391288 | 0.455644 |
| B0_RayDN | clean_no_corruption | 0.426671 | 0.478304 |
| B0_RayDN | camera_crash_back_5f | 0.418604 | 0.472612 |
| B0_RayDN | camera_crash_back_10f | 0.411142 | 0.468911 |
| B0_RayDN | compound_fog_crash_10f | 0.387005 | 0.452579 |
| M1 | clean_no_corruption | 0.426659 | 0.479204 |
| M1 | camera_crash_back_5f | 0.413024 | 0.469133 |
| M1 | camera_crash_back_10f | 0.411156 | 0.469925 |
| M1 | compound_fog_crash_10f | 0.390621 | 0.455839 |
| M1_RayDN | clean_no_corruption | 0.421666 | 0.475009 |
| M1_RayDN | camera_crash_back_5f | 0.421644 | 0.474325 |
| M1_RayDN | camera_crash_back_10f | 0.409227 | 0.467575 |
| M1_RayDN | compound_fog_crash_10f | 0.387483 | 0.452349 |

## Pre-registered decision

| Candidate | Fault mean NDS delta | Clean NDS delta | Improved faults | 200iter |
|---|---:|---:|---:|---|
| B0_RayDN vs B0 | -0.003376 | -0.004076 | 0/3 | no |
| M1_RayDN vs M1 | -0.000216 | -0.004194 | 1/3 | no |

## M1 candidate/write quality

| Experiment | Protocol | RECOVER GT match | False write | Recovery delay |
|---|---|---:|---:|---:|
| M1 | camera_crash_back_5f | 0.3750 | 0.5000 | 0.0 |
| M1 | camera_crash_back_10f | 0.4641 | 0.4250 | 0.0 |
| M1 | compound_fog_crash_10f | 0.1920 | 0.7241 | 0.0 |
| M1_RayDN | camera_crash_back_5f | 0.3556 | 0.5167 | 0.0 |
| M1_RayDN | camera_crash_back_10f | 0.4430 | 0.4370 | 0.0 |
| M1_RayDN | compound_fog_crash_10f | 0.1872 | 0.7308 | 0.0 |

B0+RayDN regressed on all three fault protocols and Clean. M1+RayDN improved Crash5 but regressed on Crash10, Compound and Clean; its fault-mean gain and Clean constraint therefore fail.

## Engineering and interpretation

- Disabled inference is exactly equal on all eight baseline/protocol pairs (`max_abs_diff=0`).
- The final true full test suite passed: 89 passed, 7 warnings.
- Fixed FP16 scale 512 removed the common all-group dynamic-scale overflow; formal smoke and 50iter runs contain no NaN/Inf/OOM.
- M1 conservation and source-mass violation ratios are zero.
- RayDN adds no state-dict/checkpoint key and is absent at inference.
- Mean measured inference rates (frames/s): B0=10.225, B0_RayDN=10.075, M1=8.425, M1_RayDN=8.400.
- FP/FN/recall use a declared score threshold 0.1 and greedy class-aware 2m center matching. Candidate-write matching uses the same 2m rule in the LIDAR frame.
- The mini-screen does not support a complementary RayDN claim and does not authorize 200iter.

Source specification: `RAYDN_ADAPTATION_SPEC.md`.
