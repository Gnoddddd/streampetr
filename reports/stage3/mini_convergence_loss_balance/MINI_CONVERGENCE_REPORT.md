# Mini convergence and loss-balance report

The experiment used 323 iterations per mini-equivalent epoch and 3,876 iterations total. All 3 groups completed without NaN, Inf, OOM, or DN losses. Conservation, unsupported-growth, and source-mass violations were zero for both Evidence3D groups.

## Epoch-12 metrics

| experiment | Clean NDS | Crash5 | Crash10 | Compound | fault avg |
|---|---:|---:|---:|---:|---:|
| B0 | 0.470538 | 0.461832 | 0.462936 | 0.443707 | 0.456158 |
| M1 | 0.448262 | 0.442578 | 0.432062 | 0.423845 | 0.432829 |
| M1-Ramp | 0.470722 | 0.461148 | 0.451732 | 0.437602 | 0.450161 |

## Pre-registered gate

- M1: **FAIL** — fault_average_gain=fail, clean_non_regression=fail, two_faults_improve=fail, no_protocol_regression=fail, engineering=pass, epoch6_to_12_not_reversed=pass
- M1-Ramp: **FAIL** — fault_average_gain=fail, clean_non_regression=pass, two_faults_improve=fail, no_protocol_regression=fail, engineering=pass, epoch6_to_12_not_reversed=pass

## Convergence curve

| experiment | epoch | Clean NDS | fault-average NDS | loss_cls | loss_bbox | loss_ternary |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 1 | 0.477949 | 0.464026 | 0.2963 | 0.6593 | 0.0000 |
| B0 | 3 | 0.476228 | 0.464804 | 0.2563 | 0.6348 | 0.0000 |
| B0 | 6 | 0.468728 | 0.454108 | 0.2345 | 0.6126 | 0.0000 |
| B0 | 12 | 0.470538 | 0.456158 | 0.2184 | 0.5870 | 0.0000 |
| M1 | 1 | 0.462512 | 0.452781 | 0.2926 | 0.6891 | 1.1668 |
| M1 | 3 | 0.426923 | 0.416605 | 0.3253 | 0.7518 | 0.5969 |
| M1 | 6 | 0.427506 | 0.412481 | 0.3027 | 0.7313 | 0.5013 |
| M1 | 12 | 0.448262 | 0.432829 | 0.2606 | 0.6843 | 0.3745 |
| M1-Ramp | 1 | 0.479560 | 0.465404 | 0.2876 | 0.6693 | 0.0000 |
| M1-Ramp | 3 | 0.455549 | 0.440740 | 0.2608 | 0.6553 | 0.0445 |
| M1-Ramp | 6 | 0.439733 | 0.419567 | 0.2910 | 0.7095 | 0.4239 |
| M1-Ramp | 12 | 0.470722 | 0.450161 | 0.2426 | 0.6454 | 0.3499 |

## Epoch-12 candidate quality

| experiment | protocol | GT recall@2m | RECOVER GT match | false RECOVER write | recovery delay |
|---|---|---:|---:|---:|---:|
| B0 | clean_no_corruption | 0.7550 | N/A | N/A | N/A |
| B0 | camera_crash_back_5f | 0.7548 | N/A | N/A | 0.0000 |
| B0 | camera_crash_back_10f | 0.7539 | N/A | N/A | 0.0000 |
| B0 | compound_fog_crash_10f | 0.7365 | N/A | N/A | 1.5000 |
| M1 | clean_no_corruption | 0.7149 | 1.0000 | 0.0000 | N/A |
| M1 | camera_crash_back_5f | 0.7140 | 0.1806 | 0.7917 | 1.0000 |
| M1 | camera_crash_back_10f | 0.6976 | 0.2753 | 0.6892 | 1.0000 |
| M1 | compound_fog_crash_10f | 0.6832 | 0.3351 | 0.6169 | 1.0000 |
| M1-Ramp | clean_no_corruption | 0.7341 | 1.0000 | 0.0000 | N/A |
| M1-Ramp | camera_crash_back_5f | 0.7311 | 0.2435 | 0.7400 | 0.0000 |
| M1-Ramp | camera_crash_back_10f | 0.7172 | 0.3774 | 0.6183 | 1.0000 |
| M1-Ramp | compound_fog_crash_10f | 0.7037 | 0.5025 | 0.4624 | 0.0000 |

M1-Ramp substantially improves M1 candidate quality, but its epoch-12 GT recall remains below B0 on every protocol. Its Crash10 and Compound NDS regressions exceed the allowed 0.002, so better RECOVER behavior is not sufficient to establish an overall gain.

## Decision

A pass permits quality-estimation work in the next task. If neither candidate passes, no module should be stacked; the Evidence3D core training objective must be revised first. **Neither candidate passes, therefore quality estimation is not authorized by this screen.**
