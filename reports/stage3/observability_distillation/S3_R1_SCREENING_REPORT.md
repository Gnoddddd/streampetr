# S3-R1 screening report

Checkpoints were selected solely by Clean NDS: B0=epoch 3, R0=epoch 1, R1=epoch 1. No fault metric participated in selection.

## Selected-checkpoint metrics

| group | epoch | Clean mAP/NDS | Crash5 mAP/NDS | Crash10 mAP/NDS | Compound mAP/NDS | fault avg NDS |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 3 | 0.419054/0.473282 | 0.408000/0.465364 | 0.405870/0.464871 | 0.379278/0.447774 | 0.459336 |
| R0 | 1 | 0.426250/0.477353 | 0.418232/0.471322 | 0.406708/0.465045 | 0.385942/0.450439 | 0.462269 |
| R1 | 1 | 0.425783/0.478087 | 0.412868/0.468677 | 0.413820/0.471418 | 0.382928/0.450586 | 0.463560 |

Selected checkpoints: B0=`outputs/stage3/observability_distillation/b0/iter_969.pth`, R0=`outputs/stage3/observability_distillation/r0/iter_323.pth`, R1=`outputs/stage3/observability_distillation/r1/iter_323.pth`.

## Six-epoch screening curve

| group | epoch | Clean NDS | fault-average NDS |
|---|---:|---:|---:|
| B0 | 1 | 0.473091 | 0.459785 |
| B0 | 3 | 0.473282 | 0.459336 |
| B0 | 6 | 0.469395 | 0.455080 |
| R0 | 1 | 0.477353 | 0.462269 |
| R0 | 3 | 0.475716 | 0.460619 |
| R0 | 6 | 0.471003 | 0.456502 |
| R1 | 1 | 0.478087 | 0.463560 |
| R1 | 3 | 0.474572 | 0.459417 |
| R1 | 6 | 0.471840 | 0.455605 |

## Gate

Overall decision: **FAIL**.

- clean_non_regression: pass
- fault_average_gain: pass
- two_faults_improve: pass
- no_protocol_regression: pass
- gt_recall_non_regression: pass
- false_write_reduction_10pct: fail
- r1_beats_r0_fault_average: pass
- engineering_stability: pass

Mean GT recall@2m: B0=0.745891, R1=0.750338. Fault-protocol false Top-K memory writes: B0=50970, R1=50904, reduction=0.1295%.

The student inference graph has the same 37,259,345 parameters and the same operations as B0. The disabled four-protocol replay has max_abs_diff=0. The EMA teacher has no gradients, is not registered, and does not appear among the 591 deployment checkpoint keys. Training used only the mini train annotation file; fixed val protocols were evaluation-only.

Because the full preregistered gate is not met, this screening stops here: no distillation-weight tuning, additional seed, or full-data run is authorized.
