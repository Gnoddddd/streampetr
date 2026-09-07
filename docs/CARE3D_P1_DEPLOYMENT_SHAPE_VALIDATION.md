# CARE-3D P1 deployment-shape validation evidence

Pre-test diagnosis on the frozen 552-scene train/val supervision cohort:

- scenes checked: 552;
- protocol-object rows checked: 356,568;
- query cache: FP32;
- source feature cache: FP16;
- classifier parameter dtype: FP32;
- CUDA matmul TF32: enabled;
- probe-test read: false;
- frozen replay tolerance: `5e-4`.

Observed maximum target-score absolute replay differences:

- legacy trainer-style `[512,256]`: `0.0005164146423339844`;
- packed deployed-shape `[1,900,256]`: `1.1920928955078125e-07`;
- scene/frame/query-index-matched `[1,900,256]`: `1.1920928955078125e-07`.

These measurements justify replacing the standalone classifier execution path
with `packed_deployment_shape_900_v1` without changing the frozen tolerance.
