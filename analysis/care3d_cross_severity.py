"""Pure helpers for CARE-3D P0 cross-severity transfer.

The transfer experiment reuses the frozen severe-fault P0 predictor without
retraining and evaluates it on lighter Blur/Dark interventions.  This module
contains only detector-independent invariants so they can be unit-tested
without importing StreamPETR.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


TRANSFER_PROTOCOLS = ("blur_s03", "dark_s03")
MAIN_PROTOCOL_INDEX = {
    "blur_s03": 0,  # blur_back head trained at severity 0.9
    "dark_s03": 2,  # dark_back head trained at severity 0.9
}


def assert_exact_sample_alignment(
    main_sample_ids: Sequence[str], transfer_sample_ids: Sequence[str]
) -> None:
    """Require the lighter-severity cohort to be the exact frozen P0 cohort."""
    main = [str(value) for value in main_sample_ids]
    transfer = [str(value) for value in transfer_sample_ids]
    if main != transfer:
        raise RuntimeError("cross-severity sample order/cohort differs from frozen main P0")
    if len(main) != len(set(main)):
        raise RuntimeError("duplicate sample_id in frozen cross-severity cohort")


def assert_predictor_inputs_exact(
    main_arrays: Mapping[str, np.ndarray], transfer_arrays: Mapping[str, np.ndarray]
) -> None:
    """Require bitwise-identical clean-anchor predictor inputs.

    The transfer test is allowed to change only the t+1 intervention severity;
    the clean anchor representation fed to CARE must be exactly the same as in
    the already-frozen main P0 probe-test export.
    """
    keys = (
        "object_features",
        "temporal_features",
        "decision_features",
        "camera_support",
        "camera_quality",
    )
    for key in keys:
        if key not in main_arrays or key not in transfer_arrays:
            raise RuntimeError(f"missing cross-severity predictor array: {key}")
        left = np.asarray(main_arrays[key])
        right = np.asarray(transfer_arrays[key])
        if left.shape != right.shape:
            raise RuntimeError(
                f"cross-severity predictor shape changed for {key}: {right.shape} != {left.shape}"
            )
        if not np.array_equal(left, right):
            raise RuntimeError(f"cross-severity predictor input changed for {key}")


def transfer_gate_flags(
    point: Mapping[str, float],
    scene_ci: Mapping[str, float],
    instance_ci: Mapping[str, float],
    *,
    min_boundary_auroc: float,
    min_boundary_auroc_ci_low: float,
    min_auprc_excess_ci_low: float,
) -> dict[str, bool]:
    """Apply the frozen main-P0 stability criteria to one transfer result."""

    def finite_positive(value: float) -> bool:
        return bool(np.isfinite(value) and value > 0.0)

    rank_pass = bool(
        finite_positive(float(point["spearman"]))
        and finite_positive(float(scene_ci["spearman_ci_low"]))
        and finite_positive(float(instance_ci["spearman_ci_low"]))
    )
    separation_pass = bool(
        finite_positive(float(point["decile_drop_delta"]))
        and finite_positive(float(scene_ci["decile_drop_delta_ci_low"]))
        and finite_positive(float(instance_ci["decile_drop_delta_ci_low"]))
    )
    boundary_auroc_point_pass = bool(
        np.isfinite(float(point["auroc"]))
        and float(point["auroc"]) >= float(min_boundary_auroc)
    )
    boundary_auroc_ci_pass = bool(
        np.isfinite(float(scene_ci["auroc_ci_low"]))
        and np.isfinite(float(instance_ci["auroc_ci_low"]))
        and float(scene_ci["auroc_ci_low"]) > float(min_boundary_auroc_ci_low)
        and float(instance_ci["auroc_ci_low"]) > float(min_boundary_auroc_ci_low)
    )
    boundary_auprc_ci_pass = bool(
        np.isfinite(float(scene_ci["auprc_minus_base_rate_ci_low"]))
        and np.isfinite(float(instance_ci["auprc_minus_base_rate_ci_low"]))
        and float(scene_ci["auprc_minus_base_rate_ci_low"])
        > float(min_auprc_excess_ci_low)
        and float(instance_ci["auprc_minus_base_rate_ci_low"])
        > float(min_auprc_excess_ci_low)
    )
    boundary_pass = bool(
        boundary_auroc_point_pass
        and boundary_auroc_ci_pass
        and boundary_auprc_ci_pass
    )
    passed = bool(rank_pass and separation_pass and boundary_pass)
    return {
        "rank_pass": rank_pass,
        "separation_pass": separation_pass,
        "boundary_auroc_point_pass": boundary_auroc_point_pass,
        "boundary_auroc_ci_pass": boundary_auroc_ci_pass,
        "boundary_auprc_ci_pass": boundary_auprc_ci_pass,
        "boundary_pass": boundary_pass,
        "seed_protocol_pass": passed,
    }
