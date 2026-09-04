"""Pure state-swap and decision helpers for Stage5 temporal attribution."""

from __future__ import annotations

import itertools
import math

import numpy as np
import torch


def swap_one_component(base: dict, donor: dict, component: str) -> dict:
    """Clone ``base`` and replace exactly one persistent state tensor."""
    if component not in base or set(base) != set(donor):
        raise KeyError(component)
    output = {}
    for name, value in base.items():
        source = donor[name] if name == component else value
        if source is None:
            output[name] = None
        else:
            output[name] = source.detach().clone()
    return output


def assert_one_component_swap(base: dict, donor: dict, swapped: dict,
                              component: str) -> float:
    """Assert non-target tensors are exact and return target donor distance."""
    if set(base) != set(donor) or set(base) != set(swapped):
        raise AssertionError("state keys differ")
    target_difference = 0.0
    for name in base:
        expected = donor[name] if name == component else base[name]
        actual = swapped[name]
        if expected is None or actual is None:
            if expected is not actual:
                raise AssertionError(f"{name}: None mismatch")
            continue
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise AssertionError(f"{name}: shape/dtype mismatch")
        if not torch.equal(expected, actual):
            raise AssertionError(f"{name}: non-exact state swap")
        if name == component:
            source = base[name]
            if source is not None:
                target_difference = float(
                    (source.float() - actual.float()).abs().max().item())
    return target_difference


def explanation_ratio(component_effect: float, full_effect: float) -> dict:
    component_effect, full_effect = float(component_effect), float(full_effect)
    if not math.isfinite(component_effect) or not math.isfinite(full_effect) \
            or abs(full_effect) <= 1e-12:
        return {"raw": float("nan"), "clipped": float("nan")}
    raw = component_effect / full_effect
    return {"raw": float(raw), "clipped": float(np.clip(raw, 0.0, 1.0))}


def _core(record: dict, arm: str) -> bool:
    lost = float(record[f"{arm}_lost"])
    retained = float(record[f"{arm}_retained"])
    enrichment = float(record[f"{arm}_enrichment"])
    if arm == "bd":
        statistical = (lost >= 0.01 and float(record["bd_ci_low"]) > 0.0
                       and enrichment >= 0.01
                       and float(record["bd_enrichment_ci_low"]) > 0.0)
    else:
        statistical = (lost <= -0.01 and float(record["ca_ci_high"]) < 0.0
                       and enrichment <= -0.01
                       and float(record["ca_enrichment_ci_high"]) < 0.0)
    retained_small = abs(retained) <= max(0.01, 0.5 * abs(lost))
    return bool(statistical and record[f"{arm}_cross_protocol"] and retained_small)


def _operational(record: dict, arm: str) -> bool:
    values = (float(record[f"{arm}_topk_ratio"]),
              float(record[f"{arm}_tp_ratio"]))
    return any(math.isfinite(value) and value >= 0.5 for value in values)


def decide_attribution(records: list[dict]) -> dict:
    """Apply the preregistered single-component and sparse-pair gates."""
    evaluated = []
    for source in records:
        record = dict(source)
        for arm in ("bd", "ca"):
            record[f"{arm}_core"] = _core(record, arm)
            record[f"{arm}_arm"] = bool(
                record[f"{arm}_core"]
                and float(record[f"{arm}_spos_ratio"]) >= 0.5
                and _operational(record, arm))
        record["dominant"] = record["bd_arm"] and record["ca_arm"]
        evaluated.append(record)

    dominant = sorted(record["component"] for record in evaluated
                      if record["dominant"])
    sparse = []
    for left, right in itertools.combinations(evaluated, 2):
        if not all(record[f"{arm}_core"] for record in (left, right)
                   for arm in ("bd", "ca")):
            continue
        arm_pass = []
        for arm in ("bd", "ca"):
            spos = float(left[f"{arm}_spos_ratio"]) + float(
                right[f"{arm}_spos_ratio"])
            topk = float(left[f"{arm}_topk_ratio"]) + float(
                right[f"{arm}_topk_ratio"])
            tp = float(left[f"{arm}_tp_ratio"]) + float(
                right[f"{arm}_tp_ratio"])
            arm_pass.append(spos >= 0.5 and max(topk, tp) >= 0.5)
        if all(arm_pass):
            sparse.append(tuple(sorted((left["component"], right["component"]))))
    sparse.sort()

    if dominant:
        decision = "GO_DOMINANT_TEMPORAL_STATE_COMPONENT"
        selected = dominant
    elif sparse:
        decision = "GO_SPARSE_TEMPORAL_STATE_COMPONENT_SET"
        selected = list(sparse[0])
    else:
        decision = "NO_GO_TEMPORAL_STATE_ATTRIBUTION"
        selected = []
    return {"decision": decision, "selected_components": selected,
            "dominant_components": dominant, "passing_pairs": sparse,
            "records": evaluated}
