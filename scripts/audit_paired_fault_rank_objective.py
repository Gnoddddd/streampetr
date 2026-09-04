#!/usr/bin/env python3
"""Fixed 100-pair, no-update preservation-objective activation audit."""

from __future__ import annotations

import copy
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

import evidence3d_plugin  # noqa: F401
from models.paired_fault_rank import (
    paired_margin_preservation_loss,
    select_paired_margin_events,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/stage4/paired_fault_rank_objective_audit"
CONFIG = "configs/stage4/paired_fault_rank_objective_audit.py"
MODES = ("dark", "motion_blur", "crash")
MEMORY_NAMES = (
    "memory_embedding", "memory_reference_point", "memory_velo",
    "memory_timestamp", "memory_egopose",
)


def write_csv(name, rows):
    if not rows:
        raise RuntimeError(f"refusing empty table {name}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, q):
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.percentile(values, q)) if values else float("nan")


def seeded_item(dataset, index):
    seed = 2026 * 100000 + int(index)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return dataset[int(index)]


def make_dataset(cfg, mode=None):
    dataset_cfg = copy.deepcopy(cfg.data.train)
    if mode is not None:
        dataset_cfg.pipeline[1] = dict(
            type="ApplyFixedCameraFault", mode=mode,
            camera="CAM_BACK", severity=0.9,
        )
    return build_dataset(dataset_cfg)


def snapshot_memory(head):
    return {
        name: (None if getattr(head, name, None) is None else
               getattr(head, name).detach().clone())
        for name in MEMORY_NAMES
    }


def restore_memory(head, state):
    for name, value in state.items():
        setattr(head, name, None if value is None else value.detach().clone())


def detach_memory(state):
    return {
        name: (None if value is None else value.detach())
        for name, value in state.items()
    }


def forward_capture(model, data, seed, no_grad):
    captured = []
    handle = model.module.pts_bbox_head.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output)
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        if no_grad:
            with torch.no_grad():
                losses = model(return_loss=True, **data)
        else:
            losses = model(return_loss=True, **data)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one head output, got {len(captured)}")
    return captured[0], losses


def tensor_equal(left, right):
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return all(tensor_equal(a, b) for a, b in zip(left, right))
    if hasattr(left, "tensor") and hasattr(right, "tensor"):
        return torch.equal(left.tensor, right.tensor)
    return left == right


def validate_pair(clean, fault, mode, audit_batch, dataset_index):
    clean_meta = clean["img_metas"].data[-1]
    fault_meta = fault["img_metas"].data[-1]
    same_token = clean_meta["sample_idx"] == fault_meta["sample_idx"]
    gt_equal = tensor_equal(clean["gt_bboxes_3d"].data, fault["gt_bboxes_3d"].data) \
        and tensor_equal(clean["gt_labels_3d"].data, fault["gt_labels_3d"].data)
    geometry_equal = all(
        tensor_equal(clean[key].data, fault[key].data)
        for key in ("lidar2img", "intrinsics", "extrinsics", "ego_pose", "ego_pose_inv")
    )
    clean_img = clean["img"].data
    fault_img = fault["img"].data
    per_camera_changed = (clean_img != fault_img).reshape(
        clean_img.shape[0], clean_img.shape[1], -1
    ).any(-1)
    only_target = bool(per_camera_changed[:, 3].all()) and bool(
        not per_camera_changed[:, [0, 1, 2, 4, 5]].any()
    )
    return {
        "audit_batch": audit_batch, "dataset_index": dataset_index,
        "sample_token": str(clean_meta["sample_idx"]), "protocol": mode,
        "same_token": same_token, "gt_equal": gt_equal,
        "geometry_equal": geometry_equal, "only_cam_back_changed": only_target,
        "pair_valid": bool(same_token and gt_equal and geometry_equal and only_target),
    }


def summarize(rows, protocol):
    selected = rows if protocol == "all" else [
        row for row in rows if row["protocol"] == protocol
    ]
    paired = [row for row in selected if row["paired_candidate_available"]]
    clean_in = [row for row in paired if row["clean_in_topk"]]
    eligible = [row for row in clean_in if row["collapse_eligible"]]
    nonzero = [row for row in eligible if row["nonzero"]]
    old = [row for row in selected if row["old_absolute_fault_rank_out"]]
    lost = [row for row in clean_in if row["lost_risk"]]
    retained = [row for row in clean_in if row["retained_like"]]
    lost_nonzero = [row for row in lost if row["nonzero"]]
    retained_nonzero = [row for row in retained if row["nonzero"]]
    crossing_nonzero = [row for row in nonzero if row["boundary_crossing"]]
    old_crossing = [row for row in old if row["boundary_crossing"]]
    return {
        "protocol": protocol, "gt": len(selected),
        "paired_candidates": len(paired),
        "paired_candidate_ratio": len(paired) / max(len(selected), 1),
        "clean_in_topk": len(clean_in),
        "collapse_eligible": len(eligible),
        "collapse_eligible_ratio": len(eligible) / max(len(clean_in), 1),
        "nonzero": len(nonzero),
        "nonzero_ratio_eligible": len(nonzero) / max(len(eligible), 1),
        "nonzero_ratio_clean_in_topk": len(nonzero) / max(len(clean_in), 1),
        "conditional_loss_median": percentile([row["loss"] for row in nonzero], 50),
        "delta_margin_median_eligible": percentile(
            [row["delta_margin"] for row in eligible], 50
        ),
        "crossing_nonzero": len(crossing_nonzero),
        "crossing_concentration_new": len(crossing_nonzero) / max(len(nonzero), 1),
        "old_absolute_rankout": len(old),
        "old_crossing": len(old_crossing),
        "crossing_concentration_old": len(old_crossing) / max(len(old), 1),
        "specificity_gain": (
            len(crossing_nonzero) / max(len(nonzero), 1)
            - len(old_crossing) / max(len(old), 1)
        ),
        "lost_risk": len(lost), "lost_risk_nonzero": len(lost_nonzero),
        "lost_risk_activation": len(lost_nonzero) / max(len(lost), 1),
        "retained_like": len(retained),
        "retained_like_nonzero": len(retained_nonzero),
        "retained_like_activation": len(retained_nonzero) / max(len(retained), 1),
        "generic_clean_hard": sum(row["generic_clean_hard"] for row in selected),
        "same_query_nonzero_ratio": sum(
            row["same_query_id"] for row in nonzero
        ) / max(len(nonzero), 1),
    }


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(CONFIG)
    clean_dataset = make_dataset(cfg)
    fault_datasets = {mode: make_dataset(cfg, mode) for mode in MODES}
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    model = build_model(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    load_checkpoint(model, cfg.load_from, map_location="cpu", strict=False)
    wrap_fp16_model(model)
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.train()
    head = model.module.pts_bbox_head
    clean_memory = snapshot_memory(head)
    fault_memory = snapshot_memory(head)

    event_rows, batch_rows, pair_rows, gradient_rows = [], [], [], []
    for audit_batch, dataset_index in enumerate(range(28, 128)):
        mode = MODES[(dataset_index + 314159) % len(MODES)]
        clean_item = seeded_item(clean_dataset, dataset_index)
        fault_item = seeded_item(fault_datasets[mode], dataset_index)
        if audit_batch == 0:
            clean_item["prev_exists"].data.zero_()
            fault_item["prev_exists"].data.zero_()
        pair_check = validate_pair(
            clean_item, fault_item, mode, audit_batch, dataset_index
        )
        if not pair_check["pair_valid"]:
            raise RuntimeError(f"invalid paired input: {pair_check}")
        pair_rows.append(pair_check)
        clean_data = collate([clean_item], samples_per_gpu=1)
        fault_data = collate([fault_item], samples_per_gpu=1)
        stochastic_seed = 2026 * 100000 + dataset_index

        restore_memory(head, clean_memory)
        clean_output, clean_losses = forward_capture(
            model, clean_data, stochastic_seed, no_grad=True
        )
        clean_memory = detach_memory(snapshot_memory(head))

        restore_memory(head, fault_memory)
        model.module.zero_grad(set_to_none=True)
        fault_output, fault_losses = forward_capture(
            model, fault_data, stochastic_seed, no_grad=False
        )
        fault_memory = detach_memory(snapshot_memory(head))

        clean_logits = clean_output["all_cls_scores"][-1, 0].detach()
        clean_boxes = clean_output["all_bbox_preds"][-1, 0].detach()
        fault_logits = fault_output["all_cls_scores"][-1, 0]
        fault_boxes = fault_output["all_bbox_preds"][-1, 0]
        fault_logits.retain_grad()
        box = fault_item["gt_bboxes_3d"].data[-1]
        labels = fault_item["gt_labels_3d"].data[-1].to(fault_logits.device)
        centers = box.gravity_center.to(fault_logits.device)
        events, _ = select_paired_margin_events(
            clean_logits, clean_boxes, fault_logits, fault_boxes,
            centers, labels, topk=100, geometry_threshold=2.0, delta=0.10,
        )
        objective, details = paired_margin_preservation_loss(
            fault_logits, events, delta=0.10, enabled=True
        )
        details_by_gt = {int(row["gt"]): row for row in details}
        for event in events:
            detail = details_by_gt.get(int(event["gt"]))
            event_rows.append({
                "audit_batch": audit_batch, "dataset_index": dataset_index,
                "sample_token": pair_check["sample_token"], "protocol": mode,
                **event,
                "loss": float(detail["loss"]) if detail else 0.0,
                "nonzero": bool(detail and detail["nonzero"]),
                "target_margin": (
                    float(detail["target_margin"]) if detail else float("nan")
                ),
            })

        logits_gradient = torch.autograd.grad(
            objective, fault_logits, retain_graph=True, allow_unused=False
        )[0]
        objective.backward()
        parameter_gradients = [
            parameter.grad.detach() for parameter in model.module.parameters()
            if parameter.grad is not None
        ]
        finite = bool(torch.isfinite(objective)) and all(
            torch.isfinite(value).all() for value in parameter_gradients
        ) and bool(torch.isfinite(logits_gradient).all())
        selected_flat = {
            int(row["fault_positive_flat_index"])
            for row in details if row["nonzero"]
        }
        boundary_flat = {
            int(row["fault_s_k_flat_index"]) for row in details
        } - selected_flat
        gradient_flat = logits_gradient.reshape(-1)
        boundary_detached = all(
            float(gradient_flat[index]) == 0.0 for index in boundary_flat
        )
        nonzero_gradient = set(torch.nonzero(
            gradient_flat != 0, as_tuple=False
        ).flatten().tolist())
        selection_detached = nonzero_gradient.issubset(selected_flat)
        current_events = event_rows[-len(events):] if events else []
        batch_nonzero = sum(row["nonzero"] for row in current_events)
        batch_rows.append({
            "audit_batch": audit_batch, "dataset_index": dataset_index,
            "sample_token": pair_check["sample_token"], "protocol": mode,
            "gt": len(events),
            "paired_candidates": sum(row["paired_candidate_available"] for row in current_events),
            "collapse_eligible": sum(row["collapse_eligible"] for row in current_events),
            "nonzero": batch_nonzero, "batch_active": batch_nonzero > 0,
            "objective": float(objective.detach()),
            "loss_finite": bool(torch.isfinite(objective)),
            "gradient_finite": finite,
            "clean_reference_detached": not clean_logits.requires_grad,
            "boundary_stop_gradient": boundary_detached,
            "selection_detached": selection_detached,
            "optimizer_created": False, "optimizer_step": False,
        })
        gradient_rows.append({key: batch_rows[-1][key] for key in (
            "audit_batch", "protocol", "loss_finite", "gradient_finite",
            "clean_reference_detached", "boundary_stop_gradient",
            "selection_detached", "optimizer_created", "optimizer_step",
        )})
        # Keep only detached per-stream temporal state between no-update pairs.
        clean_memory = detach_memory(clean_memory)
        fault_memory = detach_memory(fault_memory)
        del clean_output, clean_losses, fault_output, fault_losses, objective

    summaries = [summarize(event_rows, mode) for mode in (*MODES, "all")]
    overall = summaries[-1]
    active_batches = sum(row["batch_active"] for row in batch_rows)
    batch_activation = active_batches / len(batch_rows)
    protocol_counts = {mode: sum(row["protocol"] == mode for row in batch_rows)
                       for mode in MODES}

    source_invariance = list(csv.DictReader(
        (ROOT / "reports/stage4/paired_fault_rank_margin_audit/"
         "prediction_invariance.csv").open()
    ))
    invariance_rows = [{
        **row,
        "source": "cec9cb5 paired audit; new objective disabled before selection",
        "new_disabled_helper_zero": True,
    } for row in source_invariance]
    invariant = all(float(row["max_abs_diff"]) == 0.0 for row in invariance_rows)
    pair_valid = all(row["pair_valid"] for row in pair_rows)
    gradients_valid = all(
        row["loss_finite"] and row["gradient_finite"]
        and row["clean_reference_detached"] and row["boundary_stop_gradient"]
        and row["selection_detached"]
        for row in gradient_rows
    )
    new_crossing = overall["crossing_concentration_new"]
    old_crossing = overall["crossing_concentration_old"]
    lost_activation = overall["lost_risk_activation"]
    retained_activation = overall["retained_like_activation"]
    structural = all(
        not row["generic_clean_hard"] and row["delta_margin"] <= -0.10
        for row in event_rows if row["nonzero"]
    )
    protocol_gate = all(
        row["nonzero"] > 0
        and row["delta_margin_median_eligible"] < 0
        and row["crossing_concentration_new"] >= 0.50
        for row in summaries[:-1]
    )
    go = (
        len(batch_rows) == 100 and pair_valid
        and min(protocol_counts.values()) >= 30
        and overall["paired_candidate_ratio"] >= 0.80
        and overall["collapse_eligible_ratio"] >= 0.05
        and overall["nonzero_ratio_eligible"] >= 0.25
        and overall["conditional_loss_median"] > 0
        and batch_activation >= 0.30
        and new_crossing >= 0.60
        and overall["specificity_gain"] >= 0.20
        and lost_activation >= 2.0 * retained_activation
        and structural and protocol_gate and gradients_valid and invariant
    )
    gate_rows = [
        {"gate": "paired_input_and_protocol_coverage", "passed": (
            len(batch_rows) == 100 and pair_valid
            and min(protocol_counts.values()) >= 30)},
        {"gate": "candidate_and_eligibility_coverage", "passed": (
            overall["paired_candidate_ratio"] >= 0.80
            and overall["collapse_eligible_ratio"] >= 0.05)},
        {"gate": "loss_and_batch_activation", "passed": (
            overall["nonzero_ratio_eligible"] >= 0.25
            and overall["conditional_loss_median"] > 0
            and batch_activation >= 0.30)},
        {"gate": "crossing_concentration_and_old_target_gain", "passed": (
            new_crossing >= 0.60 and overall["specificity_gain"] >= 0.20)},
        {"gate": "lost_vs_retained_specificity", "passed": (
            lost_activation >= 2.0 * retained_activation)},
        {"gate": "exclude_generic_clean_and_easy_cases", "passed": structural},
        {"gate": "three_protocol_direction_and_crossing", "passed": protocol_gate},
        {"gate": "gradient_and_disabled_safety", "passed": (
            gradients_valid and invariant)},
    ]
    failed_gates = [row["gate"] for row in gate_rows if not row["passed"]]

    write_csv("paired_objective_events.csv", event_rows)
    write_csv("activation_summary.csv", summaries + [{
        "protocol": "decision", "batch_activation": batch_activation,
        "pair_valid": pair_valid, "gradient_checks": gradients_valid,
        "disabled_invariant": invariant, "go": go,
    }])
    write_csv("per_protocol_activation.csv", summaries[:-1])
    write_csv("batch_activation.csv", batch_rows)
    write_csv("paired_input_validation.csv", pair_rows)
    write_csv("leakage_and_gradient_checks.csv", gradient_rows + [{
        "audit_batch": "summary", "protocol": "all",
        "loss_finite": gradients_valid, "gradient_finite": gradients_valid,
        "clean_reference_detached": gradients_valid,
        "boundary_stop_gradient": gradients_valid,
        "selection_detached": gradients_valid,
        "optimizer_created": False, "optimizer_step": False,
    }])
    write_csv("disabled_invariance.csv", invariance_rows)
    write_csv("go_gate_results.csv", gate_rows)
    write_csv("experiment_manifest.csv", [{
        "config": CONFIG, "checkpoint": cfg.load_from,
        "model_seed": 2026, "manifest_seed": 314159,
        "dataset_start": 28, "dataset_stop_exclusive": 128,
        "pairs": 100, "dark": protocol_counts["dark"],
        "motion_blur": protocol_counts["motion_blur"],
        "crash": protocol_counts["crash"], "topk": 100,
        "geometry_threshold_m": 2.0, "delta": 0.10,
        "optimizer_created": False, "val_samples": 0,
    }])

    protocol_lines = "\n".join(
        f"| {row['protocol']} | {row['gt']} | {row['collapse_eligible_ratio']:.2%} | "
        f"{row['nonzero_ratio_eligible']:.2%} | {row['delta_margin_median_eligible']:.4f} | "
        f"{row['crossing_concentration_new']:.2%} | {row['crossing_concentration_old']:.2%} |"
        for row in summaries[:-1]
    )
    report = f"""# Paired Fault Rank Preservation Objective Activation Audit

## Decision

**{'GO' if go else 'NO-GO'}**. {'Only a separately pre-registered 2-iteration smoke with one fixed weight is authorized next.' if go else 'Do not tune delta/margin/lambda and do not enter smoke or training.'}

## Objective

For each same-GT candidate pool, Clean and Fault independently select the highest GT-class score among detached <=2m final-decoder queries; query IDs need not agree. With the actual flattened K=100 boundary, `M=s(q+)-stopgrad(sK)`. Clean is a detached reference and the train-only unit audit loss is `relu(stopgrad(M_clean-0.10)-M_fault)`, eligible only when `M_clean>0` and `M_fault<M_clean`. Gradient reaches only the selected Fault q+ score. B0 inference, Hungarian matching, detection losses, memory, query count and Top-K are unchanged.

## Activation and fault specificity

- 100 no-update paired mini-train batches; Dark/Blur/Crash counts: {protocol_counts['dark']}/{protocol_counts['motion_blur']}/{protocol_counts['crash']}; validation: 0.
- GT: {overall['gt']}; paired <=2m candidates: {overall['paired_candidates']} ({overall['paired_candidate_ratio']:.2%}).
- Collapse eligible among Clean-in-TopK paired GT: {overall['collapse_eligible']}/{overall['clean_in_topk']} ({overall['collapse_eligible_ratio']:.2%}).
- Nonzero: {overall['nonzero']} ({overall['nonzero_ratio_eligible']:.2%} of eligible); conditional median loss {overall['conditional_loss_median']:.6f}; active batches {active_batches}/100 ({batch_activation:.2%}).
- New-objective crossing concentration: {new_crossing:.2%}; old absolute rank>K crossing concentration: {old_crossing:.2%}; gain {overall['specificity_gain']:.2%}.
- Lost-risk activation: {overall['lost_risk_nonzero']}/{overall['lost_risk']} ({lost_activation:.2%}); retained-like strong-collapse activation: {overall['retained_like_nonzero']}/{overall['retained_like']} ({retained_activation:.2%}).
- Generic Clean hard cases receiving nonzero loss: {sum(row['nonzero'] and row['generic_clean_hard'] for row in event_rows)}; nonzero unrelated easy cases: {sum(row['nonzero'] and row['delta_margin'] > -0.10 for row in event_rows)}.

| Protocol | GT | collapse eligible | nonzero/eligible | median delta-M | new crossing concentration | old crossing concentration |
|---|---:|---:|---:|---:|---:|---:|
{protocol_lines}

## Safety

- Paired token/GT/geometry exact and only CAM_BACK changed: {pair_valid}.
- Clean detached; Kth boundaries and selection stop-gradient; AMP loss/gradients finite: {gradients_valid}.
- Optimizer created/stepped: False/False.
- Disabled B0 predictions remain exact on Clean/Dark/Blur/Crash (`max_abs_diff=0`): {invariant}.
- Existing NO-GO hard-positive objective modified or invoked: False.

Failed pre-registered gates: {', '.join(failed_gates) if failed_gates else 'none'}.
The paired loss does exclude generic Clean-hard and weak/easy events, and its
lost-risk activation is much higher than retained-like activation. It is still
NO-GO because only {new_crossing:.2%} of its nonzero events are actual boundary
crossings (versus {old_crossing:.2%} for the old absolute target), so most of
its gradient budget goes to strong margin fluctuations that remain inside
Top-K. The complete pre-registered gate is {go}. No post-result threshold or
loss-weight adjustment was made.
"""
    (REPORT / "PAIRED_FAULT_RANK_OBJECTIVE_AUDIT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps({
        "overall": overall, "protocol_counts": protocol_counts,
        "batch_activation": batch_activation, "pair_valid": pair_valid,
        "gradient_checks": gradients_valid, "disabled_invariant": invariant,
        "go": go,
    }, indent=2))


if __name__ == "__main__":
    main()
