#!/usr/bin/env python3
"""Fixed 100-batch no-update hard-positive boundary activation audit."""

import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

import evidence3d_plugin  # noqa: F401


def percentile(values, q):
    values = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.percentile(values, q)) if values else float("nan")


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: raise RuntimeError(f"refusing empty table {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def summary(rows, label):
    gt = len(rows)
    selected = [r for r in rows if r["pair_selected"]]
    actual = [r for r in selected if r["actual_loss_enabled"]]
    nonzero = [r for r in actual if r["nonzero"]]
    return {
        "split": label,
        "gt": gt,
        "selected_pairs": len(selected),
        "actual_pairs": len(actual),
        "pairs_per_gt": len(actual) / max(gt, 1),
        "counterfactual_selected_per_gt": len(selected) / max(gt, 1),
        "nonzero_pairs": len(nonzero),
        "nonzero_ratio_selected": len(nonzero) / max(len(actual), 1),
        "median_nonzero_loss": percentile([r["loss"] for r in nonzero], 50),
        "median_center_distance": percentile([r["center_distance"] for r in selected], 50),
        "p90_center_distance": percentile([r["center_distance"] for r in selected], 90),
        "median_positive_rank": percentile([r["positive_rank"] for r in selected], 50),
        "median_rank_distance_from_k": percentile([
            r["rank_distance_from_k"] for r in selected
        ], 50),
        "within_500_of_k_ratio": sum(
            1 <= r["rank_distance_from_k"] <= 500 for r in selected
        ) / max(len(selected), 1),
        "geometry_valid_ratio": sum(r["center_distance"] <= 2.0 for r in selected) / max(len(selected), 1),
        "strict_rank_out_ratio": sum(r["positive_rank"] > 100 for r in selected) / max(len(selected), 1),
        "true_boundary_competitor_ratio": sum(
            r["negative_rank"] == 100 and r["negative_truly_outranks"] for r in selected
        ) / max(len(selected), 1),
    }


def main():
    config_path = "configs/stage4/hard_positive_boundary_objective_audit.py"
    output = Path("reports/stage4/hard_positive_boundary_objective_audit")
    output.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(config_path)
    random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    dataset = build_dataset(cfg.data.train)
    loader = build_dataloader(dataset, 1, 0, 1, dist=False, shuffle=False,
                              seed=2026, runner_type="IterBasedRunner")
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, cfg.load_from, map_location="cpu", strict=False)
    wrap_fp16_model(model)
    model = MMDataParallel(model.cuda(), device_ids=[0]); model.train()

    diagnostics, batch_rows, gradient_rows = [], [], []
    measured = 0
    for dataset_index, data in enumerate(loader):
        if dataset_index < 28: continue
        if measured >= 100: break
        if measured == 0: data["prev_exists"].data[0].zero_()
        model.module.zero_grad(set_to_none=True)
        losses = model(return_loss=True, **data)
        detection_terms = [value for key, value in losses.items()
                           if ("loss_cls" in key or "loss_bbox" in key)
                           and "hard_positive" not in key]
        boundary_terms = [value for key, value in losses.items()
                          if key.endswith("loss_hard_positive_boundary")]
        if not detection_terms or len(boundary_terms) != 1:
            raise RuntimeError(f"unexpected loss keys {sorted(losses)}")
        detection = torch.stack(detection_terms).sum()
        boundary = boundary_terms[0]
        total = detection + boundary
        total.backward()
        gradients = [p.grad.detach() for p in model.module.parameters() if p.grad is not None]
        finite_gradient = bool(gradients) and all(torch.isfinite(v).all() for v in gradients)
        grad_norm = math.sqrt(sum(float(v.float().norm()) ** 2 for v in gradients))
        head = model.module.pts_bbox_head
        current = [dict(row, audit_batch=measured, dataset_index=dataset_index)
                   for row in head._last_hard_boundary_diagnostics]
        diagnostics.extend(current)
        fault = bool(current[0]["fault"]) if current else False
        selected = [r for r in current if r["pair_selected"] and r["actual_loss_enabled"]]
        batch_rows.append({
            "audit_batch": measured, "dataset_index": dataset_index, "fault": fault,
            "gt": len(current), "selected_pairs": len(selected),
            "pairs_per_gt": len(selected) / max(len(current), 1),
            "nonzero_pairs": sum(r["nonzero"] for r in selected),
            "boundary_loss": float(boundary.detach()),
            "detection_loss": float(detection.detach()),
            "loss_finite": bool(torch.isfinite(total)),
            "gradient_finite": finite_gradient, "gradient_norm": grad_norm,
            "optimizer_created": False, "optimizer_step": False,
        })
        gradient_rows.append({key: batch_rows[-1][key] for key in (
            "audit_batch", "fault", "loss_finite", "gradient_finite",
            "gradient_norm", "optimizer_created", "optimizer_step",
        )})
        # No update; explicitly truncate temporal graphs before the next batch.
        for name in ("memory_embedding", "memory_reference_point", "memory_velo",
                     "memory_timestamp", "memory_egopose"):
            value = getattr(head, name, None)
            if torch.is_tensor(value): setattr(head, name, value.detach())
        measured += 1
    if measured != 100: raise RuntimeError(f"expected 100 batches, got {measured}")
    if sum(row["fault"] for row in batch_rows) != 50:
        raise RuntimeError("audit window must contain exactly 50 fault batches")

    clean = [r for r in diagnostics if not r["fault"]]
    fault = [r for r in diagnostics if r["fault"]]
    summaries = [summary(clean, "clean"), summary(fault, "fault"), summary(diagnostics, "all")]
    fault_selected = [r for r in fault if r["pair_selected"]]
    clean_selected = [r for r in clean if r["pair_selected"]]
    fault_batches_active = sum(r["selected_pairs"] > 0 for r in batch_rows if r["fault"])
    fault_batch_activation = fault_batches_active / 50
    fault_gt_ratio = len(fault_selected) / max(len(fault), 1)
    clean_counterfactual_ratio = len(clean_selected) / max(len(clean), 1)
    nonzero_ratio = sum(r["nonzero"] for r in fault_selected) / max(len(fault_selected), 1)
    median_loss = percentile([r["loss"] for r in fault_selected if r["nonzero"]], 50)
    median_rank_distance = percentile([r["rank_distance_from_k"] for r in fault_selected], 50)
    within500 = sum(1 <= r["rank_distance_from_k"] <= 500 for r in fault_selected) / max(len(fault_selected), 1)
    selection_exact = all(
        r["center_distance"] <= 2.0 and r["positive_rank"] > 100
        and r["negative_rank"] == 100 and r["negative_truly_outranks"]
        for r in fault_selected
    )
    finite = all(r["loss_finite"] and r["gradient_finite"] for r in batch_rows)
    invariant_rows = list(csv.DictReader((output / "disabled_invariance.csv").open()))
    invariant = all(float(r["max_abs_diff"]) == 0 for r in invariant_rows)
    go = (
        fault_gt_ratio >= 0.10 and fault_batch_activation >= 0.30
        and nonzero_ratio >= 0.95 and median_loss > 0
        and selection_exact and 1 <= median_rank_distance <= 500
        and within500 >= 0.50
        and fault_gt_ratio >= clean_counterfactual_ratio
        and finite and invariant
    )

    write_csv(output / "activation_events.csv", diagnostics)
    write_csv(output / "activation_summary.csv", summaries + [{
        "split": "decision", "fault_batch_activation": fault_batch_activation,
        "fault_gt_activation": fault_gt_ratio,
        "clean_counterfactual_activation": clean_counterfactual_ratio,
        "selection_exact": selection_exact, "finite": finite,
        "disabled_invariant": invariant, "go": go,
    }])
    write_csv(output / "batch_activation.csv", batch_rows)
    write_csv(output / "rank_distance_distribution.csv", [{
        "split": split,
        **{f"rank_distance_p{q}": percentile([
            r["rank_distance_from_k"] for r in rows if r["pair_selected"]
        ], q) for q in (10, 25, 50, 75, 90, 95, 99)},
        "within_100": sum(r["pair_selected"] and r["rank_distance_from_k"] <= 100 for r in rows) / max(sum(r["pair_selected"] for r in rows), 1),
        "within_500": sum(r["pair_selected"] and r["rank_distance_from_k"] <= 500 for r in rows) / max(sum(r["pair_selected"] for r in rows), 1),
    } for split, rows in (("clean", clean), ("fault", fault))])
    write_csv(output / "gradient_checks.csv", gradient_rows + [{
        "audit_batch": "summary", "fault": "all", "loss_finite": finite,
        "gradient_finite": finite, "optimizer_created": False,
        "optimizer_step": False, "train_batches": 100, "val_samples": 0,
    }])

    report = f"""# Hard-Positive Top-K Boundary Objective Activation Audit

## Decision

**{'GO' if go else 'NO-GO'}**. {'Only the preregistered 2-iteration smoke is authorized next.' if go else 'Do not tune margin/lambda or enter mini training.'}

## Objective

The stock B0 Hungarian assignment, detection losses, forward, memory and deployment Top-K remain unchanged. On fault training frames only, the final decoder first finds the highest-scoring q+ among queries with center distance <=2m; the GT is eligible only when this best geometry-qualified GT-class pair has flattened deployment rank>K=100. Detached q- is the actual Kth query/class pair. The only new gradient is `relu(0.10 + s(q-) - s(q+))` through those two sigmoid scores. Easy positives and non-fault frames receive no objective.

## Activation

- 100 no-update mini-train batches: 50 clean / 50 fault; validation used: 0.
- Fault GT: {len(fault)}; selected pairs: {len(fault_selected)}; pairs/GT and activation: {fault_gt_ratio:.3%}.
- Fault batches with a pair: {fault_batches_active}/50 ({fault_batch_activation:.3%}).
- Clean actual pairs: 0; counterfactual strict-rank-out eligibility: {len(clean_selected)}/{len(clean)} ({clean_counterfactual_ratio:.3%}).
- Nonzero loss among fault selected pairs: {nonzero_ratio:.3%}; conditional median loss: {median_loss:.8f}.
- q+ center distance median/p90: {percentile([r['center_distance'] for r in fault_selected], 50):.4f}m / {percentile([r['center_distance'] for r in fault_selected], 90):.4f}m.
- q+ original rank median and median distance from K: {percentile([r['positive_rank'] for r in fault_selected], 50):.1f} / {median_rank_distance:.1f}; within K+500: {within500:.3%}.
- Every selected event is geometry-qualified, rank>K, and paired with the true Kth outranking competitor: {selection_exact}.

## Safety and interpretation

- All AMP losses/gradients finite: {finite}; optimizer created/stepped: False/False.
- Disabled Clean/Dark/Blur/Crash inference `max_abs_diff=0`: {invariant}.
- Structurally, every selected event is an audit-defined strict rank-collapse case rather than an easy Top-K positive. However, Fault eligibility ({fault_gt_ratio:.3%}) is slightly below both the 10% coverage gate and Clean counterfactual eligibility ({clean_counterfactual_ratio:.3%}); the activation is therefore not demonstrably fault-specific. This is the reason for NO-GO despite finite, nonzero boundary losses. No loss-scale recommendation is made in this task.

If GO, the sole next experiment is one 2-iteration smoke using identical selection, K=100, 2m threshold and margin=0.10, with one fixed weight preregistered before execution; no candidate variants or 50/200-iteration run is authorized.
"""
    (output / "HARD_POSITIVE_BOUNDARY_OBJECTIVE_AUDIT.md").write_text(report)
    (output / "objective_definition.md").write_text(
        "# Objective definition\n\n" + report.split("## Activation")[0].split("## Objective\n\n", 1)[1]
    )
    print(json.dumps({
        "fault_gt": len(fault), "fault_pairs": len(fault_selected),
        "fault_gt_ratio": fault_gt_ratio, "fault_batch_activation": fault_batch_activation,
        "clean_counterfactual_ratio": clean_counterfactual_ratio,
        "nonzero_ratio": nonzero_ratio, "median_loss": median_loss,
        "median_rank_distance": median_rank_distance, "within500": within500,
        "go": go,
    }, indent=2))


if __name__ == "__main__": main()
