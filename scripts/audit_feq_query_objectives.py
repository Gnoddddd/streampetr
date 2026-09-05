#!/usr/bin/env python3
"""Run the fixed 100-batch, train-only FEQ objective activation audit."""

import argparse
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
    return float(np.percentile(values, q)) if values else 0.0


def summary(values):
    values = [float(v) for v in values]
    nonzero = [v for v in values if v > 0]
    return {
        "records": len(values),
        "mean": float(np.mean(values)) if values else 0.0,
        "median": percentile(values, 50),
        "nonzero_conditional_median": percentile(nonzero, 50),
        "p75": percentile(values, 75), "p90": percentile(values, 90),
        "p95": percentile(values, 95), "p99": percentile(values, 99),
        "nonzero_ratio": len(nonzero) / len(values) if values else 0.0,
        "finite": all(math.isfinite(v) for v in values),
    }


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"status": "no_records"}]
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def group_rows(rows, keys, value_key):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, []).append(float(row[value_key]))
    return [{**dict(zip(keys, key)), **summary(values)}
            for key, values in sorted(groups.items(), key=lambda item: str(item[0]))]


def top_fraction_share(values, fraction=.05):
    values = sorted((float(v) for v in values if v > 0), reverse=True)
    if not values: return 0.0
    count = max(1, int(math.ceil(len(values) * fraction)))
    return sum(values[:count]) / sum(values)


def one_sided_proportion_pvalue(success_a, count_a, success_b, count_b):
    """P[p_a <= p_b] for the standard pooled two-proportion z statistic."""
    pooled = (success_a + success_b) / (count_a + count_b)
    standard_error = math.sqrt(
        pooled * (1 - pooled) * (1 / count_a + 1 / count_b)
    )
    if standard_error == 0:
        return 1.0
    z_value = (success_a / count_a - success_b / count_b) / standard_error
    return 0.5 * math.erfc(z_value / math.sqrt(2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage4/feq_objective_activation_audit.py")
    parser.add_argument("--output", default="reports/stage4/feq_query_objective_activation")
    parser.add_argument("--start", type=int, default=28)
    parser.add_argument("--batches", type=int, default=100)
    args = parser.parse_args()
    if (args.start, args.batches) != (28, 100):
        raise ValueError("preregistered window is train indices [28,128)")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(args.config)
    random.seed(2026); np.random.seed(2026)
    torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
    dataset = build_dataset(cfg.data.train)
    loader = build_dataloader(dataset, 1, 0, 1, dist=False, shuffle=False,
                              seed=2026, runner_type="IterBasedRunner")
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"),
                        test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, cfg.load_from, map_location="cpu", strict=False)
    wrap_fp16_model(model)
    model = MMDataParallel(model.cuda(), device_ids=[0]); model.train()

    diagnostics, batch_rows, gradient_rows = [], [], []
    measured = 0
    for dataset_index, data in enumerate(loader):
        if dataset_index < args.start: continue
        if measured >= args.batches: break
        if measured == 0:
            # Explicit audit-window scene reset; no unprocessed history leaks in.
            data["prev_exists"].data[0].zero_()
        model.module.zero_grad(set_to_none=True)
        losses = model(return_loss=True, **data)
        detection_terms = [value for key, value in losses.items()
                           if ("loss_cls" in key or "loss_bbox" in key)
                           and "feq" not in key]
        otm_terms = [value for key, value in losses.items()
                     if key.endswith("loss_feq_otm")]
        boundary_terms = [value for key, value in losses.items()
                          if key.endswith("loss_feq_boundary")]
        if not detection_terms or len(otm_terms) != 1 or len(boundary_terms) != 1:
            raise RuntimeError(f"unexpected loss keys: {sorted(losses)}")
        detection = torch.stack(detection_terms).sum()
        otm, boundary = otm_terms[0], boundary_terms[0]
        total = detection + otm + boundary
        print(f"audit_batch={measured} forward_complete", flush=True)
        total.backward()
        grads = [parameter.grad.detach() for parameter in model.module.parameters()
                 if parameter.grad is not None]
        grad_finite = bool(grads) and all(torch.isfinite(grad).all() for grad in grads)
        grad_norm = math.sqrt(sum(float(grad.float().norm()) ** 2 for grad in grads))
        head = model.module.pts_bbox_head
        # Mirror iteration-boundary truncated BPTT explicitly. StreamPETR
        # already detaches proposal embeddings/boxes, but this also severs
        # pose/timestamp container graphs before the next no-optimizer batch.
        for name in ("memory_embedding", "memory_reference_point", "memory_velo",
                     "memory_timestamp", "memory_egopose"):
            value = getattr(head, name, None)
            if torch.is_tensor(value):
                setattr(head, name, value.detach())
        current = [dict(row, audit_batch=measured, dataset_index=dataset_index)
                   for row in head._last_feq_diagnostics]
        diagnostics.extend(current)
        fault = bool(current[0]["fault"]) if current else False
        batch_rows.append({
            "audit_batch": measured, "dataset_index": dataset_index,
            "fault": fault, "detection_loss": float(detection.detach()),
            "otm_loss": float(otm.detach()), "boundary_loss": float(boundary.detach()),
            "total_loss": float(total.detach()), "loss_finite": bool(torch.isfinite(total)),
            "grad_finite": grad_finite, "grad_norm": grad_norm,
        })
        gradient_rows.append({"audit_batch": measured, "amp_enabled": True,
                              "loss_finite": bool(torch.isfinite(total)),
                              "gradient_finite": grad_finite, "gradient_norm": grad_norm,
                              "optimizer_created": False, "optimizer_step": False})
        measured += 1
    if measured != 100:
        raise RuntimeError(f"expected 100 measured batches, got {measured}")
    if sum(row["fault"] for row in batch_rows) != 50:
        raise RuntimeError("preregistered window is not exactly 50 clean/50 fault")

    # Event-level loss activation summaries.
    otm_rows = []
    boundary_rows = []
    for split_name, predicate in (
        ("all", lambda r: True), ("clean", lambda r: not r["fault"]),
        ("fault", lambda r: r["fault"]),
        ("present", lambda r: r["state"] == "Present"),
        ("unobserved", lambda r: r["state"] == "Unobserved"),
    ):
        selected = [r for r in diagnostics if predicate(r) and r["weight"] > 0]
        otm_rows.append({"group": split_name, "eligible_gt": len(selected),
                         **summary([r["otm_loss"] for r in selected])})
        boundary_rows.append({"group": split_name, "eligible_gt": len(selected),
                              **summary([r["boundary_loss"] for r in selected]),
                              "suppression_consistency": (
                                  sum(r["boundary_violation"] and not r["positive_in_topk"] for r in selected) /
                                  max(sum(r["boundary_violation"] for r in selected), 1))})
    for fault in (False, True):
        split = "fault" if fault else "clean"
        for layer in range(6):
            selected = [r for r in diagnostics if r["fault"] == fault and
                        r["layer"] == layer and r["weight"] > 0]
            otm_rows.append({"group": f"{split}_layer", "layer": layer,
                             "eligible_gt": len(selected),
                             **summary([r["otm_loss"] for r in selected])})
            boundary_rows.append({"group": f"{split}_layer", "layer": layer,
                                  "eligible_gt": len(selected),
                                  **summary([r["boundary_loss"] for r in selected]),
                                  "suppression_consistency": (
                                      sum(r["boundary_violation"] and not r["positive_in_topk"] for r in selected) /
                                      max(sum(r["boundary_violation"] for r in selected), 1))})
        for state in ("Present", "Unobserved"):
            selected = [r for r in diagnostics if r["fault"] == fault and
                        r["state"] == state and r["weight"] > 0]
            otm_rows.append({"group": f"{split}_state", "state": state,
                             "eligible_gt": len(selected),
                             **summary([r["otm_loss"] for r in selected])})
            boundary_rows.append({"group": f"{split}_state", "state": state,
                                  "eligible_gt": len(selected),
                                  **summary([r["boundary_loss"] for r in selected]),
                                  "suppression_consistency": (
                                      sum(r["boundary_violation"] and not r["positive_in_topk"] for r in selected) /
                                      max(sum(r["boundary_violation"] for r in selected), 1))})
    for audit_batch in range(100):
        selected = [r for r in diagnostics if r["audit_batch"] == audit_batch and
                    r["weight"] > 0]
        otm_rows.append({"group": "batch", "audit_batch": audit_batch,
                         "fault": bool(selected[0]["fault"]) if selected else "",
                         "eligible_gt": len(selected),
                         **summary([r["otm_loss"] for r in selected])})
        boundary_rows.append({"group": "batch", "audit_batch": audit_batch,
                              "fault": bool(selected[0]["fault"]) if selected else "",
                              "eligible_gt": len(selected),
                              **summary([r["boundary_loss"] for r in selected]),
                              "suppression_consistency": (
                                  sum(r["boundary_violation"] and not r["positive_in_topk"] for r in selected) /
                                  max(sum(r["boundary_violation"] for r in selected), 1))})
    write_csv(out / "geometric_otm_statistics.csv", otm_rows)
    write_csv(out / "topk_boundary_statistics.csv", boundary_rows)

    assignment_rows = []
    for split in ("clean", "fault"):
        selected = [r for r in diagnostics if r["weight"] > 0 and r["fault"] == (split == "fault")]
        geo_scores = [score for r in selected for score in r["geometric_aux_scores"]]
        old_scores = [score for r in selected for score in r["original_aux_scores"]]
        overlaps = []
        for r in selected:
            a, b = set(r["geometric_aux_queries"]), set(r["original_aux_queries"])
            overlaps.append(len(a & b) / max(len(a | b), 1))
        for method, scores, count_key in (
            ("classification_participating", old_scores, "original_aux_count"),
            ("geometry_first", geo_scores, "geometric_aux_count"),
        ):
            assignment_rows.append({
                "split": split, "method": method, "eligible_gt": len(selected),
                "aux_per_gt": float(np.mean([r[count_key] for r in selected])) if selected else 0,
                "positive_multiplicity": float(np.mean([
                    int(r["main_exists"]) + r[count_key] for r in selected])) if selected else 0,
                "initial_class_score_mean": float(np.mean(scores)) if scores else 0,
                "initial_class_score_median": percentile(scores, 50),
                "main_query_index_overlap_rate": float(np.mean([r["main_query_overlap"] for r in selected])) if selected else 0,
                "near_duplicate_box_rate": float(np.mean([r["near_duplicate_box"] for r in selected])) if selected else 0,
                "assignment_jaccard_vs_other": float(np.mean(overlaps)) if overlaps else 0,
            })
    write_csv(out / "original_vs_geometric_assignment.csv", assignment_rows)

    state_rows = group_rows(diagnostics, ["fault", "state", "reliable_history"], "boundary_loss")
    for row in state_rows:
        selected = [r for r in diagnostics if r["fault"] == row["fault"] and
                    r["state"] == row["state"] and
                    r["reliable_history"] == row["reliable_history"]]
        row["eligible_gt"] = sum(r["weight"] > 0 for r in selected)
        row["activation_ratio_eligible"] = sum(r["boundary_violation"] for r in selected if r["weight"] > 0) / max(row["eligible_gt"], 1)
    state_rows.append({"fault": "all", "state": "Absent", "reliable_history": False,
                       "records": 0, "eligible_gt": 0, "activation_ratio_eligible": 0,
                       "note": "no GT exists; auxiliary positive and boundary are undefined"})
    write_csv(out / "activation_by_state.csv", state_rows)

    layer_rows = group_rows(diagnostics, ["fault", "layer"], "boundary_loss")
    for row in layer_rows:
        selected = [r for r in diagnostics if r["fault"] == row["fault"] and r["layer"] == row["layer"] and r["weight"] > 0]
        row["eligible_gt"] = len(selected)
        row["activation_ratio"] = sum(r["boundary_violation"] for r in selected) / max(len(selected), 1)
        row["suppressed_among_activated"] = sum(r["boundary_violation"] and not r["positive_in_topk"] for r in selected) / max(sum(r["boundary_violation"] for r in selected), 1)
    write_csv(out / "activation_by_layer.csv", layer_rows)

    detection_values = [r["detection_loss"] for r in batch_rows]
    scaling_rows = []
    scale_results = {}
    for objective, target in (("otm_loss", .08), ("boundary_loss", .04)):
        values = [r[objective] for r in batch_rows]
        med = percentile(values, 50)
        lam = target * percentile(detection_values, 50) / med if med > 0 else float("inf")
        weighted_p99 = lam * percentile(values, 99)
        rare_share = top_fraction_share([r[objective.replace("_loss", "_loss")] for r in diagnostics])
        valid = (.01 <= lam <= 10 and weighted_p99 <= .30 * percentile(detection_values, 99)
                 and rare_share <= .50 and math.isfinite(lam))
        row = {"objective": objective, "target_ratio": target,
               "detection_median": percentile(detection_values, 50),
               "raw_median": med, "raw_p99": percentile(values, 99),
               "detection_p99": percentile(detection_values, 99),
               "suggested_lambda": lam, "weighted_p99": weighted_p99,
               "weighted_p99_limit": .30 * percentile(detection_values, 99),
               "top_5pct_event_loss_share": rare_share,
               "lambda_in_range": .01 <= lam <= 10,
               "p99_safe": weighted_p99 <= .30 * percentile(detection_values, 99),
               "not_rare_event_dominated": rare_share <= .50,
               "scalable": valid}
        scaling_rows.append(row); scale_results[objective] = row
    write_csv(out / "loss_scale_recommendation.csv", scaling_rows)
    write_csv(out / "leakage_and_gradient_checks.csv", gradient_rows + [{
        "audit_batch": "summary", "amp_enabled": True,
        "loss_finite": all(r["loss_finite"] for r in batch_rows),
        "gradient_finite": all(r["grad_finite"] for r in batch_rows),
        "optimizer_created": False, "optimizer_step": False,
        "train_batches": 100, "clean_batches": 50, "fault_batches": 50,
        "val_samples_used": 0, "state_dict_runtime_keys": 0,
    }])

    fault_events = [r for r in diagnostics if r["fault"] and r["weight"] > 0]
    clean_events = [r for r in diagnostics if not r["fault"] and r["weight"] > 0]
    fault_activation = sum(r["boundary_violation"] for r in fault_events) / len(fault_events)
    clean_activation = sum(r["boundary_violation"] for r in clean_events) / len(clean_events)
    clean_success = sum(r["boundary_violation"] for r in clean_events)
    fault_success = sum(r["boundary_violation"] for r in fault_events)
    clean_higher_p = one_sided_proportion_pvalue(
        clean_success, len(clean_events), fault_success, len(fault_events))
    clean_significantly_higher = clean_activation > fault_activation and clean_higher_p < .05
    suppression = sum(r["boundary_violation"] and not r["positive_in_topk"] for r in fault_events) / max(sum(r["boundary_violation"] for r in fault_events), 1)
    fault_aux = float(np.mean([r["geometric_aux_count"] for r in fault_events]))
    duplicate = float(np.mean([r["near_duplicate_box"] for r in fault_events]))
    geo_scores = [s for r in fault_events for s in r["geometric_aux_scores"]]
    all_high = bool(geo_scores) and all(s >= r["s_k"] for r in fault_events for s in r["geometric_aux_scores"])
    otm_pass = fault_aux > 0 and duplicate <= .50 and not all_high
    boundary_pass = (0.05 <= fault_activation <= .90 and
                     percentile([r["boundary_loss"] for r in fault_events if r["boundary_loss"] > 0], 50) > 0 and
                     suppression >= .50 and not clean_significantly_higher and
                     scale_results["boundary_loss"]["scalable"] and
                     all(r["loss_finite"] and r["grad_finite"] for r in batch_rows))
    go = otm_pass and boundary_pass and scale_results["otm_loss"]["scalable"]

    report = f"""# FEQ query objective activation audit

## Decision

**{'Go for the next R0/FQ1 mini task' if go else 'No-Go for mini training'}.**
This audit used exactly 100 mini-train batches (50 clean/50 fault), no validation,
no optimizer and no parameter update. The old ranking loss was not called and
survival was diagnostics-only under `torch.no_grad`.

## Geometry-first OTM

- Fault eligible events: {len(fault_events)}; auxiliary queries/GT: {fault_aux:.6f}.
- Positive multiplicity exceeds the one-to-one baseline by {fault_aux:.6f} queries/GT.
- Near-duplicate box rate: {duplicate:.6%}; exact main-query overlap is zero by construction.
- Geometry-first auxiliary initial class-score median: {percentile(geo_scores, 50):.6f}; all were already above boundary: {all_high}.
- OTM mechanism gate: **{'pass' if otm_pass else 'fail'}**.

## Top-K boundary

- Fault activation: {fault_activation:.6%}; Clean activation: {clean_activation:.6%}.
- One-sided clean-greater-than-fault activation p-value: {clean_higher_p:.8g}; significantly higher: {clean_significantly_higher}.
- Activated events corresponding to actual positive Top-K suppression: {suppression:.6%}.
- Fault nonzero conditional median: {percentile([r['boundary_loss'] for r in fault_events if r['boundary_loss'] > 0], 50):.8f}.
- Boundary mechanism/scaling gate: **{'pass' if boundary_pass else 'fail'}**.

## Scaling and safety

- OTM suggested lambda: {scale_results['otm_loss']['suggested_lambda']:.8g}; scalable: {scale_results['otm_loss']['scalable']}.
- Boundary suggested lambda: {scale_results['boundary_loss']['suggested_lambda']:.8g}; scalable: {scale_results['boundary_loss']['scalable']}.
- All 100 AMP losses and gradients finite: {all(r['loss_finite'] and r['grad_finite'] for r in batch_rows)}.
- No deployment parameter/state key, inference output rule, Top-K operation or memory update was changed.

## Interpretation

If OTM passes while boundary fails, the result is only “query coverage has a
trainable signal”; OTM is not promoted as a standalone FEQ method and no mini
training is authorized. Only a joint pass authorizes a later, separately tasked
R0 versus FQ1 comparison.
"""
    (out / "FEQ_OBJECTIVE_ACTIVATION_AUDIT.md").write_text(report)
    print(json.dumps({"measured": measured, "fault_activation": fault_activation,
                      "clean_activation": clean_activation, "suppression": suppression,
                      "fault_aux": fault_aux, "duplicate": duplicate,
                      "otm_lambda": scale_results["otm_loss"]["suggested_lambda"],
                      "boundary_lambda": scale_results["boundary_loss"]["suggested_lambda"],
                      "otm_pass": otm_pass, "boundary_pass": boundary_pass, "go": go}, indent=2))


if __name__ == "__main__":
    main()
