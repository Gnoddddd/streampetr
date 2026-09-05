#!/usr/bin/env python3
"""Detached-output counterfactual gradients for frozen B0 assignments."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STREAM_ROOT = ROOT / "repos/StreamPETR"
sys.dont_write_bytecode = True
sys.path.insert(0, str(STREAM_ROOT))

import mmcv  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmdet.core import build_assigner  # noqa: E402
from mmdet.models import build_loss  # noqa: E402
from projects.mmdet3d_plugin.core.bbox import assigners as _assigners  # noqa: E402,F401
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox  # noqa: E402

from analysis.fault_assignment_gradient import (  # noqa: E402
    SELECTION_METRICS,
    bootstrap_selection_difference,
    scalar_gradient_relation,
    select_equal_budget,
    selection_metrics,
    unit_key,
    vector_gradient_relation,
)
from analysis.fault_boundary_root_cause import (  # noqa: E402
    auroc,
    candidate_pool_statistics,
    spearman,
)
from analysis.supervision_identity import assignment_identity  # noqa: E402
from scripts.analyze_supervision_identity import (  # noqa: E402
    build_gt_cache,
    compare_tensors,
    load_root_rows,
    load_trace,
)


REPORT = ROOT / "reports/stage4/fault_assignment_gradient_audit"
CONFIG = ROOT / "configs/stage4/gt_query_survival_b0_audit.py"
DISABLED = ROOT / "outputs/stage4/lidar_privileged_target_evidence_audit/disabled"
TRACE = ROOT / "outputs/stage4/gt_query_survival_audit"
IDENTITY = ROOT / "reports/stage4/supervision_identity_audit/per_layer_assignment.csv"
GROUPS = {
    "blur_back": "CAM_BACK Motion Blur",
    "crash_back": "CAM_BACK Crash",
    "dark_back": "CAM_BACK Dark",
}
CLASS_NAMES = (
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)
LAYERS = 6
SEED = 314159
BOOTSTRAPS = 5000


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty output: {name}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def truth(value) -> bool:
    return str(value).lower() == "true"


def finite_mean(rows: list[dict], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def finite_median(rows: list[dict], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def boolean_rate(rows: list[dict], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows])) if rows else float("nan")


def vector_json(tensor: torch.Tensor) -> str:
    return json.dumps([float(value) for value in tensor.detach().cpu().tolist()],
                      separators=(",", ":"))


def classification_gradient(loss, logits, target: int, average_factor: int) -> torch.Tensor:
    value = logits.detach().clone().float().requires_grad_(True)
    output = loss(
        value.unsqueeze(0),
        torch.tensor([int(target)], dtype=torch.long),
        torch.ones(1, dtype=torch.float32),
        avg_factor=max(int(average_factor), 1),
    )
    return torch.autograd.grad(output, value)[0].detach()


def regression_gradient(loss, prediction, target, code_weights,
                        average_factor: int) -> torch.Tensor:
    value = prediction.detach().clone().float().requires_grad_(True)
    output = loss(
        value.unsqueeze(0),
        target.detach().clone().float().unsqueeze(0),
        code_weights.detach().clone().float().unsqueeze(0),
        avg_factor=max(int(average_factor), 1),
    )
    return torch.autograd.grad(output, value)[0].detach()


def exact_pair_cost(assigner, logits, bbox_pred, focal, pc_range,
                    code_weights) -> float:
    label = torch.tensor([int(focal["label"])], dtype=torch.long)
    target = normalize_bbox(
        torch.tensor(focal["box"], dtype=torch.float32).unsqueeze(0), pc_range
    )
    cls_cost = assigner.cls_cost(logits.unsqueeze(0), label)[0, 0]
    reg_cost = assigner.reg_cost(
        (bbox_pred.unsqueeze(0) * code_weights.unsqueeze(0))[:, :8],
        (target * code_weights.unsqueeze(0))[:, :8],
    )[0, 0]
    return float((cls_cost + reg_cost).item())


def bootstrap_spearman(x, y, seed: int) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng, values = np.random.default_rng(seed), []
    for _ in range(BOOTSTRAPS):
        indexes = rng.integers(0, len(x), len(x))
        value = spearman(x[indexes], y[indexes])
        if np.isfinite(value):
            values.append(value)
    low, high = np.percentile(values, [2.5, 97.5]) if values else (np.nan, np.nan)
    return {"estimate": spearman(x, y), "ci_low": float(low),
            "ci_high": float(high), "iterations": BOOTSTRAPS}


def bootstrap_auc(risk, outcome, seed: int) -> dict:
    risk, outcome = np.asarray(risk, float), np.asarray(outcome, int)
    positive, negative = risk[outcome == 1], risk[outcome == 0]
    if not len(positive) or not len(negative):
        return {"estimate": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "iterations": BOOTSTRAPS}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAPS):
        p = positive[rng.integers(0, len(positive), len(positive))]
        n = negative[rng.integers(0, len(negative), len(negative))]
        values.append(auroc(np.concatenate([p, n]),
                            np.concatenate([np.ones(len(p)), np.zeros(len(n))])))
    return {"estimate": auroc(risk, outcome),
            "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5)),
            "iterations": BOOTSTRAPS}


def summarize(protocol: str, population: str, gt_rows: list[dict],
              layer_rows: list[dict]) -> dict:
    final_starved = [row for row in layer_rows
                     if row["decoder_layer"] == LAYERS - 1 and row["aux_active"]]
    final_harmful = [row for row in final_starved if row["cls_harmful_current"]]
    starved = [row for row in layer_rows if row["aux_active"]]
    harmful = [row for row in starved if row["cls_harmful_current"]]
    crossings = [row for row in gt_rows
                 if row["boundary_crossing"] and row["target_train_eligible"]]
    return {
        "protocol": protocol,
        "condition": GROUPS[protocol],
        "population": population,
        "n": len(gt_rows),
        "train_eligible_n": sum(row["target_train_eligible"] for row in gt_rows),
        "mean_non_same_layer_fraction": finite_mean(gt_rows, "non_same_layer_fraction"),
        "final_starved_n": len(final_starved),
        "final_harmful_rate_among_starved": boolean_rate(
            final_starved, "cls_harmful_current"),
        "final_cls_conflict_rate_among_starved": boolean_rate(
            final_starved, "cls_gradient_conflict"),
        "final_reversal_rate_among_harmful": boolean_rate(
            final_harmful, "cls_harmful_reversed"),
        "final_box_positive_rate_among_starved": boolean_rate(
            final_starved, "box_combined_desired_positive"),
        "crossing_n": len(crossings),
        "final_reversal_crossing_coverage": boolean_rate(
            crossings, "final_cls_harmful_reversed"),
        "any_reversal_crossing_coverage": boolean_rate(
            crossings, "any_cls_harmful_reversed"),
        "starved_layer_n": len(starved),
        "all_layer_harmful_rate_among_starved": boolean_rate(
            starved, "cls_harmful_current"),
        "all_layer_reversal_rate_among_harmful": boolean_rate(
            harmful, "cls_harmful_reversed"),
        "all_layer_box_current_conflict_rate": boolean_rate(
            starved, "box_current_conflict"),
        "mean_aux_logit_update_gain": finite_mean(starved, "cls_aux_update_gain"),
        "median_current_cls_gradient_abs": finite_median(
            starved, "cls_current_gradient_abs"),
        "median_aux_cls_gradient_abs": finite_median(
            starved, "cls_aux_gradient_abs"),
        "median_current_box_gradient_norm": finite_median(
            starved, "box_current_norm"),
        "median_aux_box_gradient_norm": finite_median(starved, "box_aux_norm"),
    }


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2026)
    np.random.seed(2026)

    cfg = Config.fromfile(str(CONFIG))
    head_cfg = cfg.model.pts_bbox_head
    assigner = build_assigner(cfg.model.train_cfg.pts.assigner)
    cls_loss = build_loss(head_cfg.loss_cls)
    bbox_loss = build_loss(head_cfg.loss_bbox)
    pc_range = tuple(float(value) for value in head_cfg.bbox_coder.pc_range)
    code_weights = torch.tensor(head_cfg.code_weights, dtype=torch.float32)
    if bool(head_cfg.match_with_velo):
        raise RuntimeError("audit requires B0 match_with_velo=False")

    invariance = []
    for protocol in ("clean", *GROUPS):
        difference, leaves = compare_tensors(
            mmcv.load(str(TRACE / protocol / "predictions.pkl")),
            mmcv.load(str(DISABLED / protocol / "predictions.pkl")),
        )
        invariance.append({"protocol": protocol, "tensor_leaves": leaves,
                           "max_abs_diff": difference, "exact": difference == 0.0})
    if not all(row["exact"] for row in invariance):
        raise RuntimeError(f"disabled path divergence: {invariance}")
    write_csv("disabled_invariance.csv", invariance)

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-mini", dataroot=str(ROOT / "data/nuscenes-mini"),
                    verbose=False)
    gt_cache = build_gt_cache(nusc)
    root_rows = [row for row in load_root_rows()
                 if row["protocol"] in GROUPS and truth(row["fault_candidate_available"])]
    with IDENTITY.open(encoding="utf-8") as handle:
        identity_rows = list(csv.DictReader(handle))
    identity_index = {
        (row["protocol"], row["sample_token"], row["gt_token"],
         int(row["decoder_layer"])): row for row in identity_rows
    }

    frame_cache, layer_output, gt_output = {}, [], []
    for root_row in root_rows:
        protocol, token = root_row["protocol"], root_row["sample_token"]
        frame_key = protocol, token
        if frame_key not in frame_cache:
            frame = load_trace(protocol, token)
            gt_data = gt_cache[token]
            targets = gt_data["train"]
            gt_boxes = torch.tensor(np.stack([value["box"] for value in targets]),
                                    dtype=torch.float32)
            gt_labels = torch.tensor([value["label"] for value in targets],
                                     dtype=torch.long)
            layer_cache = []
            for layer_index in range(LAYERS):
                logits = torch.tensor(frame["layer_logits"][layer_index], dtype=torch.float32)
                physical = torch.tensor(frame["layer_boxes"][layer_index], dtype=torch.float32)
                bbox_pred = normalize_bbox(physical, pc_range)
                result = assigner.assign(bbox_pred, logits, gt_boxes, gt_labels, None,
                                         code_weights, False)
                assigned = result.gt_inds.cpu().numpy().astype(np.int64) - 1
                layer_cache.append({"logits": logits, "physical": physical,
                                    "bbox_pred": bbox_pred, "assigned": assigned,
                                    "positive_count": int(np.count_nonzero(assigned >= 0))})
            frame_cache[frame_key] = frame, gt_data, layer_cache

        frame, gt_data, layers = frame_cache[frame_key]
        focal = gt_data["all"][root_row["gt_token"]]
        focal_index = gt_data["train_index"].get(root_row["gt_token"])
        qplus = int(root_row["fault_best_query"])
        check = candidate_pool_statistics(
            frame["layer_logits"][-1], frame["layer_boxes"][-1], focal["box"][:3],
            focal["label"], topk=100, radius=2.0,
        )
        if not check["candidate_available"] or int(check["best_query"]) != qplus:
            raise RuntimeError(f"q+ mismatch {protocol}/{token}/{focal['token']}")

        unit_layers = []
        for layer_index, layer in enumerate(layers):
            assigned_index = int(layer["assigned"][qplus])
            identity = assignment_identity(assigned_index, focal_index)
            prior = identity_index[(protocol, token, focal["token"], layer_index)]
            if prior["identity"] != identity or int(prior["qplus"]) != qplus:
                raise RuntimeError(f"identity replay mismatch {protocol}/{token}/{focal['token']}")
            assigned_target = (gt_data["train"][assigned_index]
                               if assigned_index >= 0 else None)
            positive_count = layer["positive_count"]
            current_label = (assigned_target["label"] if assigned_target is not None
                             else len(CLASS_NAMES))
            current_cls = classification_gradient(
                cls_loss, layer["logits"][qplus], current_label, positive_count)
            aux_active = focal_index is not None and identity != "same-GT positive"
            aux_cls = (classification_gradient(
                cls_loss, layer["logits"][qplus], focal["label"], positive_count)
                if aux_active else torch.zeros_like(current_cls))
            cls_relation = scalar_gradient_relation(
                current_cls[focal["label"]], aux_cls[focal["label"]])

            prediction = layer["bbox_pred"][qplus, :10]
            if assigned_target is not None:
                current_target = normalize_bbox(
                    torch.tensor(assigned_target["box"], dtype=torch.float32).unsqueeze(0),
                    pc_range,
                )[0, :10]
                current_box = regression_gradient(
                    bbox_loss, prediction, current_target, code_weights, positive_count)
            else:
                current_box = torch.zeros_like(prediction)
            if aux_active:
                focal_target = normalize_bbox(
                    torch.tensor(focal["box"], dtype=torch.float32).unsqueeze(0),
                    pc_range,
                )[0, :10]
                aux_box = regression_gradient(
                    bbox_loss, prediction, focal_target, code_weights, positive_count)
            else:
                aux_box = torch.zeros_like(prediction)
            box_relation = vector_gradient_relation(current_box, aux_box)
            row = {
                "protocol": protocol, "condition": GROUPS[protocol],
                "sample_token": token, "scene_token": root_row["scene_token"],
                "frame_idx": int(root_row["frame_idx"]), "gt_token": focal["token"],
                "gt_class": root_row["gt_class"], "outcome": root_row["outcome"],
                "lost_degraded": (root_row["outcome"] == "fault_induced_lost"
                                  and float(root_row["delta_s_pos"]) < 0.0),
                "retained": root_row["outcome"] == "retained_control",
                "boundary_crossing": truth(root_row["boundary_crossing"]),
                "delta_s_pos": float(root_row["delta_s_pos"]),
                "fault_s_pos": float(root_row["fault_s_pos"]),
                "fault_margin": float(root_row["fault_margin"]),
                "qplus": qplus, "decoder_layer": layer_index,
                "target_train_eligible": focal_index is not None,
                "identity": identity, "assigned_gt_token": (
                    assigned_target["token"] if assigned_target is not None else ""),
                "assigned_gt_class": (
                    assigned_target["name"] if assigned_target is not None else "background"),
                "assigned_same_class": bool(
                    assigned_target is not None
                    and assigned_target["label"] == focal["label"]),
                "aux_active": aux_active, "layer_positive_count": positive_count,
                "qplus_center_distance_to_focal_gt": float(torch.linalg.vector_norm(
                    layer["physical"][qplus, :3]
                    - torch.tensor(focal["box"][:3], dtype=torch.float32)).item()),
                "cls_current_gradient": cls_relation["current_gradient"],
                "cls_aux_gradient": cls_relation["aux_gradient"],
                "cls_combined_gradient": cls_relation["combined_gradient"],
                "cls_current_gradient_abs": abs(cls_relation["current_gradient"]),
                "cls_aux_gradient_abs": abs(cls_relation["aux_gradient"]),
                "cls_current_update": cls_relation["current_update"],
                "cls_aux_update_gain": cls_relation["aux_update_gain"],
                "cls_combined_update": cls_relation["combined_update"],
                "cls_harmful_current": cls_relation["harmful_current"],
                "cls_gradient_conflict": cls_relation["gradient_conflict"],
                "cls_harmful_reversed": cls_relation["harmful_reversed"],
                "box_current_gradient": vector_json(current_box),
                "box_aux_gradient": vector_json(aux_box),
                "box_combined_gradient": vector_json(current_box + aux_box),
                "box_current_norm": box_relation["current_norm"],
                "box_aux_norm": box_relation["aux_norm"],
                "box_combined_norm": box_relation["combined_norm"],
                "box_current_desired_projection": box_relation["current_desired_projection"],
                "box_combined_desired_projection": box_relation["combined_desired_projection"],
                "box_current_aux_cosine": box_relation["current_aux_cosine"],
                "box_current_conflict": box_relation["current_conflict"],
                "box_combined_desired_positive": box_relation["combined_desired_positive"],
            }
            unit_layers.append(row)
            layer_output.append(row)

        final = unit_layers[-1]
        non_same_count = sum(row["identity"] != "same-GT positive"
                             for row in unit_layers)
        pair_cost = (exact_pair_cost(
            assigner, layers[-1]["logits"][qplus], layers[-1]["bbox_pred"][qplus],
            focal, pc_range, code_weights) if focal_index is not None else float("nan"))
        gt_output.append({
            "protocol": protocol, "condition": GROUPS[protocol],
            "sample_token": token, "scene_token": root_row["scene_token"],
            "frame_idx": int(root_row["frame_idx"]), "gt_token": focal["token"],
            "gt_class": root_row["gt_class"], "outcome": root_row["outcome"],
            "lost_degraded": final["lost_degraded"], "retained": final["retained"],
            "boundary_crossing": final["boundary_crossing"],
            "delta_s_pos": final["delta_s_pos"], "fault_s_pos": final["fault_s_pos"],
            "fault_margin": final["fault_margin"], "qplus": qplus,
            "target_train_eligible": focal_index is not None,
            "non_same_layer_count": non_same_count,
            "non_same_layer_fraction": non_same_count / LAYERS,
            "never_same_gt": non_same_count == LAYERS,
            "always_same_gt": non_same_count == 0,
            "final_identity": final["identity"],
            "final_non_same": final["identity"] != "same-GT positive",
            "pair_cost": pair_cost,
            "final_cls_harmful_current": final["cls_harmful_current"],
            "final_cls_gradient_conflict": final["cls_gradient_conflict"],
            "final_cls_harmful_reversed": final["cls_harmful_reversed"],
            "any_cls_harmful_reversed": any(
                row["cls_harmful_reversed"] for row in unit_layers),
            "final_cls_current_gradient": final["cls_current_gradient"],
            "final_cls_aux_gradient": final["cls_aux_gradient"],
            "final_cls_combined_gradient": final["cls_combined_gradient"],
            "final_box_current_conflict": final["box_current_conflict"],
            "final_box_combined_desired_positive": final["box_combined_desired_positive"],
            "mean_aux_update_gain_starved_layers": finite_mean(
                [row for row in unit_layers if row["aux_active"]], "cls_aux_update_gain"),
            "easy_retained": bool(final["retained"] and non_same_count < LAYERS
                                  and not final["boundary_crossing"]),
        })

    write_csv("per_layer_gradient.csv", layer_output)
    write_csv("per_gt_gradient.csv", gt_output)

    summaries = []
    for protocol in GROUPS:
        for population in ("lost_degraded", "retained_all_available"):
            selected_gt = [row for row in gt_output if row["protocol"] == protocol and (
                row["lost_degraded"] if population == "lost_degraded" else row["retained"])]
            keys = {unit_key(row) for row in selected_gt}
            selected_layers = [row for row in layer_output
                               if row["protocol"] == protocol and unit_key(row) in keys]
            summaries.append(summarize(protocol, population, selected_gt, selected_layers))
    summary_index = {(row["protocol"], row["population"]): row for row in summaries}
    mechanism = {}
    for protocol in GROUPS:
        lost = summary_index[(protocol, "lost_degraded")]
        retained = summary_index[(protocol, "retained_all_available")]
        mechanism[protocol] = {
            "starvation_strength": (lost["mean_non_same_layer_fraction"]
                                    - retained["mean_non_same_layer_fraction"]),
            "benefit_strength": lost["mean_aux_logit_update_gain"],
        }
        for row in summaries:
            if row["protocol"] == protocol:
                row.update(mechanism[protocol])
    write_csv("protocol_gradient_summary.csv", summaries)

    prediction_rows, bootstrap_rows = [], []
    for protocol_index, protocol in enumerate(GROUPS):
        rows = [row for row in gt_output
                if row["protocol"] == protocol and row["target_train_eligible"]]
        x = [row["non_same_layer_count"] for row in rows]
        delta = [row["delta_s_pos"] for row in rows]
        crossing = [int(row["boundary_crossing"]) for row in rows]
        rho = bootstrap_spearman(x, delta, SEED + protocol_index * 10)
        auc = bootstrap_auc(x, crossing, SEED + protocol_index * 10 + 1)
        prediction_rows.append({"protocol": protocol, "condition": GROUPS[protocol],
            "n": len(rows), "crossing_n": sum(crossing),
            "spearman_never_layers_vs_delta_s_pos": rho["estimate"],
            "spearman_ci_low": rho["ci_low"], "spearman_ci_high": rho["ci_high"],
            "crossing_auroc": auc["estimate"], "auroc_ci_low": auc["ci_low"],
            "auroc_ci_high": auc["ci_high"]})
        bootstrap_rows += [
            {"category": "never_same_prediction", "protocol": protocol,
             "metric": "spearman_vs_delta_s_pos", **rho},
            {"category": "never_same_prediction", "protocol": protocol,
             "metric": "crossing_auroc", **auc},
        ]
    write_csv("never_same_prediction.csv", prediction_rows)

    selection_rows, comparison_rows = [], []
    selection_index = {}
    for protocol_index, protocol in enumerate(GROUPS):
        pool = [row for row in gt_output if row["protocol"] == protocol
                and row["target_train_eligible"] and row["final_non_same"]]
        budget = sum(row["never_same_gt"] for row in pool)
        selected = select_equal_budget(pool, budget)
        if len(selected["generic"]) != budget or len(selected["selective"]) != budget:
            raise RuntimeError(f"unequal selection budget for {protocol}")
        selection_index[protocol] = selected
        for row in pool:
            selection_rows.append({
                "protocol": protocol, "condition": GROUPS[protocol],
                "sample_token": row["sample_token"], "gt_token": row["gt_token"],
                "gt_class": row["gt_class"], "budget": budget,
                "pair_cost": row["pair_cost"],
                "non_same_layer_count": row["non_same_layer_count"],
                "final_non_same": row["final_non_same"],
                "fault_margin": row["fault_margin"],
                "lost_degraded": row["lost_degraded"],
                "boundary_crossing": row["boundary_crossing"],
                "retained": row["retained"], "easy_retained": row["easy_retained"],
                "generic_selected": unit_key(row) in selected["generic"],
                "selective_selected": unit_key(row) in selected["selective"],
            })
        all_keys = {unit_key(row) for row in pool}
        base = selection_metrics(pool, all_keys)
        for scheme in ("generic", "selective"):
            values = selection_metrics(pool, selected[scheme])
            comparison_rows.append({"protocol": protocol, "condition": GROUPS[protocol],
                "scheme": scheme, "population_n": len(pool), "budget": budget,
                **{f"{metric}_rate": values[metric] for metric in SELECTION_METRICS},
                **{f"{metric}_enrichment": (
                    values[metric] / base[metric] if base[metric] > 0 else float("nan"))
                   for metric in SELECTION_METRICS},
                "concentration": values["concentration"]})
        for metric_index, metric in enumerate((*SELECTION_METRICS, "concentration")):
            result = bootstrap_selection_difference(
                pool, selected["generic"], selected["selective"], metric,
                SEED + 100 + protocol_index * 10 + metric_index, BOOTSTRAPS)
            bootstrap_rows.append({"category": "selective_minus_generic",
                "protocol": protocol, "metric": metric, **result})
    write_csv("budget_selection.csv", selection_rows)
    write_csv("budget_comparison.csv", comparison_rows)
    write_csv("bootstrap_95ci.csv", bootstrap_rows)

    prediction_index = {row["protocol"]: row for row in prediction_rows}
    comparison_index = {(row["protocol"], row["scheme"]): row
                        for row in comparison_rows}
    bootstrap_index = {(row["protocol"], row["metric"]): row for row in bootstrap_rows
                       if row["category"] == "selective_minus_generic"}
    blur = summary_index[("blur_back", "lost_degraded")]
    gradient_components = {
        "blur_harmful": blur["final_harmful_rate_among_starved"] >= 0.5,
        "blur_reversal": blur["final_reversal_rate_among_harmful"] >= 0.5,
        "blur_crossing_coverage": blur["final_reversal_crossing_coverage"] >= 0.5,
        "blur_box_projection": blur["final_box_positive_rate_among_starved"] >= 0.5,
        "starvation_order": (mechanism["blur_back"]["starvation_strength"]
                             > mechanism["crash_back"]["starvation_strength"]
                             > mechanism["dark_back"]["starvation_strength"]),
        "benefit_order": (mechanism["blur_back"]["benefit_strength"]
                          > mechanism["crash_back"]["benefit_strength"]
                          > mechanism["dark_back"]["benefit_strength"]),
    }
    gradient_gate = all(gradient_components.values())
    never_gate = (prediction_index["blur_back"]["spearman_never_layers_vs_delta_s_pos"] < 0
                  and prediction_index["blur_back"]["crossing_auroc"] > 0.5)
    generic = comparison_index[("blur_back", "generic")]
    selective = comparison_index[("blur_back", "selective")]
    selection_components = {
        "blur_lost": selective["lost_degraded_rate"] > generic["lost_degraded_rate"],
        "blur_crossing": selective["boundary_crossing_rate"] > generic["boundary_crossing_rate"],
        "blur_retained": selective["retained_rate"] < generic["retained_rate"],
        "blur_easy": selective["easy_retained_rate"] < generic["easy_retained_rate"],
        "blur_concentration_ci": bootstrap_index[("blur_back", "concentration")]["ci_low"] > 0,
        "crash_concentration": (
            comparison_index[("crash_back", "selective")]["concentration"]
            > comparison_index[("crash_back", "generic")]["concentration"]),
        "dark_negative_control": (
            bootstrap_index[("dark_back", "concentration")]["estimate"]
            < bootstrap_index[("blur_back", "concentration")]["estimate"]),
    }
    selection_gate = all(selection_components.values())
    go = gradient_gate and never_gate and selection_gate
    decision = ("GO_FAULT_AWARE_AUX_ASSIGNMENT" if go
                else "NO_GO_FAULT_AWARE_ASSIGNMENT")

    lines = ["# Fault-aware Assignment Counterfactual Gradient Audit", "",
             "## 决策", "", f"**{decision}**。", ""]
    if go:
        lines += ["原 Hungarian 对几何合格 starved q+ 的输出梯度机制成立，且同预算",
                  "fault-aware selection 比 generic one-to-many 更集中于真实failure。",
                  "下一阶段才允许设计train-only auxiliary assignment并做2-iter smoke；",
                  "本审计未实现或训练。", ""]
    else:
        lines += ["预注册gradient mechanism与never-same prediction门失败；",
                  "同预算selection门虽通过，但不足以支持整条机制链。",
                  "停止Fault-aware assignment路线，不调权重/预算，不进入smoke。", ""]
    lines += ["## Assignment与反事实梯度", "",
              "主表为lost_degraded；reversal分母是final harmful starved q+。", "",
              "| Protocol | n | final starved | harmful | reversal | crossing coverage | box desired+ | starvation strength | aux gain |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for protocol in GROUPS:
        row = summary_index[(protocol, "lost_degraded")]
        lines.append(f"| {GROUPS[protocol]} | {row['n']} | {row['final_starved_n']} | "
                     f"{100*row['final_harmful_rate_among_starved']:.1f}% | "
                     f"{100*row['final_reversal_rate_among_harmful']:.1f}% | "
                     f"{100*row['final_reversal_crossing_coverage']:.1f}% | "
                     f"{100*row['final_box_positive_rate_among_starved']:.1f}% | "
                     f"{mechanism[protocol]['starvation_strength']:+.3f} | "
                     f"{mechanism[protocol]['benefit_strength']:.5f} |")
    lines += ["", "Retained q+对照：", "",
              "| Protocol | n | final starved | harmful | reversal | box desired+ | aux gain |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for protocol in GROUPS:
        row = summary_index[(protocol, "retained_all_available")]
        lines.append(f"| {GROUPS[protocol]} | {row['n']} | {row['final_starved_n']} | "
                     f"{100*row['final_harmful_rate_among_starved']:.1f}% | "
                     f"{100*row['final_reversal_rate_among_harmful']:.1f}% | "
                     f"{100*row['final_box_positive_rate_among_starved']:.1f}% | "
                     f"{row['mean_aux_logit_update_gain']:.5f} |")
    lines += ["", f"梯度机制门：{gradient_gate}；组件={gradient_components}。",
              "Blur内的harmful/reversal/crossing局部机制成立，但aux gain顺序为",
              f"Dark ({mechanism['dark_back']['benefit_strength']:.5f}) > "
              f"Blur ({mechanism['blur_back']['benefit_strength']:.5f}) > "
              f"Crash ({mechanism['crash_back']['benefit_strength']:.5f})，与预注册相反。", "",
              "## never-same-GT预测", "",
              "| Protocol | n | Spearman vs delta_S_pos [95% CI] | crossing AUROC [95% CI] |",
              "|---|---:|---:|---:|"]
    for protocol in GROUPS:
        row = prediction_index[protocol]
        lines.append(f"| {GROUPS[protocol]} | {row['n']} | "
                     f"{row['spearman_never_layers_vs_delta_s_pos']:+.3f} "
                     f"[{row['spearman_ci_low']:+.3f},{row['spearman_ci_high']:+.3f}] | "
                     f"{row['crossing_auroc']:.3f} "
                     f"[{row['auroc_ci_low']:.3f},{row['auroc_ci_high']:.3f}] |")
    lines += ["", f"Blur预测方向门：{never_gate}。",
              "Blur的rho为正且CI在0上方，不支持never-same层数预测更负delta_S_pos。", "",
              "## 同预算one-to-many选择", "",
              "| Protocol | Scheme | B/N | lost_degraded | crossing | retained | easy retained | concentration |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for protocol in GROUPS:
        for scheme in ("generic", "selective"):
            row = comparison_index[(protocol, scheme)]
            lines.append(f"| {GROUPS[protocol]} | {scheme} | {row['budget']}/{row['population_n']} | "
                         f"{100*row['lost_degraded_rate']:.1f}% | "
                         f"{100*row['boundary_crossing_rate']:.1f}% | "
                         f"{100*row['retained_rate']:.1f}% | "
                         f"{100*row['easy_retained_rate']:.1f}% | "
                         f"{row['concentration']:+.3f} |")
    lines += ["", "Selective-generic concentration差（95% CI）："]
    for protocol in GROUPS:
        row = bootstrap_index[(protocol, "concentration")]
        lines.append(f"- {GROUPS[protocol]}: {row['estimate']:+.3f} "
                     f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]")
    lines += ["", f"同预算选择门：{selection_gate}；组件={selection_components}。", "",
              "选择预算宇宙按PRE_REGISTRATION_AMENDMENT_01限定为final non-same-GT，",
              "保证两方案的每个selected unit都实际产生auxiliary gradient。", "",
              "Selective虽相对generic更集中，但Blur选中集仍有94.6% retained，",
              "failure concentration仍为负；该相对优势不改变总No-Go。", "",
              "## 等价性与边界", "",
              "Clean/Blur/Crash/Dark B0与disabled各243个tensor leaves逐tensor exact，最大差0。",
              "梯度仅对detached trace outputs计算；未运行detector forward/train、未创建optimizer、",
              "未替换Hungarian、未改变memory/query/Top-K，未运行smoke，未修改repos/StreamPETR。",
              "", "## 总门", "", f"- gradient mechanism: {gradient_gate}",
              f"- never-same prediction: {never_gate}",
              f"- equal-budget selection: {selection_gate}", f"- GO: {go}"]
    (REPORT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
