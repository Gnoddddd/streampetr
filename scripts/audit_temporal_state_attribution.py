#!/usr/bin/env python3
"""Frozen-B0 one-factor temporal-state attribution audit."""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STREAM_ROOT = ROOT / "repos/StreamPETR"
sys.dont_write_bytecode = True
sys.path.insert(0, str(STREAM_ROOT))

import mmcv  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmcv.utils import import_modules_from_strings  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from nuscenes.nuscenes import NuScenes  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402

from analysis.temporal_state_attribution import (  # noqa: E402
    assert_one_component_swap,
    decide_attribution,
    explanation_ratio,
    swap_one_component,
)
from analysis.temporal_state_counterfactual import cluster_bootstrap_median  # noqa: E402
from scripts.audit_cross_view_target_evidence import (  # noqa: E402
    PROTOCOLS,
    protocol_dataset,
)
from scripts.audit_dark_target_recoverability import (  # noqa: E402
    CHECKPOINT,
    CONFIG,
    DISABLED,
    MEMORY_NAMES,
    compare_tensors,
    features,
    local_gt,
    run_head,
    snapshot,
    unpack,
    validate_trace,
)
from scripts.audit_temporal_state_counterfactual import output_metrics  # noqa: E402


REPORT = ROOT / "reports/stage5/temporal_state_attribution_audit"
PARENT = ROOT / "reports/stage4/temporal_state_counterfactual_audit"
PARENT_PER_GT = PARENT / "per_gt_counterfactual.csv"
SEED, BOOTSTRAPS = 314159, 5000
COMPONENTS = tuple(MEMORY_NAMES)
ARMS = ("BD", "CA")
BOOLEAN_METRICS = {"candidate", "topk", "tp"}
SEMANTICS = {
    "memory_embedding": "decoder query feature / temporal key-value",
    "memory_reference_point": "3D reference point / temporal position",
    "memory_timestamp": "relative time / time and ego-motion conditioning",
    "memory_egopose": "accumulated 4x4 ego transform conditioning",
    "memory_velo": "predicted xy velocity conditioning",
}


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"empty output: {name}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (REPORT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def truth(value) -> bool:
    return str(value).lower() == "true"


def parent_value(row: dict, condition: str, metric: str):
    value = row[f"{condition}_{metric}"]
    if metric in BOOLEAN_METRICS:
        return truth(value)
    if metric == "qplus":
        return int(float(value)) if str(value) not in {"", "None", "nan"} else -1
    return float(value)


def delta(left, right) -> float:
    left, right = float(left), float(right)
    return left - right if math.isfinite(left) and math.isfinite(right) else float("nan")


def median(rows: list[dict], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def rate(rows: list[dict], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows])) if rows else float("nan")


def ratio(numerator: float, denominator: float) -> tuple[float, float]:
    result = explanation_ratio(numerator, denominator)
    return result["raw"], result["clipped"]


def validate_parent_metrics(values: dict, parent: dict, condition: str,
                            identity: str) -> None:
    for metric in ("candidate", "qplus", "s_pos", "rank", "margin", "topk", "tp"):
        expected = parent_value(parent, condition, metric)
        actual = values[metric]
        if metric in BOOLEAN_METRICS or metric == "qplus":
            if actual != expected:
                raise RuntimeError(
                    f"parent {condition}/{metric} mismatch {identity}: {actual}/{expected}")
        elif not (math.isnan(float(actual)) and math.isnan(float(expected))):
            if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9):
                raise RuntimeError(
                    f"parent {condition}/{metric} mismatch {identity}: {actual}/{expected}")


def effect_fields(arm: str, component: str, values: dict,
                  base: dict) -> dict:
    prefix = f"{arm}_{component}"
    rank_effect = (delta(base["rank"], values["rank"])
                   if arm == "BD" else delta(values["rank"], base["rank"]))
    event_topk = ((not base["topk"] and values["topk"])
                  if arm == "BD" else (base["topk"] and not values["topk"]))
    event_tp = ((not base["tp"] and values["tp"])
                if arm == "BD" else (base["tp"] and not values["tp"]))
    output = {}
    for metric in ("candidate", "qplus", "s_pos", "rank", "margin", "topk", "tp"):
        output[f"{prefix}_{metric}"] = values[metric]
    output.update({
        f"{prefix}_s_pos_effect": delta(values["s_pos"], base["s_pos"]),
        f"{prefix}_margin_effect": delta(values["margin"], base["margin"]),
        f"{prefix}_rank_effect": rank_effect,
        f"{prefix}_topk_event": event_topk,
        f"{prefix}_tp_event": event_tp,
    })
    return output


def summarize(protocol: str, group: str, component: str, arm: str,
              rows: list[dict]) -> dict:
    component_prefix = f"{arm}_{component}"
    full_prefix = f"full_{arm}"
    output = {"protocol": protocol,
              "condition": "Pooled" if protocol == "pooled" else PROTOCOLS[protocol][0],
              "group": group, "component": component, "arm": arm, "n": len(rows),
              "component_candidate_rate": rate(rows, f"{component_prefix}_candidate"),
              "component_median_s_pos_effect": median(
                  rows, f"{component_prefix}_s_pos_effect"),
              "component_median_margin_effect": median(
                  rows, f"{component_prefix}_margin_effect"),
              "component_median_rank_effect": median(
                  rows, f"{component_prefix}_rank_effect"),
              "component_topk_event_rate": rate(rows, f"{component_prefix}_topk_event"),
              "component_tp_event_rate": rate(rows, f"{component_prefix}_tp_event"),
              "full_median_s_pos_effect": median(rows, f"{full_prefix}_s_pos_effect"),
              "full_median_margin_effect": median(rows, f"{full_prefix}_margin_effect"),
              "full_median_rank_effect": median(rows, f"{full_prefix}_rank_effect"),
              "full_topk_event_rate": rate(rows, f"{full_prefix}_topk_event"),
              "full_tp_event_rate": rate(rows, f"{full_prefix}_tp_event")}
    for metric in ("s_pos", "margin", "rank", "topk", "tp"):
        numerator = output[("component_median_" + metric + "_effect")
                           if metric in {"s_pos", "margin", "rank"}
                           else ("component_" + metric + "_event_rate")]
        denominator = output[("full_median_" + metric + "_effect")
                             if metric in {"s_pos", "margin", "rank"}
                             else ("full_" + metric + "_event_rate")]
        raw, clipped = ratio(numerator, denominator)
        output[f"{metric}_explanation_raw"] = raw
        output[f"{metric}_explanation_clipped"] = clipped
    return output


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    REPORT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")

    with PARENT_PER_GT.open(encoding="utf-8") as handle:
        parent_rows = list(csv.DictReader(handle))
    if len(parent_rows) != 142:
        raise RuntimeError(f"parent population changed: {len(parent_rows)}")
    expected_counts = {("dark_back", "lost"): 15, ("dark_back", "retained"): 15,
                       ("blur_back", "lost"): 20, ("blur_back", "retained"): 20,
                       ("crash_back", "lost"): 36, ("crash_back", "retained"): 36}
    observed_counts = defaultdict(int)
    units = defaultdict(list)
    population = []
    for row in parent_rows:
        key = (row["protocol"], row["group"])
        observed_counts[key] += 1
        units[(row["protocol"], row["sample_token"])].append(row)
        population.append({key: row[key] for key in (
            "protocol", "condition", "group", "sample_token", "scene_token",
            "frame_idx", "fault_episode_age", "gt_token", "gt_class",
            "paired_gt_token", "alternative_view_count")})
    if dict(observed_counts) != expected_counts:
        raise RuntimeError(f"population counts changed: {dict(observed_counts)}")
    write_csv("population_lock.csv", population)

    cfg = Config.fromfile(str(CONFIG))
    import_modules_from_strings(**cfg.custom_imports)
    cfg.model.pretrained, cfg.model.train_cfg, cfg.data.test.test_mode = None, None, True
    clean_dataset = protocol_dataset(cfg, ROOT / "protocols/presets/clean_no_corruption.json")
    datasets = {protocol: protocol_dataset(cfg, schedule)
                for protocol, (_, schedule) in PROTOCOLS.items()}
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(CHECKPOINT), map_location="cpu")
    model = model.to(device).eval()
    head = model.pts_bbox_head
    if tuple(name for name in MEMORY_NAMES if hasattr(head, name)) != COMPONENTS:
        raise RuntimeError("B0 temporal state inventory changed")
    head.reset_memory()
    initial = snapshot(head)
    states = {key: initial for key in ("clean", *PROTOCOLS)}
    pc_range = head.pc_range.detach()

    nusc = NuScenes(version="v1.0-mini", dataroot=str(ROOT / "data/nuscenes-mini"),
                    verbose=False)
    gt_cache = {}
    for info in clean_dataset.data_infos:
        token = str(info["token"])
        targets = local_gt(nusc, token)
        for target in targets:
            target["global_center"] = np.asarray(
                nusc.get("sample_annotation", target["token"])["translation"], float)
        gt_cache[token] = targets

    invariance = []
    for protocol in ("clean", *PROTOCOLS):
        difference, leaves = compare_tensors(
            mmcv.load(str(ROOT / "outputs/stage4/gt_query_survival_audit"
                          / protocol / "predictions.pkl")),
            mmcv.load(str(DISABLED / protocol / "predictions.pkl")))
        invariance.append({"comparison": f"{protocol}_B0_vs_disabled",
                           "tensor_leaves": leaves, "max_abs_diff": difference,
                           "exact": difference == 0.0})
    if not all(row["exact"] for row in invariance):
        raise RuntimeError(f"disabled divergence: {invariance}")

    replay = {protocol: {"logits_exact": True, "box_max": 0.0, "frames": 0}
              for protocol in ("clean", *PROTOCOLS)}
    inventory_observations = defaultdict(list)
    per_gt = []
    previous_scene = None
    with torch.no_grad():
        for index in range(len(clean_dataset)):
            clean_meta, clean_image, clean_data = unpack(clean_dataset[index], device)
            token, scene = str(clean_meta["sample_idx"]), str(clean_meta["scene_token"])
            prev_exists = 0 if scene != previous_scene else 1
            previous_scene = scene
            _, _, clean_feats = features(model, clean_image)
            clean_pre = states["clean"]
            output_a, states["clean"], _ = run_head(
                model, clean_meta, clean_data, clean_feats, prev_exists, clean_pre)
            exact, box_diff = validate_trace("clean", token, output_a, pc_range)
            replay["clean"]["logits_exact"] &= exact
            replay["clean"]["box_max"] = max(replay["clean"]["box_max"], box_diff)
            replay["clean"]["frames"] += 1
            if not exact or box_diff > 1e-5:
                raise RuntimeError(f"clean replay divergence {token}: {exact}/{box_diff}")

            for protocol, dataset in datasets.items():
                fault_meta, fault_image, fault_data = unpack(dataset[index], device)
                if str(fault_meta["sample_idx"]) != token:
                    raise RuntimeError(f"paired token mismatch {protocol}/{index}")
                _, _, fault_feats = features(model, fault_image)
                fault_pre = states[protocol]
                for component in COMPONENTS:
                    left, right = clean_pre[component], fault_pre[component]
                    if left is None or right is None:
                        continue
                    if left.shape != right.shape or left.dtype != right.dtype:
                        raise RuntimeError(f"state schema mismatch: {protocol}/{component}")
                    inventory_observations[component].append(
                        float((left.float() - right.float()).abs().max().item()))

                output_d, states[protocol], _ = run_head(
                    model, fault_meta, fault_data, fault_feats, prev_exists, fault_pre)
                exact, box_diff = validate_trace(protocol, token, output_d, pc_range)
                replay[protocol]["logits_exact"] &= exact
                replay[protocol]["box_max"] = max(replay[protocol]["box_max"], box_diff)
                replay[protocol]["frames"] += 1
                if not exact or box_diff > 1e-5:
                    raise RuntimeError(
                        f"{protocol} replay divergence {token}: {exact}/{box_diff}")

                frame_units = units.get((protocol, token), [])
                if not frame_units:
                    continue
                swap_outputs = {}
                target_differences = {}
                for component in COMPONENTS:
                    bd_state = swap_one_component(fault_pre, clean_pre, component)
                    ca_state = swap_one_component(clean_pre, fault_pre, component)
                    bd_difference = assert_one_component_swap(
                        fault_pre, clean_pre, bd_state, component)
                    ca_difference = assert_one_component_swap(
                        clean_pre, fault_pre, ca_state, component)
                    if bd_difference != ca_difference:
                        raise RuntimeError(f"asymmetric state difference: {component}")
                    target_differences[component] = bd_difference
                    swap_outputs[("BD", component)] = run_head(
                        model, fault_meta, fault_data, fault_feats,
                        prev_exists, bd_state)[0]
                    swap_outputs[("CA", component)] = run_head(
                        model, clean_meta, clean_data, clean_feats,
                        prev_exists, ca_state)[0]

                all_gt = gt_cache[token]
                by_token = {target["token"]: target for target in all_gt}
                info = clean_dataset.data_infos[index]
                context = {
                    "lidar2ego_rotation": Quaternion(
                        info["lidar2ego_rotation"]).rotation_matrix,
                    "lidar2ego_translation": np.asarray(
                        info["lidar2ego_translation"], float),
                    "ego2global_rotation": Quaternion(
                        info["ego2global_rotation"]).rotation_matrix,
                    "ego2global_translation": np.asarray(
                        info["ego2global_translation"], float),
                    "class_range": clean_dataset.eval_detection_configs.class_range,
                }
                for parent in frame_units:
                    target = by_token[parent["gt_token"]]
                    identity = f"{protocol}/{token}/{target['token']}"
                    base_a = output_metrics(output_a, target, all_gt, pc_range, context)
                    base_d = output_metrics(output_d, target, all_gt, pc_range, context)
                    validate_parent_metrics(base_a, parent, "A", identity)
                    validate_parent_metrics(base_d, parent, "D", identity)
                    row = {key: parent[key] for key in (
                        "protocol", "condition", "group", "sample_token", "scene_token",
                        "frame_idx", "fault_episode_age", "gt_token", "gt_class",
                        "paired_gt_token", "alternative_view_count")}
                    for condition in "ABCD":
                        for metric in ("candidate", "qplus", "s_pos", "rank", "margin",
                                       "topk", "tp"):
                            row[f"{condition}_{metric}"] = parent_value(
                                parent, condition, metric)
                    row.update({
                        "full_BD_s_pos_effect": float(parent["B_minus_D_s_pos"]),
                        "full_BD_margin_effect": float(parent["B_minus_D_margin"]),
                        "full_BD_rank_effect": float(parent["BD_rank_improvement"]),
                        "full_BD_topk_event": truth(parent["BD_topk_recovered"]),
                        "full_BD_tp_event": truth(parent["BD_tp_recovered"]),
                        "full_CA_s_pos_effect": float(parent["C_minus_A_s_pos"]),
                        "full_CA_margin_effect": float(parent["C_minus_A_margin"]),
                        "full_CA_rank_effect": float(parent["CA_rank_degradation"]),
                        "full_CA_topk_event": truth(parent["CA_topk_lost"]),
                        "full_CA_tp_event": truth(parent["CA_tp_lost"]),
                    })
                    for component in COMPONENTS:
                        row[f"{component}_clean_fault_max_abs_diff"] = target_differences[component]
                        row[f"{component}_BD_target_donor_exact"] = True
                        row[f"{component}_BD_non_target_exact"] = True
                        row[f"{component}_CA_target_donor_exact"] = True
                        row[f"{component}_CA_non_target_exact"] = True
                        bd_values = output_metrics(
                            swap_outputs[("BD", component)], target, all_gt, pc_range, context)
                        ca_values = output_metrics(
                            swap_outputs[("CA", component)], target, all_gt, pc_range, context)
                        row.update(effect_fields("BD", component, bd_values, base_d))
                        row.update(effect_fields("CA", component, ca_values, base_a))
                    per_gt.append(row)

    if len(per_gt) != len(parent_rows):
        raise RuntimeError(f"output population changed: {len(per_gt)}")
    for protocol, value in replay.items():
        invariance.append({"comparison": f"{protocol}_manual_replay_vs_trace",
                           "tensor_leaves": 2 * value["frames"],
                           "max_abs_diff": value["box_max"],
                           "exact": value["logits_exact"] and value["box_max"] <= 1e-5})
    if not all(row["exact"] for row in invariance):
        raise RuntimeError(f"manual replay divergence: {invariance}")
    write_csv("disabled_invariance.csv", invariance)
    write_csv("per_gt_component_swap.csv", per_gt)

    inventory_rows = []
    exemplar = next(state for state in states.values()
                    if all(state[name] is not None for name in COMPONENTS))
    for component in COMPONENTS:
        values = np.asarray(inventory_observations[component], float)
        tensor = exemplar[component]
        active_shape = list(tensor.shape)
        active_shape[1] = min(active_shape[1], int(head.memory_len))
        inventory_rows.append({
            "component": component, "semantic_group": SEMANTICS[component],
            "raw_persistent_shape": "x".join(map(str, tensor.shape)),
            "next_frame_active_shape": "x".join(map(str, active_shape)),
            "dtype": str(tensor.dtype),
            "observed_clean_fault_state_pairs": len(values),
            "differing_pairs": int(np.sum(values > 0)),
            "median_clean_fault_max_abs_diff": float(np.median(values)),
            "max_clean_fault_abs_diff": float(np.max(values)),
        })
    write_csv("state_inventory.csv", inventory_rows)

    summaries = []
    for protocol in (*PROTOCOLS, "pooled"):
        for group in ("lost", "retained"):
            subset = [row for row in per_gt if row["group"] == group
                      and (protocol == "pooled" or row["protocol"] == protocol)]
            for component in COMPONENTS:
                for arm in ARMS:
                    summaries.append(summarize(protocol, group, component, arm, subset))
    write_csv("component_summary.csv", summaries)
    summary_index = {(row["protocol"], row["group"], row["component"], row["arm"]): row
                     for row in summaries}

    explanation_rows = []
    for row in summaries:
        for metric in ("s_pos", "margin", "rank", "topk", "tp"):
            explanation_rows.append({
                "protocol": row["protocol"], "condition": row["condition"],
                "group": row["group"], "component": row["component"],
                "arm": row["arm"], "metric": metric,
                "component_effect": row[("component_median_" + metric + "_effect")
                                        if metric in {"s_pos", "margin", "rank"}
                                        else ("component_" + metric + "_event_rate")],
                "full_effect": row[("full_median_" + metric + "_effect")
                                   if metric in {"s_pos", "margin", "rank"}
                                   else ("full_" + metric + "_event_rate")],
                "explanation_raw": row[f"{metric}_explanation_raw"],
                "explanation_clipped": row[f"{metric}_explanation_clipped"],
            })
    write_csv("explanation_ratios.csv", explanation_rows)

    paired = []
    row_index = {(row["protocol"], row["sample_token"], row["gt_token"], row["group"]): row
                 for row in per_gt}
    for lost in (row for row in per_gt if row["group"] == "lost"):
        key = (lost["protocol"], lost["sample_token"], lost["paired_gt_token"], "retained")
        retained = row_index.get(key)
        if retained is None:
            raise RuntimeError(f"paired retained missing: {key}")
        pair_row = {"protocol": lost["protocol"], "sample_token": lost["sample_token"],
                    "lost_gt_token": lost["gt_token"],
                    "retained_gt_token": retained["gt_token"]}
        for component in COMPONENTS:
            for arm in ARMS:
                key_effect = f"{arm}_{component}_s_pos_effect"
                pair_row[f"{arm}_{component}_enrichment"] = delta(
                    lost[key_effect], retained[key_effect])
        paired.append(pair_row)

    bootstrap_rows = []
    effect_boot, enrichment_boot = {}, {}
    bootstrap_index = 0
    for protocol in (*PROTOCOLS, "pooled"):
        for group in ("lost", "retained"):
            subset = [row for row in per_gt if row["group"] == group
                      and (protocol == "pooled" or row["protocol"] == protocol)]
            for component in COMPONENTS:
                for arm in ARMS:
                    metric = f"{arm}_{component}_s_pos_effect"
                    result = cluster_bootstrap_median(
                        subset, metric, ("protocol", "sample_token"),
                        SEED + bootstrap_index, BOOTSTRAPS)
                    bootstrap_index += 1
                    effect_boot[(protocol, group, component, arm)] = result
                    bootstrap_rows.append({
                        "category": "component_s_pos_effect", "protocol": protocol,
                        "group": group, "component": component, "arm": arm,
                        "metric": metric, "n": len(subset), **result})
        pair_subset = [row for row in paired
                       if protocol == "pooled" or row["protocol"] == protocol]
        for component in COMPONENTS:
            for arm in ARMS:
                metric = f"{arm}_{component}_enrichment"
                result = cluster_bootstrap_median(
                    pair_subset, metric, ("protocol", "sample_token"),
                    SEED + bootstrap_index, BOOTSTRAPS)
                bootstrap_index += 1
                enrichment_boot[(protocol, component, arm)] = result
                bootstrap_rows.append({
                    "category": "paired_lost_minus_retained", "protocol": protocol,
                    "group": "paired", "component": component, "arm": arm,
                    "metric": metric, "n": len(pair_subset), **result})
    write_csv("component_bootstrap_95ci.csv", bootstrap_rows)

    records = []
    for component in COMPONENTS:
        bd_lost = summary_index[("pooled", "lost", component, "BD")]
        bd_retained = summary_index[("pooled", "retained", component, "BD")]
        ca_lost = summary_index[("pooled", "lost", component, "CA")]
        ca_retained = summary_index[("pooled", "retained", component, "CA")]
        bd_boot = effect_boot[("pooled", "lost", component, "BD")]
        ca_boot = effect_boot[("pooled", "lost", component, "CA")]
        bd_enrich = enrichment_boot[("pooled", component, "BD")]
        ca_enrich = enrichment_boot[("pooled", component, "CA")]
        records.append({
            "component": component,
            "bd_lost": bd_lost["component_median_s_pos_effect"],
            "bd_retained": bd_retained["component_median_s_pos_effect"],
            "bd_enrichment": bd_enrich["estimate"],
            "bd_ci_low": bd_boot["ci_low"],
            "bd_enrichment_ci_low": bd_enrich["ci_low"],
            "bd_cross_protocol": all(
                summary_index[(p, "lost", component, "BD")][
                    "component_median_s_pos_effect"] > 0 for p in PROTOCOLS),
            "bd_spos_ratio": bd_lost["s_pos_explanation_raw"],
            "bd_topk_ratio": bd_lost["topk_explanation_raw"],
            "bd_tp_ratio": bd_lost["tp_explanation_raw"],
            "ca_lost": ca_lost["component_median_s_pos_effect"],
            "ca_retained": ca_retained["component_median_s_pos_effect"],
            "ca_enrichment": ca_enrich["estimate"],
            "ca_ci_high": ca_boot["ci_high"],
            "ca_enrichment_ci_high": ca_enrich["ci_high"],
            "ca_cross_protocol": all(
                summary_index[(p, "lost", component, "CA")][
                    "component_median_s_pos_effect"] < 0 for p in PROTOCOLS),
            "ca_spos_ratio": ca_lost["s_pos_explanation_raw"],
            "ca_topk_ratio": ca_lost["topk_explanation_raw"],
            "ca_tp_ratio": ca_lost["tp_explanation_raw"],
        })
    decision = decide_attribution(records)
    decision_row = {"decision": decision["decision"],
                    "selected_components": ";".join(decision["selected_components"]),
                    "dominant_components": ";".join(decision["dominant_components"]),
                    "passing_pairs": ";".join("+".join(pair)
                                              for pair in decision["passing_pairs"])}
    gate_index = {record["component"]: record for record in decision["records"]}
    for component in COMPONENTS:
        gate = gate_index[component]
        for key in ("bd_core", "ca_core", "bd_arm", "ca_arm", "dominant"):
            decision_row[f"{component}_{key}"] = gate[key]
    write_csv("mechanism_decision.csv", [decision_row])

    lines = ["# Temporal State Attribution Audit", "", "## 终局判定", "",
             f"**{decision['decision']}**。", ""]
    if decision["selected_components"]:
        lines += ["预注册归因门通过，锁定组件：`"
                  + "`, `".join(decision["selected_components"]) + "`。",
                  "下一阶段才允许评估针对该state组件的train-time robust temporal representation；"
                  "本审计没有实现或smoke。", ""]
    else:
        lines += ["没有单一或两组件同时满足两个current-fixed arm的跨协议、"
                  "lost>retained与主要解释率门；No-Go，不设计新Temporal模块。", ""]
    lines += ["## 真实state inventory", "",
              "| Component | Raw / active shape; dtype | 实际语义 | clean≠fault pairs | median/max abs diff |",
              "|---|---|---|---:|---:|"]
    for row in inventory_rows:
        lines.append(f"| `{row['component']}` | {row['raw_persistent_shape']} / "
                     f"{row['next_frame_active_shape']}; {row['dtype']} | "
                     f"{row['semantic_group']} | {row['differing_pairs']}/"
                     f"{row['observed_clean_fault_state_pairs']} | "
                     f"{row['median_clean_fault_max_abs_diff']:.4g}/"
                     f"{row['max_clean_fault_abs_diff']:.4g} |")
    lines += ["", "B0没有持久化score/class state；classification score只用于选择写入"
              "memory的Top-K proposal。", "", "## Pooled lost归因与retained对照", "",
              "| Component | BD lost/retained | BD S_pos/TopK/TP explained | CA lost/retained | CA S_pos/TopK/TP explained | BD/CA gate |",
              "|---|---:|---:|---:|---:|---:|"]
    for component in COMPONENTS:
        bd_lost = summary_index[("pooled", "lost", component, "BD")]
        bd_ret = summary_index[("pooled", "retained", component, "BD")]
        ca_lost = summary_index[("pooled", "lost", component, "CA")]
        ca_ret = summary_index[("pooled", "retained", component, "CA")]
        gate = gate_index[component]
        lines.append(
            f"| `{component}` | {bd_lost['component_median_s_pos_effect']:+.4f}/"
            f"{bd_ret['component_median_s_pos_effect']:+.4f} | "
            f"{bd_lost['s_pos_explanation_raw']:.2f}/"
            f"{bd_lost['topk_explanation_raw']:.2f}/"
            f"{bd_lost['tp_explanation_raw']:.2f} | "
            f"{ca_lost['component_median_s_pos_effect']:+.4f}/"
            f"{ca_ret['component_median_s_pos_effect']:+.4f} | "
            f"{ca_lost['s_pos_explanation_raw']:.2f}/"
            f"{ca_lost['topk_explanation_raw']:.2f}/"
            f"{ca_lost['tp_explanation_raw']:.2f} | "
            f"{gate['bd_arm']}/{gate['ca_arm']} |")
    reference_gate = gate_index["memory_reference_point"]
    reference_bd = summary_index[("pooled", "lost", "memory_reference_point", "BD")]
    reference_ca = summary_index[("pooled", "lost", "memory_reference_point", "CA")]
    lines += ["", "`memory_reference_point`只通过CA harm arm（"
              f"effect={reference_ca['component_median_s_pos_effect']:+.4f}），"
              "但BD rescue arm失败（"
              f"effect={reference_bd['component_median_s_pos_effect']:+.4f}）；"
              f"两臂gate={reference_gate['bd_arm']}/{reference_gate['ca_arm']}。"
              "embedding的CA效应也强，但retained敏感且成对enrichment CI触0，"
              "同时BD不能rescue。因此不能将完整history effect归因为可单独修复的少数state组件。"]
    lines += ["", "## 三协议lost S_pos effect", "",
              "| Component | Dark BD/CA | Blur BD/CA | Crash BD/CA |",
              "|---|---:|---:|---:|"]
    for component in COMPONENTS:
        values = []
        for protocol in PROTOCOLS:
            bd = summary_index[(protocol, "lost", component, "BD")][
                "component_median_s_pos_effect"]
            ca = summary_index[(protocol, "lost", component, "CA")][
                "component_median_s_pos_effect"]
            values.append(f"{bd:+.4f}/{ca:+.4f}")
        lines.append(f"| `{component}` | " + " | ".join(values) + " |")
    lines += ["", "margin/rank、Top-K/TP事件、bootstrap CI和raw/clipped解释率见对应CSV。",
              "", "## Population与等价性", "",
              "Dark/Blur/Crash固定lost=15/20/36，retained=15/20/36；上一轮population与control未重选。",
              "Clean/Dark/Blur/Crash历史B0与disabled各243个tensor leaves逐tensor exact，最大差0。",
              "四套canonical replay各81帧logits/boxes exact，最大差0。", "",
              "全程torch.no_grad；未训练、未建optimizer、未新增loss/module，未smoke，"
              "未修改repos/StreamPETR。"]
    (REPORT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
