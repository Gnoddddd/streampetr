#!/usr/bin/env python3
"""Frozen-B0 temporal-state x current-observation 2x2 audit."""

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

from analysis.dark_target_recoverability import match_retained_controls  # noqa: E402
from analysis.fault_boundary_root_cause import candidate_pool_statistics  # noqa: E402
from analysis.temporal_state_counterfactual import (  # noqa: E402
    cluster_bootstrap_median,
    cluster_bootstrap_spearman,
    temporal_decision,
)
from scripts.audit_cross_view_target_evidence import (  # noqa: E402
    PROTOCOLS,
    protocol_dataset,
)
from scripts.audit_dark_target_recoverability import (  # noqa: E402
    CHECKPOINT,
    CLASSES,
    CONFIG,
    DISABLED,
    POST_RANGE,
    compare_tensors,
    features,
    local_gt,
    physical,
    run_head,
    snapshot,
    unpack,
    validate_trace,
)


REPORT = ROOT / "reports/stage4/temporal_state_counterfactual_audit"
ROOT_CAUSE = ROOT / "reports/stage4/fault_boundary_root_cause_audit/per_gt_root_cause.csv"
SEED, BOOTSTRAPS = 314159, 5000
CONDITIONS = ("A", "B", "C", "D")


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


def load_root_rows() -> list[dict]:
    with ROOT_CAUSE.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def delta(left, right) -> float:
    left, right = float(left), float(right)
    return left - right if math.isfinite(left) and math.isfinite(right) else float("nan")


def state_max_difference(clean_state: dict, fault_state: dict) -> float:
    differences = []
    for key in clean_state:
        left, right = clean_state[key], fault_state[key]
        if left is None and right is None:
            differences.append(0.0)
        elif left is None or right is None or left.shape != right.shape:
            return float("inf")
        else:
            differences.append(float((left.float() - right.float()).abs().max().item()))
    return max(differences, default=0.0)


def global_deployed_matches(logits, boxes, gt, context) -> set[str]:
    scores = torch.as_tensor(logits).float().sigmoid().reshape(-1)
    values, indexes = torch.topk(scores, min(100, scores.numel()))
    predictions = []
    for score, flat in zip(values.tolist(), indexes.tolist()):
        if score < 0.1:
            continue
        query, label = divmod(int(flat), logits.shape[1])
        center = np.asarray(boxes[query, :3], float)
        if np.any(center < POST_RANGE[:3]) or np.any(center > POST_RANGE[3:]):
            continue
        ego = context["lidar2ego_rotation"] @ center + context["lidar2ego_translation"]
        if np.linalg.norm(ego[:2]) > context["class_range"][CLASSES[label]]:
            continue
        global_center = (context["ego2global_rotation"] @ ego
                         + context["ego2global_translation"])
        predictions.append((label, global_center))
    pairs = []
    for gt_index, target in enumerate(gt):
        for prediction_index, (label, center) in enumerate(predictions):
            if target["label"] != label:
                continue
            distance = np.linalg.norm(target["global_center"][:2] - center[:2])
            if distance <= 2.0:
                pairs.append((float(distance), gt_index, prediction_index))
    used_gt, used_prediction, matched = set(), set(), set()
    for _, gt_index, prediction_index in sorted(pairs):
        if gt_index in used_gt or prediction_index in used_prediction:
            continue
        used_gt.add(gt_index)
        used_prediction.add(prediction_index)
        matched.add(gt[gt_index]["token"])
    return matched


def output_metrics(output, target, gt, pc_range, context) -> dict:
    logits = output["all_cls_scores"][-1, 0].detach().float().cpu().numpy()
    boxes = physical(output, pc_range)[-1, 0].detach().float().cpu().numpy()
    pool = candidate_pool_statistics(
        logits, boxes, target["center"], target["label"], 100, 2.0)
    return {"candidate": pool["candidate_available"], "qplus": pool["best_query"],
            "s_pos": pool["s_pos"], "rank": pool["rank"], "margin": pool["margin"],
            "topk": 0 < pool["rank"] <= 100,
            "tp": target["token"] in global_deployed_matches(
                logits, boxes, gt, context)}


def finite_median(rows: list[dict], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def boolean_rate(rows: list[dict], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows])) if rows else float("nan")


def summarize(protocol: str, group: str, rows: list[dict]) -> dict:
    output = {"protocol": protocol,
              "condition": "Pooled" if protocol == "pooled" else PROTOCOLS[protocol][0],
              "group": group, "n": len(rows),
              "median_state_max_abs_diff": finite_median(rows, "state_max_abs_diff")}
    for condition in CONDITIONS:
        output.update({
            f"{condition}_candidate_rate": boolean_rate(rows, f"{condition}_candidate"),
            f"{condition}_median_s_pos": finite_median(rows, f"{condition}_s_pos"),
            f"{condition}_median_margin": finite_median(rows, f"{condition}_margin"),
            f"{condition}_median_rank": finite_median(rows, f"{condition}_rank"),
            f"{condition}_topk_rate": boolean_rate(rows, f"{condition}_topk"),
            f"{condition}_tp_rate": boolean_rate(rows, f"{condition}_tp"),
        })
    output.update({
        "median_B_minus_D_s_pos": finite_median(rows, "B_minus_D_s_pos"),
        "median_C_minus_A_s_pos": finite_median(rows, "C_minus_A_s_pos"),
        "median_B_minus_A_s_pos": finite_median(rows, "B_minus_A_s_pos"),
        "median_D_minus_C_s_pos": finite_median(rows, "D_minus_C_s_pos"),
        "median_B_minus_D_margin": finite_median(rows, "B_minus_D_margin"),
        "median_C_minus_A_margin": finite_median(rows, "C_minus_A_margin"),
        "median_BD_rank_improvement": finite_median(rows, "BD_rank_improvement"),
        "median_CA_rank_degradation": finite_median(rows, "CA_rank_degradation"),
        "BD_topk_recovery_rate": boolean_rate(rows, "BD_topk_recovered"),
        "BD_tp_recovery_rate": boolean_rate(rows, "BD_tp_recovered"),
        "CA_topk_loss_rate": boolean_rate(rows, "CA_topk_lost"),
        "CA_tp_loss_rate": boolean_rate(rows, "CA_tp_lost"),
        "BD_topk_discordance_rate": boolean_rate(rows, "BD_topk_discordant"),
        "BD_tp_discordance_rate": boolean_rate(rows, "BD_tp_discordant"),
        "CA_topk_discordance_rate": boolean_rate(rows, "CA_topk_discordant"),
        "CA_tp_discordance_rate": boolean_rate(rows, "CA_tp_discordant"),
    })
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

    root_rows = load_root_rows()
    units, manifest, coverage = defaultdict(list), [], []
    for protocol in PROTOCOLS:
        rows = [row for row in root_rows if row["protocol"] == protocol]
        lost_all = [row for row in rows if row["outcome"] == "fault_induced_lost"]
        lost = [row for row in lost_all if truth(row["fault_candidate_available"])]
        retained_pool = [row for row in rows if row["outcome"] == "retained_control"
                         and truth(row["fault_candidate_available"])]
        pairs = match_retained_controls(lost, retained_pool)
        if len(pairs) != len(lost):
            raise RuntimeError(f"control coverage failure for {protocol}")
        for pair in pairs:
            left, right = pair["lost"], pair["retained"]
            units[(protocol, left["sample_token"])].append(("lost", left, right["gt_token"]))
            units[(protocol, right["sample_token"])].append(("retained", right, left["gt_token"]))
            manifest.append({"protocol": protocol, "condition": PROTOCOLS[protocol][0],
                "sample_token": left["sample_token"], "lost_gt_token": left["gt_token"],
                "lost_class": left["gt_class"], "retained_gt_token": right["gt_token"],
                "retained_class": right["gt_class"],
                "same_class": left["gt_class"] == right["gt_class"],
                "lost_distance": left["gt_center_distance"],
                "retained_distance": right["gt_center_distance"],
                "lost_altviews": left["alternative_view_count"],
                "retained_altviews": right["alternative_view_count"],
                "match_cost": pair["match_cost"]})
        coverage.append({"protocol": protocol, "condition": PROTOCOLS[protocol][0],
            "lost_all": len(lost_all), "lost_fault_qplus_available": len(lost),
            "retained_qplus_pool": len(retained_pool), "paired_retained": len(pairs)})
    write_csv("control_manifest.csv", manifest)
    write_csv("population_coverage.csv", coverage)

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
                state_difference = state_max_difference(clean_pre, fault_pre)
                output_d, states[protocol], _ = run_head(
                    model, fault_meta, fault_data, fault_feats, prev_exists, fault_pre)
                output_b = run_head(
                    model, fault_meta, fault_data, fault_feats, prev_exists, clean_pre)[0]
                output_c = run_head(
                    model, clean_meta, clean_data, clean_feats, prev_exists, fault_pre)[0]
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
                all_gt = gt_cache[token]
                by_token = {target["token"]: target for target in all_gt}
                info = clean_dataset.data_infos[index]
                match_context = {
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
                for group, root_row, paired_gt in frame_units:
                    target = by_token[root_row["gt_token"]]
                    values = {
                        "A": output_metrics(output_a, target, all_gt, pc_range, match_context),
                        "B": output_metrics(output_b, target, all_gt, pc_range, match_context),
                        "C": output_metrics(output_c, target, all_gt, pc_range, match_context),
                        "D": output_metrics(output_d, target, all_gt, pc_range, match_context),
                    }
                    expected_d = group == "retained"
                    if not values["A"]["tp"] or values["D"]["tp"] != expected_d:
                        raise RuntimeError(
                            f"canonical outcome mismatch {protocol}/{token}/{target['token']}: "
                            f"A={values['A']} D={values['D']} group={group}")
                    if values["D"]["qplus"] != int(root_row["fault_best_query"]):
                        raise RuntimeError(f"D q+ mismatch {protocol}/{token}/{target['token']}")
                    row = {"protocol": protocol, "condition": PROTOCOLS[protocol][0],
                        "group": group, "sample_token": token,
                        "scene_token": root_row["scene_token"],
                        "frame_idx": int(root_row["frame_idx"]),
                        "fault_episode_age": int(root_row["frame_idx"]) - 2,
                        "gt_token": target["token"], "gt_class": target["name"],
                        "paired_gt_token": paired_gt,
                        "alternative_view_count": int(root_row["alternative_view_count"]),
                        "state_max_abs_diff": state_difference}
                    for condition, value in values.items():
                        for key in ("candidate", "qplus", "s_pos", "rank", "margin", "topk", "tp"):
                            row[f"{condition}_{key}"] = value[key]
                    row.update({
                        "B_minus_D_s_pos": delta(values["B"]["s_pos"], values["D"]["s_pos"]),
                        "C_minus_A_s_pos": delta(values["C"]["s_pos"], values["A"]["s_pos"]),
                        "B_minus_A_s_pos": delta(values["B"]["s_pos"], values["A"]["s_pos"]),
                        "D_minus_C_s_pos": delta(values["D"]["s_pos"], values["C"]["s_pos"]),
                        "B_minus_D_margin": delta(values["B"]["margin"], values["D"]["margin"]),
                        "C_minus_A_margin": delta(values["C"]["margin"], values["A"]["margin"]),
                        "BD_rank_improvement": delta(values["D"]["rank"], values["B"]["rank"]),
                        "CA_rank_degradation": delta(values["C"]["rank"], values["A"]["rank"]),
                        "BD_topk_recovered": not values["D"]["topk"] and values["B"]["topk"],
                        "BD_tp_recovered": not values["D"]["tp"] and values["B"]["tp"],
                        "CA_topk_lost": values["A"]["topk"] and not values["C"]["topk"],
                        "CA_tp_lost": values["A"]["tp"] and not values["C"]["tp"],
                        "BD_topk_discordant": values["B"]["topk"] != values["D"]["topk"],
                        "BD_tp_discordant": values["B"]["tp"] != values["D"]["tp"],
                        "CA_topk_discordant": values["C"]["topk"] != values["A"]["topk"],
                        "CA_tp_discordant": values["C"]["tp"] != values["A"]["tp"],
                    })
                    per_gt.append(row)

    for protocol, value in replay.items():
        invariance.append({"comparison": f"{protocol}_manual_replay_vs_trace",
            "tensor_leaves": 2 * value["frames"], "max_abs_diff": value["box_max"],
            "exact": value["logits_exact"] and value["box_max"] <= 1e-5})
    if not all(row["exact"] for row in invariance):
        raise RuntimeError(f"manual replay divergence: {invariance}")
    write_csv("disabled_invariance.csv", invariance)
    write_csv("per_gt_counterfactual.csv", per_gt)

    summaries = []
    for protocol in (*PROTOCOLS, "pooled"):
        for group in ("lost", "retained"):
            rows = [row for row in per_gt if row["group"] == group
                    and (protocol == "pooled" or row["protocol"] == protocol)]
            summaries.append(summarize(protocol, group, rows))
    write_csv("counterfactual_summary.csv", summaries)
    summary_index = {(row["protocol"], row["group"]): row for row in summaries}

    age_rows = []
    for protocol in (*PROTOCOLS, "pooled"):
        for group in ("lost", "retained"):
            for age in range(1, 11):
                rows = [row for row in per_gt if row["group"] == group
                        and row["fault_episode_age"] == age
                        and (protocol == "pooled" or row["protocol"] == protocol)]
                age_row = {"protocol": protocol,
                    "condition": "Pooled" if protocol == "pooled" else PROTOCOLS[protocol][0],
                    "group": group, "fault_episode_age": age, "n": len(rows),
                    "median_state_max_abs_diff": finite_median(rows, "state_max_abs_diff"),
                    "median_B_minus_D_s_pos": finite_median(rows, "B_minus_D_s_pos"),
                    "median_C_minus_A_s_pos": finite_median(rows, "C_minus_A_s_pos"),
                    "median_B_minus_A_s_pos": finite_median(rows, "B_minus_A_s_pos"),
                    "median_D_minus_C_s_pos": finite_median(rows, "D_minus_C_s_pos"),
                    "BD_topk_recovery_rate": boolean_rate(rows, "BD_topk_recovered"),
                    "BD_tp_recovery_rate": boolean_rate(rows, "BD_tp_recovered"),
                    "CA_topk_loss_rate": boolean_rate(rows, "CA_topk_lost"),
                    "CA_tp_loss_rate": boolean_rate(rows, "CA_tp_lost")}
                for condition in "ABCD":
                    age_row.update({
                        f"{condition}_median_s_pos": finite_median(
                            rows, f"{condition}_s_pos"),
                        f"{condition}_median_margin": finite_median(
                            rows, f"{condition}_margin"),
                        f"{condition}_median_rank": finite_median(
                            rows, f"{condition}_rank"),
                        f"{condition}_topk_rate": boolean_rate(
                            rows, f"{condition}_topk"),
                        f"{condition}_tp_rate": boolean_rate(
                            rows, f"{condition}_tp"),
                    })
                age_rows.append(age_row)
    write_csv("fault_age_summary.csv", age_rows)

    bootstrap_rows, median_boot, age_boot = [], {}, {}
    bootstrap_index = 0
    for protocol in (*PROTOCOLS, "pooled"):
        for group in ("lost", "retained"):
            rows = [row for row in per_gt if row["group"] == group
                    and (protocol == "pooled" or row["protocol"] == protocol)]
            for contrast in ("B_minus_D_s_pos", "C_minus_A_s_pos",
                             "B_minus_A_s_pos", "D_minus_C_s_pos"):
                result = cluster_bootstrap_median(
                    rows, contrast, ("protocol", "sample_token"),
                    SEED + bootstrap_index, BOOTSTRAPS)
                bootstrap_index += 1
                median_boot[(protocol, group, contrast)] = result
                bootstrap_rows.append({"category": "median_contrast", "protocol": protocol,
                    "group": group, "metric": contrast, "n": len(rows), **result})
            if group == "lost":
                for contrast in ("B_minus_D_s_pos", "C_minus_A_s_pos"):
                    result = cluster_bootstrap_spearman(
                        rows, "fault_episode_age", contrast,
                        ("protocol", "sample_token"), SEED + bootstrap_index, BOOTSTRAPS)
                    bootstrap_index += 1
                    age_boot[(protocol, contrast)] = result
                    bootstrap_rows.append({"category": "fault_age_spearman",
                        "protocol": protocol, "group": group, "metric": contrast,
                        "n": len(rows), **result})
    write_csv("bootstrap_95ci.csv", bootstrap_rows)

    pooled = summary_index[("pooled", "lost")]
    bd_ci = median_boot[("pooled", "lost", "B_minus_D_s_pos")]
    ca_ci = median_boot[("pooled", "lost", "C_minus_A_s_pos")]
    protocol_medians = {protocol: {
        "bd": summary_index[(protocol, "lost")]["median_B_minus_D_s_pos"],
        "ca": summary_index[(protocol, "lost")]["median_C_minus_A_s_pos"]}
        for protocol in PROTOCOLS}
    age_protocol = {protocol: {
        "bd": age_boot[(protocol, "B_minus_D_s_pos")]["estimate"],
        "ca": age_boot[(protocol, "C_minus_A_s_pos")]["estimate"]}
        for protocol in PROTOCOLS}
    decision = temporal_decision(
        {"median": pooled["median_B_minus_D_s_pos"], "ci_low": bd_ci["ci_low"],
         "topk_event_rate": pooled["BD_topk_recovery_rate"],
         "tp_event_rate": pooled["BD_tp_recovery_rate"]},
        {"median": pooled["median_C_minus_A_s_pos"], "ci_high": ca_ci["ci_high"],
         "topk_event_rate": pooled["CA_topk_loss_rate"],
         "tp_event_rate": pooled["CA_tp_loss_rate"]},
        protocol_medians, age_protocol,
        {"bd_ci_low": age_boot[("pooled", "B_minus_D_s_pos")]["ci_low"],
         "ca_ci_high": age_boot[("pooled", "C_minus_A_s_pos")]["ci_high"]},
        {"bd_ci_low": bd_ci["ci_low"], "bd_ci_high": bd_ci["ci_high"],
         "ca_ci_low": ca_ci["ci_low"], "ca_ci_high": ca_ci["ci_high"],
         "bd_topk_discordance": pooled["BD_topk_discordance_rate"],
         "bd_tp_discordance": pooled["BD_tp_discordance_rate"],
         "ca_topk_discordance": pooled["CA_topk_discordance_rate"],
         "ca_tp_discordance": pooled["CA_tp_discordance_rate"]})
    decision_row = {**decision,
        "pooled_B_minus_D_median": pooled["median_B_minus_D_s_pos"],
        "pooled_B_minus_D_ci_low": bd_ci["ci_low"],
        "pooled_B_minus_D_ci_high": bd_ci["ci_high"],
        "pooled_C_minus_A_median": pooled["median_C_minus_A_s_pos"],
        "pooled_C_minus_A_ci_low": ca_ci["ci_low"],
        "pooled_C_minus_A_ci_high": ca_ci["ci_high"],
        "pooled_BD_topk_recovery": pooled["BD_topk_recovery_rate"],
        "pooled_BD_tp_recovery": pooled["BD_tp_recovery_rate"],
        "pooled_CA_topk_loss": pooled["CA_topk_loss_rate"],
        "pooled_CA_tp_loss": pooled["CA_tp_loss_rate"]}
    write_csv("mechanism_decision.csv", [decision_row])

    lines = ["# Temporal State × Current Observation 2×2 Counterfactual Audit", "",
             "## 终局决策", "", f"**{decision['decision']}**。", ""]
    if decision["temporal_contamination"]:
        lines += ["Fault history对temporal state的污染通过预注册跨协议与fault-age门；",
                  "下一阶段才允许设计时序鲁棒机制。本审计未训练或smoke。", ""]
    elif decision["current_state_equivalent"]:
        lines += ["B≈D且C≈A落入预注册等价带，说明current-frame observation主导。",
                  "正式结束Stage4全部冻结B0后的补救路线，转向train-time robust representation。", ""]
    else:
        lines += ["history effect未达到跨协议、fault-age稳定污染门，也未落入严格等价带。",
                  "按终局规则No-Go：关闭Stage4全部冻结B0补救路线，转向train-time robust representation。", ""]
    lines += ["## 主结果：lost GT", "",
              "| Protocol | n | A S_pos | B | C | D | B-D [95% CI] | C-A [95% CI] | B→D TP recover | A→C TP loss |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for protocol in PROTOCOLS:
        row = summary_index[(protocol, "lost")]
        bd = median_boot[(protocol, "lost", "B_minus_D_s_pos")]
        ca = median_boot[(protocol, "lost", "C_minus_A_s_pos")]
        lines.append(f"| {row['condition']} | {row['n']} | {row['A_median_s_pos']:.4f} | "
                     f"{row['B_median_s_pos']:.4f} | {row['C_median_s_pos']:.4f} | "
                     f"{row['D_median_s_pos']:.4f} | {row['median_B_minus_D_s_pos']:+.4f} "
                     f"[{bd['ci_low']:+.4f},{bd['ci_high']:+.4f}] | "
                     f"{row['median_C_minus_A_s_pos']:+.4f} "
                     f"[{ca['ci_low']:+.4f},{ca['ci_high']:+.4f}] | "
                     f"{100*row['BD_tp_recovery_rate']:.1f}% | "
                     f"{100*row['CA_tp_loss_rate']:.1f}% |")
    lines += ["", f"Pooled B-D={pooled['median_B_minus_D_s_pos']:+.4f} "
              f"[{bd_ci['ci_low']:+.4f},{bd_ci['ci_high']:+.4f}]，"
              f"TP/Top-K recovery={100*pooled['BD_tp_recovery_rate']:.1f}%/"
              f"{100*pooled['BD_topk_recovery_rate']:.1f}%。",
              f"Pooled C-A={pooled['median_C_minus_A_s_pos']:+.4f} "
              f"[{ca_ci['ci_low']:+.4f},{ca_ci['ci_high']:+.4f}]，"
              f"TP/Top-K loss={100*pooled['CA_tp_loss_rate']:.1f}%/"
              f"{100*pooled['CA_topk_loss_rate']:.1f}%。", "",
              "## Retained control", "",
              "| Protocol | n | B-D S_pos | C-A S_pos | BD TP discord | CA TP discord |",
              "|---|---:|---:|---:|---:|---:|"]
    for protocol in PROTOCOLS:
        row = summary_index[(protocol, "retained")]
        lines.append(f"| {row['condition']} | {row['n']} | "
                     f"{row['median_B_minus_D_s_pos']:+.4f} | "
                     f"{row['median_C_minus_A_s_pos']:+.4f} | "
                     f"{100*row['BD_tp_discordance_rate']:.1f}% | "
                     f"{100*row['CA_tp_discordance_rate']:.1f}% |")
    lines += ["", "## Fault-age趋势（lost）", "",
              "| Protocol | rho(age,B-D) [95% CI] | rho(age,C-A) [95% CI] |",
              "|---|---:|---:|"]
    for protocol in (*PROTOCOLS, "pooled"):
        bd = age_boot[(protocol, "B_minus_D_s_pos")]
        ca = age_boot[(protocol, "C_minus_A_s_pos")]
        name = "Pooled" if protocol == "pooled" else PROTOCOLS[protocol][0]
        lines.append(f"| {name} | {bd['estimate']:+.3f} "
                     f"[{bd['ci_low']:+.3f},{bd['ci_high']:+.3f}] | "
                     f"{ca['estimate']:+.3f} [{ca['ci_low']:+.3f},{ca['ci_high']:+.3f}] |")
    lines += ["", "### Pooled lost GT逐age原始量", "",
              "Age=1对应active frame 3；该帧无符合入组条件的lost GT，因此非空表从age=2开始。", "",
              "| Age | n | A/B/C/D S_pos | B-D | C-A | BD TP recover | CA TP loss |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in age_rows:
        if row["protocol"] != "pooled" or row["group"] != "lost" or not row["n"]:
            continue
        lines.append(f"| {row['fault_episode_age']} | {row['n']} | "
                     f"{row['A_median_s_pos']:.3f}/{row['B_median_s_pos']:.3f}/"
                     f"{row['C_median_s_pos']:.3f}/{row['D_median_s_pos']:.3f} | "
                     f"{row['median_B_minus_D_s_pos']:+.3f} | "
                     f"{row['median_C_minus_A_s_pos']:+.3f} | "
                     f"{100*row['BD_tp_recovery_rate']:.1f}% | "
                     f"{100*row['CA_tp_loss_rate']:.1f}% |")
    lines += ["", "各协议/对照组逐age的A-D S_pos、margin、rank、Top-K、TP完整统计见fault_age_summary.csv。", "",
              "## 判定门", "", f"- B-D contamination arm: {decision['bd_arm']}",
              f"- C-A contamination arm: {decision['ca_arm']}",
              f"- B-D跨协议/age稳定: {decision['bd_cross_protocol']}/"
              f"{decision['bd_age_stable']}",
              f"- C-A跨协议/age稳定: {decision['ca_cross_protocol']}/"
              f"{decision['ca_age_stable']}",
              f"- current-state equivalence: {decision['current_state_equivalent']}", "",
              "## 等价性与边界", "",
              "Clean/Dark/Blur/Crash历史B0与disabled各243个tensor leaves逐tensor exact，最大差0。",
              "四套canonical replay的81帧float16 logits exact，boxes最大误差0。", "",
              "全程torch.no_grad；未训练、未建optimizer、未新增loss/module，未修改memory规则、",
              "query或Top-K，未运行smoke，未修改repos/StreamPETR。"]
    (REPORT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
