#!/usr/bin/env python3
"""Stage5 LiDAR teacher coverage and cross-modal signal audit."""

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
from mmcv.parallel import MMDataParallel  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmdet3d.datasets import build_dataloader, build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from nuscenes.eval.detection.utils import category_to_detection_name  # noqa: E402
from nuscenes.nuscenes import NuScenes  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402

from analysis.lidar_privileged_signal import (  # noqa: E402
    circular_error,
    cosine_distance,
    greedy_class_center_match,
    sample_bev_features,
    teacher_coverage_decision,
)
from scripts.audit_dark_target_recoverability import (  # noqa: E402
    DISABLED,
    compare_tensors,
)


REPORT = ROOT / "reports/stage5/lidar_privileged_signal_audit"
POPULATION = ROOT / "reports/stage5/temporal_state_attribution_audit/population_lock.csv"
TEACHER_CONFIG = ROOT / "configs/stage5/centerpoint_01voxel_nuscenes_mini_val.py"
TEACHER_CHECKPOINT = ROOT / "outputs/stage5/lidar_privileged_signal_audit/teacher/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth"
CLASSES = ("car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
           "motorcycle", "bicycle", "pedestrian", "traffic_cone")
CLASS_INDEX = {name: index for index, name in enumerate(CLASSES)}
PROTOCOL_NAMES = {"dark_back": "CAM_BACK Dark",
                  "blur_back": "CAM_BACK Motion Blur",
                  "crash_back": "CAM_BACK Crash"}


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


def finite_median(rows: list[dict], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def rate(rows: list[dict], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows])) if rows else float("nan")


def gt_targets(nusc: NuScenes, sample_token: str) -> list[dict]:
    sample = nusc.get("sample", sample_token)
    lidar_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    calibrated = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    global_to_lidar = (Quaternion(calibrated["rotation"]).inverse.rotation_matrix
                       @ Quaternion(ego_pose["rotation"]).inverse.rotation_matrix)
    _, boxes, _ = nusc.get_sample_data(sample["data"]["LIDAR_TOP"])
    output = []
    for box in boxes:
        name = category_to_detection_name(box.name)
        if name not in CLASS_INDEX:
            continue
        annotation = nusc.get("sample_annotation", box.token)
        velocity_global = np.asarray(nusc.box_velocity(box.token), float)
        velocity_lidar = global_to_lidar @ velocity_global
        output.append({
            "token": str(box.token), "instance_token": annotation["instance_token"],
            "label": CLASS_INDEX[name], "name": name,
            "center": np.asarray(box.center, float),
            "size": np.asarray(box.wlh, float),
            "yaw": float(box.orientation.yaw_pitch_roll[0]),
            "velocity": np.asarray(velocity_lidar[:2], float),
            "num_lidar_pts": int(annotation["num_lidar_pts"]),
        })
    return output


def teacher_frame(model, data: dict, targets: list[dict], capture: dict) -> dict:
    capture.clear()
    with torch.no_grad():
        output = model(return_loss=False, rescale=True, **data)[0]["pts_bbox"]
    feature = capture.pop("feature")
    boxes = output["boxes_3d"]
    centers = boxes.gravity_center.detach().float().cpu().numpy()
    dimensions = boxes.dims.detach().float().cpu().numpy()
    yaws = boxes.yaw.detach().float().cpu().numpy()
    tensor = boxes.tensor.detach().float().cpu().numpy()
    velocities = tensor[:, 7:9] if tensor.shape[1] >= 9 else np.full((len(tensor), 2), np.nan)
    scores = output["scores_3d"].detach().float().cpu().numpy()
    labels = output["labels_3d"].detach().long().cpu().numpy()
    matches = greedy_class_center_match(targets, labels, centers, scores, 2.0)
    representations = sample_bev_features(
        feature, np.asarray([target["center"] for target in targets], float))
    result = {}
    for index, target in enumerate(targets):
        prediction = matches.get(target["token"])
        row = {"teacher_matched": prediction is not None,
               "teacher_representation": representations[index],
               "teacher_representation_norm": float(np.linalg.norm(representations[index]))}
        if prediction is None:
            row.update({key: float("nan") for key in (
                "teacher_score", "teacher_xy_center_error", "teacher_3d_center_error",
                "teacher_abs_size_l1", "teacher_relative_size_l1",
                "teacher_yaw_error_rad", "teacher_velocity_error")})
            row.update({"teacher_center": np.full(3, np.nan),
                        "teacher_size": np.full(3, np.nan),
                        "teacher_yaw": float("nan"),
                        "teacher_velocity": np.full(2, np.nan)})
        else:
            center_delta = centers[prediction] - target["center"]
            size_delta = np.abs(dimensions[prediction] - target["size"])
            velocity_error = float(np.linalg.norm(
                velocities[prediction] - target["velocity"]))
            row.update({
                "teacher_score": float(scores[prediction]),
                "teacher_xy_center_error": float(np.linalg.norm(center_delta[:2])),
                "teacher_3d_center_error": float(np.linalg.norm(center_delta)),
                "teacher_abs_size_l1": float(np.mean(size_delta)),
                "teacher_relative_size_l1": float(np.mean(
                    size_delta / np.maximum(target["size"], 1e-6))),
                "teacher_yaw_error_rad": circular_error(
                    yaws[prediction], target["yaw"]),
                "teacher_velocity_error": velocity_error,
                "teacher_center": centers[prediction].copy(),
                "teacher_size": dimensions[prediction].copy(),
                "teacher_yaw": float(yaws[prediction]),
                "teacher_velocity": velocities[prediction].copy(),
            })
        result[target["token"]] = {**target, **row}
    return result


def summary_row(protocol: str, group: str, rows: list[dict]) -> dict:
    matched = [row for row in rows if row["teacher_matched"]]
    return {
        "protocol": protocol,
        "condition": "Pooled" if protocol == "pooled" else PROTOCOL_NAMES[protocol],
        "group": group, "n": len(rows),
        "lidar_supported_rate": rate(rows, "lidar_supported"),
        "teacher_match_rate": rate(rows, "teacher_matched"),
        "matched_n": len(matched),
        "median_teacher_score": finite_median(matched, "teacher_score"),
        "median_xy_center_error": finite_median(matched, "teacher_xy_center_error"),
        "median_3d_center_error": finite_median(matched, "teacher_3d_center_error"),
        "median_abs_size_l1": finite_median(matched, "teacher_abs_size_l1"),
        "median_relative_size_l1": finite_median(matched, "teacher_relative_size_l1"),
        "median_yaw_error_deg": math.degrees(
            finite_median(matched, "teacher_yaw_error_rad")),
        "median_velocity_error": finite_median(matched, "teacher_velocity_error"),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    REPORT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with POPULATION.open(encoding="utf-8") as handle:
        population = list(csv.DictReader(handle))
    if len(population) != 142:
        raise RuntimeError(f"population changed: {len(population)}")

    cfg = Config.fromfile(str(TEACHER_CONFIG))
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                              dist=False, shuffle=False)
    teacher = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(teacher, str(TEACHER_CHECKPOINT), map_location="cpu")
    teacher = MMDataParallel(teacher.cuda().eval(), device_ids=[0])
    capture = {}

    def capture_neck(_, __, output):
        capture["feature"] = output[0].detach()

    handle = teacher.module.pts_neck.register_forward_hook(capture_neck)
    nusc = NuScenes(version="v1.0-mini", dataroot=str(ROOT / "data/nuscenes-mini"),
                    verbose=False)
    population_scenes = {row["scene_token"] for row in population}
    relevant_tokens = {str(info["token"]) for info in dataset.data_infos
                       if str(info["scene_token"]) in population_scenes
                       and 3 <= int(info["frame_idx"]) <= 12}
    frame_cache = {}
    try:
        for index, data in enumerate(loader):
            info = dataset.data_infos[index]
            token = str(info["token"])
            if token not in relevant_tokens:
                continue
            targets = gt_targets(nusc, token)
            frame_cache[token] = teacher_frame(teacher, data, targets, capture)
    finally:
        handle.remove()
    if set(frame_cache) != relevant_tokens:
        raise RuntimeError("teacher frame coverage incomplete")

    coverage_rows = []
    population_specs = set()
    for source in population:
        annotation = nusc.get("sample_annotation", source["gt_token"])
        population_specs.add((source["protocol"], source["group"],
                              source["scene_token"], annotation["instance_token"]))
        target = frame_cache[source["sample_token"]].get(source["gt_token"])
        if target is None:
            raise RuntimeError(f"GT absent in teacher frame: {source['gt_token']}")
        row = {key: source[key] for key in source}
        row.update({
            "instance_token": annotation["instance_token"],
            "num_lidar_pts": target["num_lidar_pts"],
            "lidar_supported": target["num_lidar_pts"] > 0,
            "teacher_matched": target["teacher_matched"],
            "teacher_score": target["teacher_score"],
            "teacher_xy_center_error": target["teacher_xy_center_error"],
            "teacher_3d_center_error": target["teacher_3d_center_error"],
            "teacher_abs_size_l1": target["teacher_abs_size_l1"],
            "teacher_relative_size_l1": target["teacher_relative_size_l1"],
            "teacher_yaw_error_rad": target["teacher_yaw_error_rad"],
            "teacher_yaw_error_deg": math.degrees(target["teacher_yaw_error_rad"]),
            "teacher_velocity_error": target["teacher_velocity_error"],
            "teacher_representation_norm": target["teacher_representation_norm"],
        })
        coverage_rows.append(row)
    write_csv("teacher_coverage.csv", coverage_rows)

    summary_rows = []
    for protocol in (*PROTOCOL_NAMES, "pooled"):
        for group in ("lost", "retained"):
            rows = [row for row in coverage_rows if row["group"] == group
                    and (protocol == "pooled" or row["protocol"] == protocol)]
            summary_rows.append(summary_row(protocol, group, rows))
    write_csv("teacher_coverage_summary.csv", summary_rows)
    summary_index = {(row["protocol"], row["group"]): row for row in summary_rows}

    info_by_scene = defaultdict(list)
    for info in dataset.data_infos:
        if str(info["token"]) in relevant_tokens:
            info_by_scene[str(info["scene_token"])].append(info)
    temporal_rows = []
    for protocol, group, scene, instance in sorted(population_specs):
        observations = []
        for info in sorted(info_by_scene[scene], key=lambda value: int(value["frame_idx"])):
            token = str(info["token"])
            target = next((value for value in frame_cache[token].values()
                           if value["instance_token"] == instance), None)
            if target is not None and target["teacher_matched"]:
                observations.append((int(info["frame_idx"]), token, target))
        for left, right in zip(observations, observations[1:]):
            if right[0] - left[0] != 1:
                continue
            temporal_rows.append({
                "protocol": protocol, "condition": PROTOCOL_NAMES[protocol],
                "group": group, "scene_token": scene, "instance_token": instance,
                "left_frame_idx": left[0], "right_frame_idx": right[0],
                "left_sample_token": left[1], "right_sample_token": right[1],
                "abs_teacher_score_delta": abs(
                    right[2]["teacher_score"] - left[2]["teacher_score"]),
                "abs_teacher_center_error_delta": abs(
                    right[2]["teacher_xy_center_error"]
                    - left[2]["teacher_xy_center_error"]),
                "teacher_representation_cosine_distance": cosine_distance(
                    left[2]["teacher_representation"],
                    right[2]["teacher_representation"]),
            })
    if not temporal_rows:
        temporal_rows = [{"not_run_reason": "no adjacent matched instance pairs"}]
    write_csv("teacher_temporal_stability.csv", temporal_rows)
    lost_temporal = [row for row in temporal_rows if row.get("group") == "lost"]
    temporal_summary = {
        "lost_pair_count": len(lost_temporal),
        "median_abs_score_delta": finite_median(
            lost_temporal, "abs_teacher_score_delta"),
        "median_abs_center_error_delta": finite_median(
            lost_temporal, "abs_teacher_center_error_delta"),
        "median_representation_cosine_distance": finite_median(
            lost_temporal, "teacher_representation_cosine_distance"),
    }
    protocols = {protocol: {
        "lost_match_rate": summary_index[(protocol, "lost")]["teacher_match_rate"],
        "lost_median_score": summary_index[(protocol, "lost")]["median_teacher_score"],
        "lost_median_xy_error": summary_index[(protocol, "lost")][
            "median_xy_center_error"],
    } for protocol in PROTOCOL_NAMES}
    pooled = {
        "lost_match_rate": summary_index[("pooled", "lost")]["teacher_match_rate"],
        "retained_match_rate": summary_index[("pooled", "retained")][
            "teacher_match_rate"],
        "lost_median_score": summary_index[("pooled", "lost")][
            "median_teacher_score"],
        "lost_median_xy_error": summary_index[("pooled", "lost")][
            "median_xy_center_error"],
        "lost_median_relative_size_l1": summary_index[("pooled", "lost")][
            "median_relative_size_l1"],
        "lost_median_yaw_error_deg": summary_index[("pooled", "lost")][
            "median_yaw_error_deg"],
    }
    decision = teacher_coverage_decision(protocols, pooled, temporal_summary)

    invariance = []
    for protocol in ("clean", *PROTOCOL_NAMES):
        difference, leaves = compare_tensors(
            mmcv.load(str(ROOT / "outputs/stage4/gt_query_survival_audit"
                          / protocol / "predictions.pkl")),
            mmcv.load(str(DISABLED / protocol / "predictions.pkl")))
        invariance.append({"comparison": f"{protocol}_B0_vs_disabled",
                           "tensor_leaves": leaves, "max_abs_diff": difference,
                           "exact": difference == 0.0})
    if not all(row["exact"] for row in invariance):
        raise RuntimeError(f"disabled divergence: {invariance}")
    write_csv("disabled_invariance.csv", invariance)

    if not decision["teacher_coverage_pass"]:
        reason = decision["decision"]
        write_csv("per_gt_signal.csv", [{"not_run_reason": reason}])
        write_csv("gap_summary.csv", [{"not_run_reason": reason}])
        write_csv("bootstrap_95ci.csv", [{"not_run_reason": reason}])
    else:
        reason = "coverage passed; cross-modal gap stage pending"
        write_csv("per_gt_signal.csv", [{"not_run_reason": reason}])
        write_csv("gap_summary.csv", [{"not_run_reason": reason}])
        write_csv("bootstrap_95ci.csv", [{"not_run_reason": reason}])
    write_csv("mechanism_decision.csv", [{**decision, **pooled, **temporal_summary}])

    lines = ["# LiDAR-Privileged Target/Temporal Representation Signal Audit", "",
             "## Teacher coverage阶段判定", "", f"**{decision['decision']}**。", "",
             "Teacher：官方CenterPoint 0.1m voxel + SECOND/SECFPN + circle NMS。", "",
             "| Protocol | lost n | match | score | xy err | size rel err | yaw err | retained match |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for protocol in PROTOCOL_NAMES:
        lost = summary_index[(protocol, "lost")]
        retained = summary_index[(protocol, "retained")]
        lines.append(f"| {PROTOCOL_NAMES[protocol]} | {lost['n']} | "
                     f"{100*lost['teacher_match_rate']:.1f}% | "
                     f"{lost['median_teacher_score']:.3f} | "
                     f"{lost['median_xy_center_error']:.3f}m | "
                     f"{lost['median_relative_size_l1']:.3f} | "
                     f"{lost['median_yaw_error_deg']:.1f}° | "
                     f"{100*retained['teacher_match_rate']:.1f}% |")
    lines += ["", f"Pooled lost/retained match={100*pooled['lost_match_rate']:.1f}%/"
              f"{100*pooled['retained_match_rate']:.1f}%；lost score="
              f"{pooled['lost_median_score']:.3f}，xy error={pooled['lost_median_xy_error']:.3f}m。", "",
              "## Fault-episode teacher stability", "",
              f"Adjacent matched lost pairs={temporal_summary['lost_pair_count']}；median |delta score|="
              f"{temporal_summary['median_abs_score_delta']:.3f}，|delta center error|="
              f"{temporal_summary['median_abs_center_error_delta']:.3f}m，representation cosine distance="
              f"{temporal_summary['median_representation_cosine_distance']:.3f}。", "",
              "## Coverage gate", "",
              f"- coverage rate: {decision['coverage_rate_pass']}",
              f"- detection quality: {decision['teacher_quality_pass']}",
              f"- temporal stability: {decision['teacher_temporal_stability_pass']}", ""]
    if decision["teacher_coverage_pass"]:
        lines += ["Coverage前置门通过，允许继续运行预注册Camera/LiDAR gap阶段。"]
    else:
        lines += ["Teacher coverage/quality/stability前置门未全部通过；按预注册直接"
                  "No-Go，未运行Camera gap、distillation、overfit或smoke。"]
    lines += ["", "Clean/Dark/Blur/Crash B0-disabled各243个tensor leaves逐tensor exact，最大差0。",
              "未修改repos/StreamPETR。"]
    (REPORT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
