#!/usr/bin/env python3

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.utils import (
    category_to_detection_name,
)


ROOT = Path.home() / "research/evidence3d"

DATA_ROOT = ROOT / "data/nuscenes-mini"

RESULT_ROOT = (
    ROOT
    / "outputs/gt_recovery_predictions"
    / "official_r50_900q_baseline"
)

CLEAN_EXPERIMENT = "clean_no_corruption"

FAULT_EXPERIMENTS = [
    "camera_crash_back_5f",
    "camera_crash_back_10f",
    "compound_fog_crash_10f",
]

# 与nuScenes常用中心距离匹配方式一致。
CENTER_DISTANCE_THRESHOLD = 2.0

# 恢复条件：达到Clean检测目标的90%。
RECOVERY_THRESHOLD = 0.90

# 必须连续满足2帧，避免单帧波动。
CONSECUTIVE_FRAMES = 2

SUMMARY_PATH = (
    RESULT_ROOT
    / "gt_recovery_delay_summary.csv"
)

DETAIL_PATH = (
    RESULT_ROOT
    / "gt_recovery_frame_details.csv"
)


def load_results(experiment):
    path = (
        RESULT_ROOT
        / experiment
        / "results_nusc.json"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    results = payload.get("results", {})

    if not results:
        raise RuntimeError(
            f"{experiment}的results为空"
        )

    return results


def load_protocol(experiment):
    path = (
        RESULT_ROOT
        / experiment
        / "protocol_used.json"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def distance_xy(first, second):
    return math.hypot(
        first[0] - second[0],
        first[1] - second[1],
    )


def get_ground_truth(nusc, sample_token):
    sample = nusc.get(
        "sample",
        sample_token,
    )

    targets = []

    for annotation_token in sample["anns"]:
        annotation = nusc.get(
            "sample_annotation",
            annotation_token,
        )

        detection_name = (
            category_to_detection_name(
                annotation["category_name"]
            )
        )

        if detection_name is None:
            continue

        targets.append({
            "token": annotation_token,
            "name": detection_name,
            "xy": (
                float(annotation["translation"][0]),
                float(annotation["translation"][1]),
            ),
        })

    return targets


def get_predictions(results, sample_token):
    predictions = []

    for box in results.get(sample_token, []):
        translation = box.get(
            "translation",
            [0.0, 0.0, 0.0],
        )

        predictions.append({
            "name": box["detection_name"],
            "score": float(
                box.get("detection_score", 0.0)
            ),
            "xy": (
                float(translation[0]),
                float(translation[1]),
            ),
        })

    return predictions


def match_ground_truth(targets, predictions):
    """类别一致、中心距离阈值内的一对一贪心匹配。"""

    matched_target_tokens = set()

    targets_by_class = defaultdict(list)
    predictions_by_class = defaultdict(list)

    for target in targets:
        targets_by_class[target["name"]].append(
            target
        )

    for prediction in predictions:
        predictions_by_class[
            prediction["name"]
        ].append(prediction)

    for class_name, class_predictions in (
        predictions_by_class.items()
    ):
        class_targets = targets_by_class.get(
            class_name,
            [],
        )

        used_target_indexes = set()

        class_predictions = sorted(
            class_predictions,
            key=lambda item: item["score"],
            reverse=True,
        )

        for prediction in class_predictions:
            best_index = None
            best_distance = None

            for index, target in enumerate(
                class_targets
            ):
                if index in used_target_indexes:
                    continue

                current_distance = distance_xy(
                    prediction["xy"],
                    target["xy"],
                )

                if (
                    current_distance
                    > CENTER_DISTANCE_THRESHOLD
                ):
                    continue

                if (
                    best_distance is None
                    or current_distance < best_distance
                ):
                    best_index = index
                    best_distance = current_distance

            if best_index is not None:
                used_target_indexes.add(
                    best_index
                )

                matched_target_tokens.add(
                    class_targets[
                        best_index
                    ]["token"]
                )

    return matched_target_tokens


def build_scene_sample_index(nusc, valid_tokens):
    """建立sample token到scene和0起始帧序号的映射。"""

    token_metadata = {}
    scene_samples = defaultdict(list)

    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_name = scene["name"]

        sample_token = scene["first_sample_token"]
        frame_index = 0

        while sample_token:
            sample = nusc.get(
                "sample",
                sample_token,
            )

            if sample_token in valid_tokens:
                token_metadata[sample_token] = {
                    "scene_token": scene_token,
                    "scene_name": scene_name,
                    "frame_idx": frame_index,
                }

                scene_samples[scene_token].append(
                    (
                        frame_index,
                        sample_token,
                    )
                )

            sample_token = sample["next"]
            frame_index += 1

    for scene_token in scene_samples:
        scene_samples[scene_token].sort()

    return token_metadata, scene_samples


def get_scene_events(
    protocol,
    scene_token,
    scene_name,
):
    scenes = protocol.get("scenes", {})

    if scene_token in scenes:
        return scenes[scene_token]

    if scene_name in scenes:
        return scenes[scene_name]

    return scenes.get("*", [])


def find_recovery_frame(frame_rows):
    for index in range(
        len(frame_rows)
        - CONSECUTIVE_FRAMES
        + 1
    ):
        window = frame_rows[
            index:
            index + CONSECUTIVE_FRAMES
        ]

        frame_indexes = [
            row["frame_idx"]
            for row in window
        ]

        expected_indexes = list(
            range(
                frame_indexes[0],
                frame_indexes[0]
                + CONSECUTIVE_FRAMES,
            )
        )

        if frame_indexes != expected_indexes:
            continue

        if any(
            row["retention"] is None
            for row in window
        ):
            continue

        if all(
            row["retention"]
            >= RECOVERY_THRESHOLD
            for row in window
        ):
            return window[0]["frame_idx"]

    return None


def main():
    nusc = NuScenes(
        version="v1.0-mini",
        dataroot=str(DATA_ROOT),
        verbose=False,
    )

    clean_results = load_results(
        CLEAN_EXPERIMENT
    )

    valid_tokens = set(clean_results)

    token_metadata, scene_samples = (
        build_scene_sample_index(
            nusc,
            valid_tokens,
        )
    )

    if len(token_metadata) != len(valid_tokens):
        missing = (
            valid_tokens
            - set(token_metadata)
        )

        raise RuntimeError(
            "部分预测sample token无法映射到nuScenes："
            f"{list(missing)[:5]}"
        )

    # GT只读取一次。
    ground_truth_cache = {
        token: get_ground_truth(
            nusc,
            token,
        )
        for token in valid_tokens
    }

    # Clean正确匹配到的GT annotation token。
    clean_matches = {}

    for token in valid_tokens:
        clean_matches[token] = match_ground_truth(
            ground_truth_cache[token],
            get_predictions(
                clean_results,
                token,
            ),
        )

    summary_rows = []
    detail_rows = []

    for experiment in FAULT_EXPERIMENTS:
        fault_results = load_results(
            experiment
        )

        if set(fault_results) != valid_tokens:
            raise RuntimeError(
                f"{experiment}与Clean样本集合不一致"
            )

        protocol = load_protocol(
            experiment
        )

        fault_matches = {}

        for token in valid_tokens:
            fault_matches[token] = match_ground_truth(
                ground_truth_cache[token],
                get_predictions(
                    fault_results,
                    token,
                ),
            )

        for scene_token, samples in (
            scene_samples.items()
        ):
            scene_name = token_metadata[
                samples[0][1]
            ]["scene_name"]

            events = get_scene_events(
                protocol,
                scene_token,
                scene_name,
            )

            if not events:
                continue

            fault_start = min(
                int(event["start_frame"])
                for event in events
            )

            fault_end = max(
                int(event["end_frame"])
                for event in events
            )

            scene_frame_rows = []

            for frame_idx, sample_token in samples:
                clean_supported = (
                    clean_matches[sample_token]
                )

                fault_supported = (
                    fault_matches[sample_token]
                )

                clean_count = len(
                    clean_supported
                )

                retained_count = len(
                    clean_supported
                    & fault_supported
                )

                if clean_count > 0:
                    retention = (
                        retained_count
                        / clean_count
                    )
                else:
                    retention = None

                if frame_idx < fault_start:
                    phase = "pre_fault"
                elif frame_idx <= fault_end:
                    phase = "active_fault"
                else:
                    phase = "post_fault"

                row = {
                    "experiment": experiment,
                    "scene_token": scene_token,
                    "scene_name": scene_name,
                    "sample_token": sample_token,
                    "frame_idx": frame_idx,
                    "phase": phase,
                    "clean_matched_gt": clean_count,
                    "fault_matched_gt": len(
                        fault_supported
                    ),
                    "retained_clean_gt": retained_count,
                    "retention": retention,
                }

                detail_rows.append(row)

                if frame_idx > fault_end:
                    scene_frame_rows.append(row)

            recovered_frame = find_recovery_frame(
                scene_frame_rows
            )

            first_post_frame = fault_end + 1

            if recovered_frame is None:
                recovery_delay = None
            else:
                recovery_delay = (
                    recovered_frame
                    - first_post_frame
                )

            valid_post_rows = [
                row
                for row in scene_frame_rows
                if row["retention"] is not None
            ]

            clean_support_total = sum(
                row["clean_matched_gt"]
                for row in valid_post_rows
            )

            if valid_post_rows:
                mean_post_retention = sum(
                    row["retention"]
                    for row in valid_post_rows
                ) / len(valid_post_rows)
            else:
                mean_post_retention = None

            if len(valid_post_rows) < 2:
                status = (
                    "insufficient_clean_support"
                )
            elif recovered_frame is None:
                status = "not_recovered"
            else:
                status = "recovered"

            summary_rows.append({
                "experiment": experiment,
                "scene_token": scene_token,
                "scene_name": scene_name,
                "fault_start_frame": fault_start,
                "fault_end_frame": fault_end,
                "first_post_frame": first_post_frame,
                "recovered_frame": recovered_frame,
                "recovery_delay_frames": recovery_delay,
                "valid_post_frames": len(
                    valid_post_rows
                ),
                "clean_supported_gt_total": (
                    clean_support_total
                ),
                "mean_post_retention": (
                    mean_post_retention
                ),
                "distance_threshold_m": (
                    CENTER_DISTANCE_THRESHOLD
                ),
                "recovery_threshold": (
                    RECOVERY_THRESHOLD
                ),
                "consecutive_frames": (
                    CONSECUTIVE_FRAMES
                ),
                "status": status,
            })

    summary_fields = [
        "experiment",
        "scene_token",
        "scene_name",
        "fault_start_frame",
        "fault_end_frame",
        "first_post_frame",
        "recovered_frame",
        "recovery_delay_frames",
        "valid_post_frames",
        "clean_supported_gt_total",
        "mean_post_retention",
        "distance_threshold_m",
        "recovery_threshold",
        "consecutive_frames",
        "status",
    ]

    detail_fields = [
        "experiment",
        "scene_token",
        "scene_name",
        "sample_token",
        "frame_idx",
        "phase",
        "clean_matched_gt",
        "fault_matched_gt",
        "retained_clean_gt",
        "retention",
    ]

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=summary_fields,
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    with DETAIL_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=detail_fields,
        )

        writer.writeheader()
        writer.writerows(detail_rows)

    print(
        f"{'实验':31s}"
        f"{'场景':16s}"
        f"{'结束':>6s}"
        f"{'恢复帧':>9s}"
        f"{'延迟':>8s}"
        f"{'有效帧':>9s}"
        f"{'GT支持':>9s}"
        f"{'状态':>28s}"
    )

    print("-" * 116)

    for row in summary_rows:
        recovered = (
            "-"
            if row["recovered_frame"] is None
            else str(row["recovered_frame"])
        )

        delay = (
            "-"
            if row["recovery_delay_frames"] is None
            else str(
                row["recovery_delay_frames"]
            )
        )

        print(
            f"{row['experiment'][:31]:31s}"
            f"{row['scene_name'][:15]:16s}"
            f"{row['fault_end_frame']:6d}"
            f"{recovered:>9s}"
            f"{delay:>8s}"
            f"{row['valid_post_frames']:9d}"
            f"{row['clean_supported_gt_total']:9d}"
            f"{row['status']:>28s}"
        )

    print()
    print("中心距离阈值：", CENTER_DISTANCE_THRESHOLD, "m")
    print("恢复阈值：", RECOVERY_THRESHOLD)
    print("连续帧要求：", CONSECUTIVE_FRAMES)
    print("汇总文件：", SUMMARY_PATH)
    print("逐帧文件：", DETAIL_PATH)


if __name__ == "__main__":
    main()
