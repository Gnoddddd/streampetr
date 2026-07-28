#!/usr/bin/env python3

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path.home() / "research/evidence3d"

TRACE_ROOT = (
    ROOT
    / "outputs/protocol_evaluations"
    / "full_candidate"
)

GT_PATH = (
    ROOT
    / "outputs/gt_recovery_predictions"
    / "source_aware_ft400_fp32"
    / "gt_recovery_delay_robust_summary.csv"
)

OUTPUT_PATH = (
    ROOT
    / "outputs/gt_recovery_predictions"
    / "source_aware_ft400_fp32"
    / "policy_gt_recovery_comparison.csv"
)

CASES = {
    "camera_crash_back_5f": {
        "trace_experiment": (
            "presets__camera_crash_back_5f"
        ),
        "protocol": (
            ROOT
            / "protocols/presets"
            / "camera_crash_back_5f.json"
        ),
    },
    "camera_crash_back_10f": {
        "trace_experiment": (
            "presets__camera_crash_back_10f"
        ),
        "protocol": (
            ROOT
            / "protocols/presets"
            / "camera_crash_back_10f.json"
        ),
    },
    "compound_fog_crash_10f": {
        "trace_experiment": (
            "presets__compound_fog_crash_10f"
        ),
        "protocol": (
            ROOT
            / "protocols/presets"
            / "compound_fog_crash_10f.json"
        ),
    },
}

KEEP_THRESHOLD = 0.98
RECOVER_THRESHOLD = 0.005
CONSECUTIVE_FRAMES = 2


def flatten(value):
    if isinstance(value, list):
        output = []

        for item in value:
            output.extend(flatten(item))

        return output

    return [value]


def optional_int(value):
    value = str(value or "").strip()

    if not value:
        return None

    return int(value)


if not GT_PATH.is_file():
    raise FileNotFoundError(GT_PATH)


with GT_PATH.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as file:
    gt_rows = list(csv.DictReader(file))


# 同时支持轨迹记录scene token或scene name。
gt_index = {}

for row in gt_rows:
    experiment = row["experiment"]

    gt_index[
        (
            experiment,
            row["scene_token"],
        )
    ] = row

    gt_index[
        (
            experiment,
            row["scene_name"],
        )
    ] = row


output_rows = []
matched_gt_keys = set()


for experiment, case in CASES.items():
    trace_directory = (
        TRACE_ROOT
        / case["trace_experiment"]
        / "traces"
    )

    protocol_path = case["protocol"]

    if not protocol_path.is_file():
        raise FileNotFoundError(
            protocol_path
        )

    trace_paths = sorted(
        trace_directory.glob("*.jsonl")
    )

    if not trace_paths:
        raise RuntimeError(
            f"没有找到轨迹：{trace_directory}"
        )

    protocol = json.loads(
        protocol_path.read_text(
            encoding="utf-8"
        )
    )

    events = protocol["scenes"]["*"]

    fault_end = max(
        int(event["end_frame"])
        for event in events
    )

    first_post_frame = fault_end + 1

    frame_counts = defaultdict(
        lambda: {
            "keep": 0,
            "recover": 0,
            "defer": 0,
            "total": 0,
        }
    )

    for trace_path in trace_paths:
        with trace_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line)

                scene_id = (
                    record.get("scene_token")
                    or record.get("scene_name")
                )

                if not scene_id:
                    raise RuntimeError(
                        f"{trace_path}中的记录"
                        "没有scene_token或scene_name"
                    )

                frame_idx = int(
                    record["frame_idx"]
                )

                diagnostics = record[
                    "diagnostics"
                ]

                prior = flatten(
                    diagnostics["prior_strength"]
                )

                actions = flatten(
                    diagnostics["action"]
                )

                if len(prior) != len(actions):
                    raise RuntimeError(
                        "prior_strength和action"
                        "数量不一致"
                    )

                key = (
                    str(scene_id),
                    frame_idx,
                )

                for prior_value, action in zip(
                    prior,
                    actions,
                ):
                    if float(prior_value) <= 1e-6:
                        continue

                    action = int(action)
                    frame_counts[key]["total"] += 1

                    if action == 0:
                        frame_counts[key]["keep"] += 1
                    elif action == 1:
                        frame_counts[key]["recover"] += 1
                    elif action == 2:
                        frame_counts[key]["defer"] += 1
                    else:
                        raise RuntimeError(
                            f"未知action：{action}"
                        )

    scene_ids = sorted({
        scene_id
        for scene_id, _ in frame_counts
    })

    for scene_id in scene_ids:
        gt_row = gt_index.get(
            (
                experiment,
                scene_id,
            )
        )

        if gt_row is None:
            print(
                "[警告] 找不到对应GT行：",
                experiment,
                scene_id,
            )
            continue

        matched_gt_keys.add(
            (
                experiment,
                gt_row["scene_token"],
            )
        )

        frame_rows = []

        for (
            current_scene,
            frame_idx,
        ), values in frame_counts.items():
            if current_scene != scene_id:
                continue

            if frame_idx <= fault_end:
                continue

            total = values["total"]

            if total <= 0:
                continue

            frame_rows.append({
                "frame_idx": frame_idx,
                "keep_ratio": (
                    values["keep"] / total
                ),
                "recover_ratio": (
                    values["recover"] / total
                ),
                "defer_ratio": (
                    values["defer"] / total
                ),
            })

        frame_rows.sort(
            key=lambda row: row["frame_idx"]
        )

        frame_map = {
            row["frame_idx"]: row
            for row in frame_rows
        }

        recovered_frame = None

        if frame_map:
            max_frame = max(frame_map)

            for start_frame in range(
                first_post_frame,
                max_frame
                - CONSECUTIVE_FRAMES
                + 2,
            ):
                expected_frames = [
                    start_frame + offset
                    for offset in range(
                        CONSECUTIVE_FRAMES
                    )
                ]

                if any(
                    frame not in frame_map
                    for frame in expected_frames
                ):
                    continue

                stable = all(
                    frame_map[frame][
                        "keep_ratio"
                    ] >= KEEP_THRESHOLD
                    and frame_map[frame][
                        "recover_ratio"
                    ] <= RECOVER_THRESHOLD
                    for frame in expected_frames
                )

                if stable:
                    recovered_frame = (
                        start_frame
                    )
                    break

        policy_delay = (
            recovered_frame
            - first_post_frame
            if recovered_frame is not None
            else None
        )

        gt_delay = optional_int(
            gt_row[
                "robust_recovery_delay"
            ]
        )

        gt_recovered_frame = optional_int(
            gt_row[
                "robust_recovered_frame"
            ]
        )

        if (
            policy_delay is None
            or gt_delay is None
        ):
            difference = None
            relation = "not_comparable"
        else:
            difference = (
                policy_delay - gt_delay
            )

            if difference == 0:
                relation = "aligned"
            elif difference < 0:
                relation = "policy_earlier"
            else:
                relation = "policy_later"

        output_rows.append({
            "experiment": experiment,
            "scene_token": (
                gt_row["scene_token"]
            ),
            "scene_name": (
                gt_row["scene_name"]
            ),
            "fault_end_frame": fault_end,
            "first_post_frame": (
                first_post_frame
            ),
            "policy_recovered_frame": (
                ""
                if recovered_frame is None
                else recovered_frame
            ),
            "policy_recovery_delay": (
                ""
                if policy_delay is None
                else policy_delay
            ),
            "gt_recovered_frame": (
                ""
                if gt_recovered_frame is None
                else gt_recovered_frame
            ),
            "gt_recovery_delay": (
                ""
                if gt_delay is None
                else gt_delay
            ),
            "policy_minus_gt_delay": (
                ""
                if difference is None
                else difference
            ),
            "relation": relation,
            "keep_threshold": (
                KEEP_THRESHOLD
            ),
            "recover_threshold": (
                RECOVER_THRESHOLD
            ),
            "consecutive_frames": (
                CONSECUTIVE_FRAMES
            ),
        })


expected_gt_keys = {
    (
        row["experiment"],
        row["scene_token"],
    )
    for row in gt_rows
}

missing = expected_gt_keys - matched_gt_keys

if missing:
    print()
    print("[警告] 以下GT结果未匹配到策略轨迹：")

    for key in sorted(missing):
        print(" ", key)


if not output_rows:
    raise RuntimeError(
        "没有生成任何策略—GT对比结果"
    )


fields = list(output_rows[0].keys())

with OUTPUT_PATH.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=fields,
    )

    writer.writeheader()
    writer.writerows(output_rows)


print()
print(
    f"{'实验':29s}"
    f"{'场景':13s}"
    f"{'策略帧':>9s}"
    f"{'策略延迟':>10s}"
    f"{'GT帧':>8s}"
    f"{'GT延迟':>9s}"
    f"{'差值':>8s}"
    f"{'关系':>22s}"
)

print("-" * 110)

for row in output_rows:
    print(
        f"{row['experiment'][:29]:29s}"
        f"{row['scene_name'][:12]:13s}"
        f"{str(row['policy_recovered_frame']):>9s}"
        f"{str(row['policy_recovery_delay']):>10s}"
        f"{str(row['gt_recovered_frame']):>8s}"
        f"{str(row['gt_recovery_delay']):>9s}"
        f"{str(row['policy_minus_gt_delay']):>8s}"
        f"{row['relation']:>22s}"
    )

print()
print("差值=策略恢复延迟-GT稳健恢复延迟")
print("负值：策略早于GT恢复，可能过度自信")
print("正值：策略晚于GT恢复，策略较保守")
print("零值：策略与GT恢复同步")
print("结果文件：", OUTPUT_PATH)
