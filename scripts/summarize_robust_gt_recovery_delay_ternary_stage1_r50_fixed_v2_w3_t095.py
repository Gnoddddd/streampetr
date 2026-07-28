#!/usr/bin/env python3

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path.home() / "research/evidence3d"

RESULT_ROOT = (
    ROOT
    / "outputs/gt_recovery_predictions"
    / "stage1_active_r50_900q"
)

RAW_SUMMARY_PATH = (
    RESULT_ROOT
    / "gt_recovery_delay_fixed_v2_summary.csv"
)

DETAIL_PATH = (
    RESULT_ROOT
    / "gt_recovery_fixed_v2_frame_details.csv"
)

OUTPUT_PATH = (
    RESULT_ROOT
    / "gt_recovery_delay_robust_fixed_v2_w3_t095_summary.csv"
)

WINDOW_OUTPUT_PATH = (
    RESULT_ROOT
    / "gt_recovery_delay_robust_fixed_v2_w3_t095_windows.csv"
)

RECOVERY_THRESHOLD = 0.95
WINDOW_FRAMES = 3
MIN_CLEAN_SUPPORT = 5


def parse_optional_int(value):
    value = str(value or "").strip()

    if not value:
        return None

    return int(value)


with RAW_SUMMARY_PATH.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as file:
    raw_rows = list(csv.DictReader(file))

with DETAIL_PATH.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as file:
    detail_rows = list(csv.DictReader(file))


details_by_case = defaultdict(list)

for row in detail_rows:
    key = (
        row["experiment"],
        row["scene_token"],
    )

    details_by_case[key].append(row)


summary_rows = []
window_rows = []

for raw in raw_rows:
    experiment = raw["experiment"]
    scene_token = raw["scene_token"]
    scene_name = raw["scene_name"]

    fault_end = int(
        raw["fault_end_frame"]
    )

    first_post_frame = fault_end + 1

    key = (
        experiment,
        scene_token,
    )

    case_rows = details_by_case.get(
        key,
        [],
    )

    frame_map = {}

    for row in case_rows:
        frame_idx = int(row["frame_idx"])

        if frame_idx <= fault_end:
            continue

        frame_map[frame_idx] = {
            "clean": int(
                row["clean_matched_gt"]
            ),
            "retained": int(
                row["retained_clean_gt"]
            ),
        }

    if not frame_map:
        summary_rows.append({
            "experiment": experiment,
            "scene_token": scene_token,
            "scene_name": scene_name,
            "fault_end_frame": fault_end,
            "first_post_frame": first_post_frame,
            "raw_recovery_delay": (
                raw["recovery_delay_frames"]
            ),
            "robust_recovered_frame": "",
            "robust_recovery_delay": "",
            "recovery_window_clean_support": 0,
            "recovery_window_retained": 0,
            "recovery_window_retention": "",
            "zero_support_frames_before_recovery": "",
            "window_frames": WINDOW_FRAMES,
            "minimum_clean_support": (
                MIN_CLEAN_SUPPORT
            ),
            "recovery_threshold": (
                RECOVERY_THRESHOLD
            ),
            "status": "no_post_fault_frames",
        })

        continue

    max_frame = max(frame_map)

    recovered_frame = None
    recovery_support = 0
    recovery_retained = 0
    recovery_ratio = None

    for start_frame in range(
        first_post_frame,
        max_frame - WINDOW_FRAMES + 2,
    ):
        expected_frames = [
            start_frame + offset
            for offset in range(
                WINDOW_FRAMES
            )
        ]

        if any(
            frame not in frame_map
            for frame in expected_frames
        ):
            continue

        clean_support = sum(
            frame_map[frame]["clean"]
            for frame in expected_frames
        )

        retained = sum(
            frame_map[frame]["retained"]
            for frame in expected_frames
        )

        retention = (
            retained / clean_support
            if clean_support > 0
            else None
        )

        qualifies = (
            clean_support
            >= MIN_CLEAN_SUPPORT
            and retention is not None
            and retention
            >= RECOVERY_THRESHOLD
        )

        window_rows.append({
            "experiment": experiment,
            "scene_token": scene_token,
            "scene_name": scene_name,
            "window_start_frame": (
                start_frame
            ),
            "window_end_frame": (
                expected_frames[-1]
            ),
            "clean_support": clean_support,
            "retained": retained,
            "retention": (
                retention
                if retention is not None
                else ""
            ),
            "qualifies": qualifies,
        })

        if qualifies:
            recovered_frame = start_frame
            recovery_support = clean_support
            recovery_retained = retained
            recovery_ratio = retention
            break

    if recovered_frame is None:
        robust_delay = None

        total_post_support = sum(
            item["clean"]
            for item in frame_map.values()
        )

        if total_post_support < MIN_CLEAN_SUPPORT:
            status = (
                "insufficient_clean_support"
            )
        else:
            status = "not_recovered"

        zero_support_count = sum(
            1
            for frame, item in frame_map.items()
            if frame >= first_post_frame
            and item["clean"] == 0
        )
    else:
        robust_delay = (
            recovered_frame
            - first_post_frame
        )

        zero_support_count = sum(
            1
            for frame in range(
                first_post_frame,
                recovered_frame + 1,
            )
            if (
                frame in frame_map
                and frame_map[frame]["clean"] == 0
            )
        )

        status = "recovered"

    summary_rows.append({
        "experiment": experiment,
        "scene_token": scene_token,
        "scene_name": scene_name,
        "fault_end_frame": fault_end,
        "first_post_frame": first_post_frame,
        "raw_recovery_delay": (
            parse_optional_int(
                raw["recovery_delay_frames"]
            )
            if raw["recovery_delay_frames"]
            else ""
        ),
        "robust_recovered_frame": (
            recovered_frame
            if recovered_frame is not None
            else ""
        ),
        "robust_recovery_delay": (
            robust_delay
            if robust_delay is not None
            else ""
        ),
        "recovery_window_clean_support": (
            recovery_support
        ),
        "recovery_window_retained": (
            recovery_retained
        ),
        "recovery_window_retention": (
            recovery_ratio
            if recovery_ratio is not None
            else ""
        ),
        "zero_support_frames_before_recovery": (
            zero_support_count
        ),
        "window_frames": WINDOW_FRAMES,
        "minimum_clean_support": (
            MIN_CLEAN_SUPPORT
        ),
        "recovery_threshold": (
            RECOVERY_THRESHOLD
        ),
        "status": status,
    })


summary_fields = list(
    summary_rows[0].keys()
)

with OUTPUT_PATH.open(
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


window_fields = [
    "experiment",
    "scene_token",
    "scene_name",
    "window_start_frame",
    "window_end_frame",
    "clean_support",
    "retained",
    "retention",
    "qualifies",
]

with WINDOW_OUTPUT_PATH.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=window_fields,
    )

    writer.writeheader()
    writer.writerows(window_rows)


print(
    f"{'实验':30s}"
    f"{'场景':13s}"
    f"{'原延迟':>9s}"
    f"{'稳健帧':>9s}"
    f"{'稳健延迟':>10s}"
    f"{'支持GT':>9s}"
    f"{'保留GT':>9s}"
    f"{'保留率':>10s}"
    f"{'零支持':>9s}"
    f"{'状态':>14s}"
)

print("-" * 131)

for row in summary_rows:
    ratio = row[
        "recovery_window_retention"
    ]

    ratio_text = (
        f"{ratio:.4f}"
        if isinstance(ratio, float)
        else "-"
    )

    print(
        f"{row['experiment'][:30]:30s}"
        f"{row['scene_name'][:12]:13s}"
        f"{str(row['raw_recovery_delay']):>9s}"
        f"{str(row['robust_recovered_frame']):>9s}"
        f"{str(row['robust_recovery_delay']):>10s}"
        f"{row['recovery_window_clean_support']:9d}"
        f"{row['recovery_window_retained']:9d}"
        f"{ratio_text:>10s}"
        f"{str(row['zero_support_frames_before_recovery']):>9s}"
        f"{row['status']:>14s}"
    )

print()
print("窗口长度：", WINDOW_FRAMES)
print("最小Clean支持GT：", MIN_CLEAN_SUPPORT)
print("恢复阈值：", RECOVERY_THRESHOLD)
print("汇总文件：", OUTPUT_PATH)
print("窗口明细：", WINDOW_OUTPUT_PATH)
