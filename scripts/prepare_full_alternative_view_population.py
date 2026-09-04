#!/usr/bin/env python3
"""Freeze the full-nuScenes alternative-view audit population before forwards."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/full_nuscenes/alternative_view_causal_audit"
SOURCE = ROOT / "reports/full_nuscenes/mechanism_confirmation/root_cause/per_gt.csv"
PREREG = REPORT / "PRE_REGISTRATION.md"
ELIGIBLE = REPORT / "population_eligible.csv"
FORWARD = REPORT / "population_forward_units.csv"
COVERAGE = REPORT / "population_coverage.csv"
MANIFEST = REPORT / "population_manifest.json"
PROTOCOLS = ("blur_back", "crash_back", "dark_back")
SCHEMA = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty population file: {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def unit_id(row: dict) -> str:
    return f"{row['protocol']}:{row['sample_token']}:{row['gt_token']}"


def match_cost(left: dict, right: dict, right_index: int) -> float:
    left_score = float(left["clean_s_pos"])
    right_score = float(right["clean_s_pos"])
    left_missing = not math.isfinite(left_score)
    right_missing = not math.isfinite(right_score)
    if left_missing:
        left_score = 0.0
    if right_missing:
        right_score = 0.0
    return float(
        100.0 * abs(int(left["frame_idx"]) - int(right["frame_idx"]))
        + 20.0 * (left["gt_class"] != right["gt_class"])
        + abs(float(left["distance_m"]) - float(right["distance_m"])) / 20.0
        + abs(int(left["alternative_view_count"])
              - int(right["alternative_view_count"]))
        + abs(int(left["visibility_token"]) - int(right["visibility_token"])) / 4.0
        + 5.0 * (left_missing != right_missing)
        + abs(left_score - right_score)
        + abs(math.log(float(left["max_projected_area_fraction"]) + 1e-6)
              - math.log(float(right["max_projected_area_fraction"]) + 1e-6))
        + right_index * 1e-12
    )


def freeze_population(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    eligible = [dict(row) for row in rows
                if row["protocol"] in PROTOCOLS
                and int(row["alternative_view_count"]) >= 1]
    eligible.sort(key=lambda row: (
        PROTOCOLS.index(row["protocol"]), row["scene_token"], int(row["frame_idx"]),
        row["sample_token"], row["gt_token"], row["outcome"],
    ))
    for index, row in enumerate(eligible):
        row["eligible_index"] = index
        row["unit_id"] = unit_id(row)
        row["selected_for_forward"] = False
        row["forward_role"] = ""
        row["matched_unit_id"] = ""
        row["match_cost"] = ""

    by_scene = defaultdict(list)
    for row in eligible:
        by_scene[(row["protocol"], row["scene_token"])].append(row)
    forward = []
    for (protocol, scene), scene_rows in sorted(
        by_scene.items(), key=lambda item: (
            PROTOCOLS.index(item[0][0]), item[0][1]
        )
    ):
        lost = sorted(
            [row for row in scene_rows if row["outcome"] == "fault_induced_lost"],
            key=lambda row: (int(row["frame_idx"]), row["sample_token"], row["gt_token"]),
        )
        retained = sorted(
            [row for row in scene_rows if row["outcome"] == "retained"],
            key=lambda row: (int(row["frame_idx"]), row["sample_token"], row["gt_token"]),
        )
        if not lost:
            continue
        if len(retained) < len(lost):
            raise RuntimeError(
                f"insufficient retained controls in {protocol}/{scene}: "
                f"lost={len(lost)} retained={len(retained)}"
            )
        costs = np.empty((len(lost), len(retained)), dtype=float)
        for i, left in enumerate(lost):
            for j, right in enumerate(retained):
                costs[i, j] = match_cost(left, right, j)
        left_indexes, right_indexes = linear_sum_assignment(costs)
        if len(left_indexes) != len(lost):
            raise RuntimeError(f"incomplete matching in {protocol}/{scene}")
        for i, j in sorted(zip(left_indexes, right_indexes)):
            left, right = lost[int(i)], retained[int(j)]
            cost = float(costs[int(i), int(j)])
            left["selected_for_forward"] = True
            left["forward_role"] = "lost_primary"
            left["matched_unit_id"] = right["unit_id"]
            left["match_cost"] = cost
            right["selected_for_forward"] = True
            right["forward_role"] = "retained_matched_control"
            right["matched_unit_id"] = left["unit_id"]
            right["match_cost"] = cost
            forward.extend((dict(left), dict(right)))

    forward.sort(key=lambda row: (
        PROTOCOLS.index(row["protocol"]), row["scene_token"], int(row["frame_idx"]),
        row["sample_token"], row["gt_token"], row["forward_role"],
    ))
    if len({row["unit_id"] for row in forward}) != len(forward):
        raise RuntimeError("forward population contains duplicate units")

    coverage = []
    for protocol in PROTOCOLS:
        protocol_eligible = [row for row in eligible if row["protocol"] == protocol]
        protocol_forward = [row for row in forward if row["protocol"] == protocol]
        for scope, selected in (("eligible", protocol_eligible),
                                ("forward", protocol_forward)):
            counts = Counter(row["outcome"] for row in selected)
            coverage.append({
                "protocol": protocol,
                "scope": scope,
                "events": len(selected),
                "lost": counts["fault_induced_lost"],
                "retained": counts["retained"],
                "scenes": len({row["scene_token"] for row in selected}),
                "trajectories": len({(row["scene_token"], row["instance_token"])
                                      for row in selected}),
                "cam_back_visible": sum(str(row["cam_back_visible"]).lower() == "true"
                                        for row in selected),
                "fault_candidate_available": sum(
                    str(row["fault_candidate_available"]).lower() == "true"
                    for row in selected
                ),
            })
    return eligible, forward, coverage


def initial_progress(population_hash: str, coverage: list[dict]) -> None:
    commands = [
        "python scripts/run_full_alternative_view_causal_audit.py --phase p0_p1 --protocol blur_back",
        "python scripts/run_full_alternative_view_causal_audit.py --phase p0_p1 --protocol crash_back",
        "python scripts/run_full_alternative_view_causal_audit.py --phase p0_p1 --protocol dark_back",
        "python scripts/run_full_alternative_view_causal_audit.py --phase p2_history --protocol blur_back",
        "python scripts/analyze_full_alternative_view_causal_audit.py",
    ]
    progress = {
        "schema_version": SCHEMA,
        "population_sha256": population_hash,
        "status": "population_frozen_no_forward_started",
        "completed": {"p0_p1": {p: [] for p in PROTOCOLS},
                      "p2_history": {p: [] for p in PROTOCOLS}},
        "resume_commands": commands,
    }
    (REPORT / "progress_manifest.json").write_text(
        json.dumps(progress, indent=2) + "\n", encoding="utf-8"
    )
    forward_coverage = [row for row in coverage if row["scope"] == "forward"]
    lines = [
        "# PARTIAL STATUS",
        "",
        "`PARTIAL_INSUFFICIENT_COVERAGE`: population 已冻结，尚未运行任何 forward。",
        "",
        "## Frozen forward population",
        "",
        "| Protocol | lost | retained | scenes |",
        "|---|---:|---:|---:|",
    ]
    for row in forward_coverage:
        lines.append(
            f"| {row['protocol']} | {row['lost']} | {row['retained']} | {row['scenes']} |"
        )
    lines += ["", "## Resume", "", "```bash", *commands, "```", ""]
    (REPORT / "PARTIAL_STATUS.md").write_text("\n".join(lines), encoding="utf-8")


def validate_existing(source_hash: str) -> bool:
    if not MANIFEST.is_file():
        return False
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if value.get("source_sha256") != source_hash:
        raise RuntimeError("source population changed after freeze")
    for key, path in (("eligible_sha256", ELIGIBLE), ("forward_sha256", FORWARD),
                      ("coverage_sha256", COVERAGE)):
        if not path.is_file() or sha256(path) != value.get(key):
            raise RuntimeError(f"frozen population artifact mismatch: {path}")
    print(json.dumps(value, indent=2))
    return True


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    if not PREREG.is_file():
        raise RuntimeError("PRE_REGISTRATION.md must exist before population freeze")
    source_hash = sha256(SOURCE)
    if validate_existing(source_hash):
        return
    if any(path.exists() for path in (ELIGIBLE, FORWARD, COVERAGE, MANIFEST)):
        raise RuntimeError("partial population artifacts exist without a valid manifest")
    eligible, forward, coverage = freeze_population(read_csv(SOURCE))
    write_csv(ELIGIBLE, eligible)
    write_csv(FORWARD, forward)
    write_csv(COVERAGE, coverage)
    manifest = {
        "schema_version": SCHEMA,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": source_hash,
        "preregistration_sha256": sha256(PREREG),
        "eligible_sha256": sha256(ELIGIBLE),
        "forward_sha256": sha256(FORWARD),
        "coverage_sha256": sha256(COVERAGE),
        "eligible_events": len(eligible),
        "forward_events": len(forward),
        "protocol_order": list(PROTOCOLS),
        "matching": "same_protocol_same_scene_hungarian_fixed_preregistered_cost",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    initial_progress(manifest["forward_sha256"], coverage)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
