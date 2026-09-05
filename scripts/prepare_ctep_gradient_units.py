#!/usr/bin/env python3
"""Freeze deterministic P1 gradient units after the preregistered P0 GO."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/full_nuscenes/ctep_method_activation"
PROTOCOLS = ("blur_back", "crash_back", "dark_back")
SEED_TEXT = "ctep-gradient-units-v1-20260902"


def order(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["selection_key"] = frame.unit_id.map(
        lambda value: hashlib.sha256((SEED_TEXT + label + value).encode()).hexdigest()
    )
    return frame.sort_values(["selection_key", "unit_id"])


def main() -> None:
    decision = json.loads((REPORT / "p0_decision.json").read_text())
    if decision["verdict"] != "P0_GO_P1_REQUIRED":
        raise RuntimeError(f"P1 is locked by {decision['verdict']}")
    output = REPORT / "gradient_units.csv"
    if output.exists():
        raise RuntimeError("gradient_units.csv exists; refusing to redefine P1 units")
    data = pd.read_csv(REPORT / "per_gt_p0.csv")
    scene_order = pd.read_csv(REPORT / "scene_list.csv").scene_token.tolist()
    chosen = []
    coverage = []
    for protocol in PROTOCOLS:
        protocol_frame = data[(data.protocol == protocol) & data.ctep_active].copy()
        eligible_scenes = []
        for scene in scene_order:
            frame = protocol_frame[protocol_frame.scene_token == scene]
            if frame.history_sensitive_lost.sum() >= 2 and frame.easy.sum() >= 2:
                eligible_scenes.append(scene)
        selected_scenes = eligible_scenes[:4]
        if len(selected_scenes) != 4:
            raise RuntimeError(f"{protocol}: fewer than four gradient scenes")
        protocol_rows = []
        for scene in selected_scenes:
            frame = protocol_frame[protocol_frame.scene_token == scene]
            history = order(frame[frame.history_sensitive_lost], f"{protocol}:history")
            history = history.head(2)
            easy = order(
                frame[frame.easy & ~frame.unit_id.isin(history.unit_id)],
                f"{protocol}:easy",
            ).head(2)
            if len(history) != 2 or len(easy) != 2:
                raise RuntimeError(f"{protocol}/{scene}: stratum quota unavailable")
            history = history.assign(gradient_stratum="history_sensitive_lost")
            easy = easy.assign(gradient_stratum="easy")
            protocol_rows.extend([history, easy])
        protocol_output = pd.concat(protocol_rows, ignore_index=True)
        protocol_output["active_terms"] = protocol_output.apply(
            lambda row: json.dumps(
                [term for term, active in (("AC", row.active_AC), ("BD", row.active_BD))
                 if bool(active)]
            ), axis=1,
        )
        term_n = sum(len(json.loads(value)) for value in protocol_output.active_terms)
        if term_n < 16:
            raise RuntimeError(f"{protocol}: active term coverage {term_n} < 16")
        coverage.append({
            "protocol": protocol,
            "events": len(protocol_output),
            "active_terms": term_n,
            "scenes": protocol_output.scene_token.nunique(),
            "history_sensitive_lost_events": int(protocol_output.history_sensitive_lost.sum()),
            "easy_events": int(protocol_output.easy.sum()),
        })
        chosen.append(protocol_output)
    columns = [
        "unit_id", "protocol", "scene_token", "sample_token", "frame_idx",
        "gt_token", "instance_token", "gt_class", "gradient_stratum",
        "active_terms", "A_qplus", "B_qplus", "C_qplus", "D_qplus",
        "A_s_pos", "B_s_pos", "C_s_pos", "D_s_pos", "L_AC", "L_BD",
        "L_CTEP", "selection_key",
    ]
    result = pd.concat(chosen, ignore_index=True)[columns].sort_values(
        ["protocol", "scene_token", "frame_idx", "unit_id"]
    )
    result.to_csv(output, index=False)
    pd.DataFrame(coverage).to_csv(REPORT / "gradient_unit_coverage.csv", index=False)
    print(pd.DataFrame(coverage).to_string(index=False))


if __name__ == "__main__":
    main()
