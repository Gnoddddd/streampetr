#!/usr/bin/env python3
"""Generate the frozen FEQ train-only episode and history manifests."""

import argparse
import csv
import json
from pathlib import Path

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes

CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT",
)


def generate(info_file, data_root, output, history_output, csv_output, seed=314159):
    payload = mmcv.load(info_file)
    infos = payload["infos"] if isinstance(payload, dict) else payload
    by_scene = {}
    for info in infos:
        by_scene.setdefault(info["scene_token"], []).append(info)
    for values in by_scene.values():
        values.sort(key=lambda x: x["frame_idx"])

    rng = np.random.default_rng(seed)
    schedule = {"version": 1, "scenes": {}}
    rows = []
    event_types = ("single_camera", "adjacent_double", "frame_lost")
    durations = (1, 3, 5)
    event_id = 0
    # Deterministic non-overlapping episodes, targeting exactly half the frames.
    scene_items = sorted(by_scene.items())
    targets = {scene: len(frames) // 2 for scene, frames in scene_items}
    remainder = len(infos) // 2 - sum(targets.values())
    for scene, _frames in scene_items[:remainder]:
        targets[scene] += 1
    for scene, frames in scene_items:
        target = targets[scene]
        active = set()
        candidates = list(range(1, max(len(frames) - 1, 1)))
        rng.shuffle(candidates)
        events = []
        for start in candidates:
            if len(active) >= target:
                break
            duration = durations[event_id % len(durations)]
            chosen = [i for i in range(start, min(start + duration, len(frames)))
                      if i not in active]
            if not chosen or len(active) + len(chosen) > target:
                continue
            kind = event_types[event_id % len(event_types)]
            camera = int(rng.integers(0, len(CAMERAS)))
            event = {"start_frame": int(chosen[0]), "end_frame": int(chosen[-1])}
            if kind == "single_camera":
                event["failed_cameras"] = [CAMERAS[camera]]
            elif kind == "adjacent_double":
                event["failed_cameras"] = [CAMERAS[camera], CAMERAS[(camera + 1) % 6]]
            else:
                event["lost_cameras"] = [CAMERAS[camera]]
            events.append(event)
            active.update(chosen)
            for index in chosen:
                rows.append({
                    "event_id": event_id, "scene_token": scene,
                    "sample_token": frames[index]["token"], "frame_idx": index,
                    "event_type": kind, "duration": duration,
                    "failed_cameras": ";".join(event.get("failed_cameras", [])),
                    "lost_cameras": ";".join(event.get("lost_cameras", [])),
                    "manifest_seed": seed,
                })
            event_id += 1
        schedule["scenes"][scene] = events

    # True instance lineage is used only to label reliable train history.
    nusc = NuScenes(version="v1.0-mini", dataroot=str(data_root), verbose=False)
    history = {"version": 1, "samples": {}}
    for info in infos:
        sample = nusc.get("sample", info["token"])
        if len(sample["anns"]) != len(info["gt_boxes"]):
            raise RuntimeError("nuScenes annotation order/count mismatch")
        centers = []
        for ann_token, center in zip(sample["anns"], info["gt_boxes"][:, :3]):
            ann = nusc.get("sample_annotation", ann_token)
            if ann["prev"]:
                centers.append([float(v) for v in center])
        history["samples"][info["token"]] = centers

    output, history_output, csv_output = map(Path, (output, history_output, csv_output))
    for path in (output, history_output, csv_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schedule, indent=2, sort_keys=True) + "\n")
    history_output.write_text(json.dumps(history, separators=(",", ":")) + "\n")
    with csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"fault_frames={len(rows)} total_frames={len(infos)} ratio={len(rows)/len(infos):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--info", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history-output", required=True)
    parser.add_argument("--csv-output", required=True)
    parser.add_argument("--seed", type=int, default=314159)
    args = parser.parse_args()
    generate(args.info, args.data_root, args.output, args.history_output,
             args.csv_output, args.seed)
