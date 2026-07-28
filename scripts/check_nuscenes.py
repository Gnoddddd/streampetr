#!/usr/bin/env python3
from pathlib import Path
from nuscenes.nuscenes import NuScenes

ROOT = Path.home() / "research/evidence3d/data/nuscenes-mini"
nusc = NuScenes(version="v1.0-mini", dataroot=str(ROOT), verbose=False)
sample = nusc.sample[0]
for name in (
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK",
    "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
):
    record = nusc.get("sample_data", sample["data"][name])
    path = ROOT / record["filename"]
    print(f"{name}: {path} exists={path.exists()}")
    if not path.exists():
        raise FileNotFoundError(path)
print("nuScenes-mini数据检查通过。")
