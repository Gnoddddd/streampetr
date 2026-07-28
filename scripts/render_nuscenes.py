#!/usr/bin/env python3
from pathlib import Path
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes

ROOT = Path.home() / "research/evidence3d/data/nuscenes-mini"
OUT = Path.home() / "research/evidence3d/outputs/nuscenes_mini_sample.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
nusc = NuScenes(version="v1.0-mini", dataroot=str(ROOT), verbose=False)
fig = nusc.render_sample(nusc.sample[0]["token"], out_path=str(OUT), verbose=False)
plt.close("all")
print(OUT)
