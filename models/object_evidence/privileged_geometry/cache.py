"""Offline LiDAR object-token cache IO."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
from torch import Tensor


REQUIRED_MANIFEST_KEYS = {
    "teacher_config_sha256", "teacher_checkpoint_sha256", "feature_layer",
    "feature_dim", "split", "scene_count", "sample_count",
}


class LidarObjectEvidenceCache:
    def __init__(self, path):
        root = Path(path)
        if root.is_dir():
            manifest = json.loads((root / "manifest.json").read_text())
            payload = torch.load(str(root / "records.pt"), map_location="cpu")
        else:
            payload = torch.load(str(root), map_location="cpu")
            manifest = payload["manifest"]
            payload = payload["records"]
        missing = REQUIRED_MANIFEST_KEYS - set(manifest)
        if missing or manifest.get("split") != "train":
            raise ValueError(f"invalid train cache manifest; missing={sorted(missing)}")
        self.manifest = manifest
        self._records = {}
        for record in payload:
            key = (str(record["sample_token"]), str(record["annotation_token"]))
            if key in self._records:
                raise ValueError(f"duplicate cache record {key}")
            self._records[key] = record

    def get(self, sample_token: str, annotation_token: str) -> Optional[Mapping[str, Any]]:
        return self._records.get((str(sample_token), str(annotation_token)))

    def teacher_tokens(self, identities, device=None, dtype=None):
        records = [self.get(sample, annotation) for sample, annotation in identities]
        valid = torch.tensor([record is not None for record in records], dtype=torch.bool, device=device)
        dimension = int(self.manifest["feature_dim"])
        tokens = torch.zeros((len(records), dimension), dtype=dtype or torch.float32, device=device)
        points = torch.zeros(len(records), dtype=torch.float32, device=device)
        for index, record in enumerate(records):
            if record is not None:
                tokens[index] = torch.as_tensor(record["teacher_token"], device=device, dtype=tokens.dtype)
                points[index] = float(record["num_lidar_pts"])
        return tokens, points, valid


def save_cache(path, records: Iterable[Dict[str, Any]], manifest: Dict[str, Any]) -> None:
    if manifest.get("split") != "train":
        raise ValueError("privileged geometry cache may only contain the train split")
    missing = REQUIRED_MANIFEST_KEYS - set(manifest)
    if missing:
        raise ValueError(f"manifest missing {sorted(missing)}")
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    serializable = []
    for record in records:
        item = dict(record)
        item["teacher_token"] = torch.as_tensor(item["teacher_token"]).detach().cpu().half()
        serializable.append(item)
    torch.save(serializable, str(root / "records.pt"))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
