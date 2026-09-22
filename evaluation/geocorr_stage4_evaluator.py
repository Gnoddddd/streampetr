"""Stage 4 helpers for GeoCorr formal subset evaluation."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch


def safe_condition_name(condition: str) -> str:
    return condition.replace("/", "__").replace(" ", "_")


def condition_records(records, conditions=None, max_pairs=None):
    if max_pairs is not None and max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    allowed = set(conditions) if conditions else None
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    seen: Dict[str, set] = {}
    for record in records:
        condition = str(record.get("raw_condition"))
        if allowed is not None and condition not in allowed:
            continue
        token = str(record.get("current_sample_token"))
        seen.setdefault(condition, set())
        if token in seen[condition]:
            raise ValueError("duplicate current sample token within condition %s: %s" % (condition, token))
        if max_pairs is not None and len(grouped.get(condition, [])) >= max_pairs:
            continue
        seen[condition].add(token)
        grouped.setdefault(condition, []).append(record)
    if allowed is not None:
        missing = sorted(allowed - set(grouped))
        if missing:
            raise ValueError("requested conditions absent from manifest: %s" % missing)
    if not grouped:
        raise ValueError("no evaluation records selected")
    return grouped


def current_tokens(records):
    tokens = [str(record["current_sample_token"]) for record in records]
    if len(tokens) != len(set(tokens)):
        raise ValueError("evaluation records contain duplicate current sample tokens")
    return tokens


def unique_records_by_current_token(grouped):
    result, seen = [], set()
    for condition in sorted(grouped):
        for record in grouped[condition]:
            token = str(record["current_sample_token"])
            if token not in seen:
                seen.add(token)
                result.append(record)
    return result


def normalize_prediction(result):
    prediction = result.get("pts_bbox", result)
    required = {"boxes_3d", "scores_3d", "labels_3d"}
    if not required.issubset(prediction):
        raise KeyError("StreamPETR result lacks decoded bbox fields")
    return {"pts_bbox": prediction}


def validate_result_token_sets(baseline_tokens, geocorr_tokens):
    if list(baseline_tokens) != list(geocorr_tokens):
        raise ValueError("baseline and GeoCorr sample-token orders differ")


def recovery_diagnostics(recovery):
    valid = recovery.query_valid.detach().bool()
    confidence = recovery.confidence.detach().squeeze(-1).masked_select(valid)
    delta = torch.linalg.norm(recovery.delta_q.detach().float(), dim=-1).masked_select(valid)
    weighted = torch.linalg.norm(
        (recovery.confidence * recovery.delta_q).detach().float(), dim=-1
    ).masked_select(valid)
    if recovery.recovered_current_fpn is None or recovery.current_dirty_fpn is None:
        raise RuntimeError("formal GeoCorr inference requires dirty and recovered FPN")
    return {
        "confidence_mean": float(confidence.mean().item()) if confidence.numel() else 0.0,
        "confidence_median": float(confidence.median().item()) if confidence.numel() else 0.0,
        "query_valid_ratio": float(valid.float().mean().item()),
        "delta_q_l2": float(delta.mean().item()) if delta.numel() else 0.0,
        "g_delta_q_l2": float(weighted.mean().item()) if weighted.numel() else 0.0,
        "fpn_residual_l2": float(torch.linalg.norm(
            (recovery.recovered_current_fpn - recovery.current_dirty_fpn).detach().float()
        ).item()),
    }


def average_diagnostics(values):
    keys = ("confidence_mean", "confidence_median", "query_valid_ratio", "delta_q_l2", "g_delta_q_l2", "fpn_residual_l2")
    if not values:
        return {key: 0.0 for key in keys}
    return {key: sum(float(value[key]) for value in values) / len(values) for key in keys}


def aggregate_fault_summary(condition_metrics):
    groups = {
        "Average Dirt": [x for x in condition_metrics if x.startswith("Dirt/")],
        "Average Water": [x for x in condition_metrics if x.startswith("Water-blur/")],
        "Average Fault": list(condition_metrics),
    }
    summary = {}
    for label, names in groups.items():
        if not names:
            continue
        summary[label] = {}
        for mode in ("baseline", "geocorr"):
            items = [condition_metrics[name][mode] for name in names if mode in condition_metrics[name]]
            if items:
                summary[label][mode] = {
                    "mAP": sum(float(x["mean_ap"]) for x in items) / len(items),
                    "NDS": sum(float(x["nd_score"]) for x in items) / len(items),
                }
        if "baseline" in summary[label] and "geocorr" in summary[label]:
            summary[label]["delta"] = {
                "mAP": summary[label]["geocorr"]["mAP"] - summary[label]["baseline"]["mAP"],
                "NDS": summary[label]["geocorr"]["NDS"] - summary[label]["baseline"]["NDS"],
            }
    return summary


def checkpoint_payload(path: Path, expected_detector_checksum: Optional[str] = None):
    payload = torch.load(str(path), map_location="cpu")
    required = {"geocorr", "config", "detector_checksum", "step", "epoch"}
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError("GeoCorr checkpoint missing keys: %s" % missing)
    if expected_detector_checksum is not None and payload["detector_checksum"] != expected_detector_checksum:
        raise RuntimeError("GeoCorr checkpoint detector checksum mismatch")
    return payload


def save_progress(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), str(path))


def load_progress(path: Path, expected_condition: str, expected_tokens):
    if not path.is_file():
        return None
    payload = torch.load(str(path), map_location="cpu")
    if payload.get("condition") != expected_condition:
        raise RuntimeError("evaluation progress condition mismatch")
    if list(payload.get("tokens", [])) != list(expected_tokens):
        raise RuntimeError("evaluation progress token order mismatch")
    cursor = int(payload.get("next_index", 0))
    if cursor < 0 or cursor > len(expected_tokens):
        raise RuntimeError("evaluation progress cursor is invalid")
    return payload


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def format_subset_results(dataset, token_order, results, output_dir: Path) -> Path:
    if len(token_order) != len(results):
        raise ValueError("prediction count and token count differ")
    index = {str(info["token"]): i for i, info in enumerate(dataset.data_infos)}
    missing = [token for token in token_order if token not in index]
    if missing:
        raise KeyError("subset tokens absent from StreamPETR val infos: %s" % missing[:5])
    subset = copy.copy(dataset)
    subset.data_infos = [dataset.data_infos[index[token]] for token in token_order]
    output_dir.mkdir(parents=True, exist_ok=True)
    result_files, tmp_dir = subset.format_results(list(results), jsonfile_prefix=str(output_dir / "formatted"))
    if tmp_dir is not None:
        raise RuntimeError("explicit subset result prefix unexpectedly created temp dir")
    if isinstance(result_files, Mapping):
        result_path = result_files.get("pts_bbox")
        if result_path is None and len(result_files) == 1:
            result_path = next(iter(result_files.values()))
        if result_path is None:
            raise RuntimeError("ambiguous formatted result files: %s" % result_files)
    else:
        result_path = result_files
    return Path(result_path)


def _subset_boxes(source, tokens):
    from nuscenes.eval.common.data_classes import EvalBoxes
    subset = EvalBoxes()
    available = set(source.sample_tokens)
    for token in tokens:
        if token not in available:
            raise KeyError("nuScenes EvalBoxes missing sample token: %s" % token)
        subset.add_boxes(token, source.boxes[token])
    return subset


class NuScenesSubsetMetric:
    """Official nuScenes detection metric algebra on an explicit token allowlist."""

    def __init__(self, nuscenes_root: Path, version: str, config):
        from nuscenes import NuScenes
        from nuscenes.eval.common.loaders import add_center_dist, load_gt
        from nuscenes.eval.detection.data_classes import DetectionBox
        self.config = config
        self.nusc = NuScenes(version=version, dataroot=str(nuscenes_root), verbose=False)
        eval_split = "val" if version == "v1.0-trainval" else "mini_val"
        self.full_gt = load_gt(self.nusc, eval_split, DetectionBox, verbose=False)
        add_center_dist(self.nusc, self.full_gt)

    def evaluate(self, result_path: Path, token_order, output_dir: Path):
        from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes, load_prediction
        from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
        from nuscenes.eval.detection.constants import TP_METRICS
        from nuscenes.eval.detection.data_classes import DetectionBox, DetectionMetricDataList, DetectionMetrics
        started = time.time()
        pred, meta = load_prediction(str(result_path), self.config.max_boxes_per_sample, DetectionBox, verbose=False)
        add_center_dist(self.nusc, pred)
        gt = _subset_boxes(self.full_gt, token_order)
        pred = _subset_boxes(pred, token_order)
        gt = filter_eval_boxes(self.nusc, gt, self.config.class_range, verbose=False)
        pred = filter_eval_boxes(self.nusc, pred, self.config.class_range, verbose=False)
        data = DetectionMetricDataList()
        for class_name in self.config.class_names:
            for dist_th in self.config.dist_ths:
                data.set(class_name, dist_th, accumulate(gt, pred, class_name, self.config.dist_fcn_callable, dist_th, verbose=False))
        metrics = DetectionMetrics(self.config)
        for class_name in self.config.class_names:
            for dist_th in self.config.dist_ths:
                md = data[(class_name, dist_th)]
                metrics.add_label_ap(class_name, dist_th, calc_ap(md, self.config.min_recall, self.config.min_precision))
            for metric_name in TP_METRICS:
                md = data[(class_name, self.config.dist_th_tp)]
                metrics.add_label_tp(class_name, metric_name, calc_tp(md, self.config.min_recall, metric_name))
        metrics.add_runtime(time.time() - started)
        out = metrics.serialize()
        out["meta"] = meta
        out["subset_sample_count"] = len(token_order)
        out["sample_tokens"] = list(token_order)
        write_json(output_dir / "metrics_summary.json", out)
        return out


__all__ = [
    "NuScenesSubsetMetric", "aggregate_fault_summary", "average_diagnostics",
    "checkpoint_payload", "condition_records", "current_tokens", "format_subset_results",
    "load_progress", "normalize_prediction", "recovery_diagnostics", "safe_condition_name",
    "save_progress", "unique_records_by_current_token", "validate_result_token_sets", "write_json",
]
