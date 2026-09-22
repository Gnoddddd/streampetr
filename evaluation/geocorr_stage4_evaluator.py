"""Evaluation helpers for GeoCorr Stage 4 formal subset experiments."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


def safe_condition_name(condition: str) -> str:
    return condition.replace("/", "__").replace(" ", "_")


def condition_records(
    records: Sequence[Mapping[str, object]],
    conditions: Optional[Sequence[str]] = None,
    max_pairs: Optional[int] = None,
) -> Dict[str, List[Mapping[str, object]]]:
    """Group records by corruption condition with strict per-condition token uniqueness.

    ``max_pairs`` is applied independently to each selected condition, which is
    convenient for short evaluator smokes while leaving the formal path uncapped.
    """
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
            raise ValueError(
                "duplicate current sample token within condition %s: %s"
                % (condition, token)
            )
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


def current_tokens(records: Sequence[Mapping[str, object]]) -> List[str]:
    tokens = [str(record["current_sample_token"]) for record in records]
    if len(tokens) != len(set(tokens)):
        raise ValueError("evaluation records contain duplicate current sample tokens")
    return tokens


def unique_records_by_current_token(
    grouped: Mapping[str, Sequence[Mapping[str, object]]]
) -> List[Mapping[str, object]]:
    """Return one canonical record for each current token across conditions."""
    result: List[Mapping[str, object]] = []
    seen = set()
    for condition in sorted(grouped):
        for record in grouped[condition]:
            token = str(record["current_sample_token"])
            if token not in seen:
                seen.add(token)
                result.append(record)
    return result


def normalize_prediction(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize StreamPETR output to one outer ``pts_bbox`` result mapping."""
    prediction = result.get("pts_bbox", result)
    required = {"boxes_3d", "scores_3d", "labels_3d"}
    if not required.issubset(prediction):
        raise KeyError("StreamPETR result lacks decoded bbox fields")
    return {"pts_bbox": prediction}


def validate_result_token_sets(
    baseline_tokens: Sequence[str], geocorr_tokens: Sequence[str]
) -> None:
    if list(baseline_tokens) != list(geocorr_tokens):
        raise ValueError("baseline and GeoCorr sample-token orders differ")


def recovery_diagnostics(recovery: Any) -> Dict[str, float]:
    valid = recovery.query_valid.detach().bool()
    confidence = recovery.confidence.detach().squeeze(-1).masked_select(valid)
    delta = torch.linalg.norm(recovery.delta_q.detach().float(), dim=-1)
    weighted = torch.linalg.norm(
        (recovery.confidence * recovery.delta_q).detach().float(), dim=-1
    )
    delta_selected = delta.masked_select(valid)
    weighted_selected = weighted.masked_select(valid)
    if recovery.recovered_current_fpn is None or recovery.current_dirty_fpn is None:
        raise RuntimeError("formal GeoCorr inference requires dirty and recovered FPN")
    return {
        "confidence_mean": float(confidence.mean().item()) if confidence.numel() else 0.0,
        "confidence_median": float(confidence.median().item()) if confidence.numel() else 0.0,
        "query_valid_ratio": float(valid.float().mean().item()),
        "delta_q_l2": float(delta_selected.mean().item()) if delta_selected.numel() else 0.0,
        "g_delta_q_l2": (
            float(weighted_selected.mean().item()) if weighted_selected.numel() else 0.0
        ),
        "fpn_residual_l2": float(
            torch.linalg.norm(
                (recovery.recovered_current_fpn - recovery.current_dirty_fpn)
                .detach()
                .float()
            ).item()
        ),
    }


def average_diagnostics(values: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not values:
        return {
            "confidence_mean": 0.0,
            "confidence_median": 0.0,
            "query_valid_ratio": 0.0,
            "delta_q_l2": 0.0,
            "g_delta_q_l2": 0.0,
            "fpn_residual_l2": 0.0,
        }
    keys = tuple(values[0].keys())
    return {
        key: float(sum(float(value[key]) for value in values) / len(values))
        for key in keys
    }


def _mean_metric(items: Sequence[Mapping[str, float]], key: str) -> float:
    if not items:
        return 0.0
    return float(sum(float(item[key]) for item in items) / len(items))


def aggregate_fault_summary(
    condition_metrics: Mapping[str, Mapping[str, Mapping[str, float]]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Aggregate severity-level mAP/NDS without mixing clean into fault averages."""
    groups = {
        "Average Dirt": [name for name in condition_metrics if name.startswith("Dirt/")],
        "Average Water": [
            name for name in condition_metrics if name.startswith("Water-blur/")
        ],
        "Average Fault": list(condition_metrics),
    }
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for label, names in groups.items():
        if not names:
            continue
        summary[label] = {}
        for mode in ("baseline", "geocorr"):
            mode_items = [
                condition_metrics[name][mode]
                for name in names
                if mode in condition_metrics[name]
            ]
            if not mode_items:
                continue
            summary[label][mode] = {
                "mAP": _mean_metric(mode_items, "mean_ap"),
                "NDS": _mean_metric(mode_items, "nd_score"),
            }
        if "baseline" in summary[label] and "geocorr" in summary[label]:
            summary[label]["delta"] = {
                "mAP": summary[label]["geocorr"]["mAP"]
                - summary[label]["baseline"]["mAP"],
                "NDS": summary[label]["geocorr"]["NDS"]
                - summary[label]["baseline"]["NDS"],
            }
    return summary


def checkpoint_payload(
    path: Path,
    expected_detector_checksum: Optional[str] = None,
) -> Mapping[str, Any]:
    payload = torch.load(str(path), map_location="cpu")
    required = {"geocorr", "config", "detector_checksum", "step", "epoch"}
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError("GeoCorr checkpoint missing keys: %s" % missing)
    if (
        expected_detector_checksum is not Ndata-info view."""
    if len(token_order) != len(results):
        raise ValueError("prediction count and token count differ")
    index = {str(info["token"]): i for i, info in enumerate(dataset.data_infos)}
    missing = [token for token in token_order if token not in index]
    if missing:
        raise KeyError("subset tokens absent from StreamPETR val infos: %s" % missing[:5])
    subset = copy.copy(dataset)
    subset.data_infos = [dataset.data_infos[index[token]] for token in token_order]
    output_dir.mkdir(parents=True, exist_ok=True)
    result_files, tmp_dir = subset.format_results(
        list(results), jsonfile_prefix=str(output_dir / "formatted")
    )
    if tmp_dir is not None:
        raise RuntimeError("explicit subset result prefix unexpectedly created a temp dir")
    if isinstance(result_files, Mapping):
        if "pts_bbox" in result_files:
            result_path = result_files["pts_bbox"]
        elif len(result_files) == 1:
            result_path = next(iter(result_files.values()))
        else:
            raise RuntimeError("ambiguous formatted result files: %s" % result_files)
    else:
        result_path = result_files
    return Path(result_path)


def _subset_eval_boxes(source: Any, tokens: Sequence[str]) -> Any:
    from nuscenes.eval.common.data_classes import EvalBoxes

    subset = EvalBoxes()
    source_tokens = set(source.sample_tokens)
    for token in tokens:
        if token not in source_tokens:
            raise KeyError("nuScenes EvalBoxes missing sample token: %s" % token)
        subset.add_boxes(token, source.boxes[token])
    return subset


class NuScenesSubsetMetric:
    """Official nuScenes detection metric algebra on an explicit token allowlist."""

    def __init__(self, nuscenes_root: Path, version: str, config: Any) -> None:
        from nuscenes import NuScenes
        from nuscenes.eval.common.loaders import add_center_dist, load_gt
        from nuscenes.eval.detection.data_classes import DetectionBox

        self.config = config
        self.nusc = NuScenes(version=version, dataroot=str(nuscenes_root), verbose=False)
        eval_split = "val" if version == "v1.0-trainval" else "mini_val"
        self.full_gt = load_gt(self.nusc, eval_split, DetectionBox, verbose=False)
        add_center_dist(self.nusc, self.full_gt)

    def evaluate(
        self,
        result_path: Path,
        token_order: Sequence[str],
        output_dir: Path,
    ) -> Dict[str, Any]:
        from nuscenes.eval.common.loaders import (
            add_center_dist,
            filter_eval_boxes,
            load_prediction,
        )
        from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
        from nuscenes.eval.detection.constants import TP_METRICS
        from nuscenes.eval.detection.data_classes import (
            DetectionBox,
            DetectionMetricDataList,
            DetectionMetrics,
        )

        started = time.time()
        pred_boxes, meta = load_prediction(
            str(result_path), self.config.max_boxes_per_sample, DetectionBox, verbose=False
        )
        add_center_dist(self.nusc, pred_boxes)
        gt_boxes = _subset_eval_boxes(self.full_gt, token_order)
        pred_boxes = _subset_eval_boxes(pred_boxes, token_order)
        gt_boxes = filter_eval_boxes(
            self.nusc, gt_boxes, self.config.class_range, verbose=False
        )
        pred_boxes = filter_eval_boxes(
            self.nusc, pred_boxes, self.config.class_range, verbose=False
        )

        metric_data_list = DetectionMetricDataList()
        for class_name in self.config.class_names:
            for dist_th in self.config.dist_ths:
                metric_data = accumulate(
                    gt_boxes,
                    pred_boxes,
                    class_name,
                    self.config.dist_fcn_callable,
                    dist_th,
                    verbose=False,
                )
                metric_data_list.set(class_name, dist_th, metric_data)

        metrics = DetectionMetrics(self.config)
        for class_name in self.config.class_names:
            for dist_th in self.config.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.config.min_recall, self.config.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.config.dist_th_tp)]
                tp = calc_tp(metric_data, self.config.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)
        metrics.add_runtime(time.time() - started)
        serialized = metrics.serialize()
        serialized["meta"] = meta
        serialized["subset_sample_count"] = len(token_order)
        serialized["sample_tokens"] = list(token_order)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "metrics_summary.json", serialized)
        return serialized


__all__ = [
    "NuScenesSubsetMetric",
    "aggregate_fault_summary",
    "average_diagnostics",
    "checkpoint_payload",
    "condition_records",
    "current_tokens",
    "format_subset_results",
    "load_progress",
    "normalize_prediction",
    "recovery_diagnostics",
    "safe_condition_name",
    "save_progress",
    "unique_records_by_current_token",
    "validate_result_token_sets",
    "write_json",
]
