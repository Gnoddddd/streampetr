"""Opt-in frozen-decoder active recovery injection audit.

Import is a no-op unless ``ACTIVE_RECOVERY_MODE`` is Q1, Q2, or Q3. The
wrapper computes an independent Q0 head forward for the canonical memory state,
then replays the same frame with emit-only query replacement. The Q0 post-frame
memory is restored before returning.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PC_RANGE = np.asarray([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], np.float32)
POST_RANGE = np.asarray([-61.2, -61.2, -10, 61.2, 61.2, 10], np.float32)
Q_VELOCITY = np.asarray(
    [[0.14987897, -0.05169269], [-0.05169269, 0.12780007]],
    np.float64,
)
SCORE_THRESHOLD = 0.10
MAX_AGE = 10
PRIMARY_R = 4


def normalize_reference(center: np.ndarray) -> np.ndarray:
    return (np.asarray(center, np.float32) - PC_RANGE[:3]) / (
        PC_RANGE[3:] - PC_RANGE[:3]
    )


def denormalize_reference(reference: np.ndarray) -> np.ndarray:
    return np.asarray(reference, np.float32) * (
        PC_RANGE[3:] - PC_RANGE[:3]
    ) + PC_RANGE[:3]


def sigma_points(mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(np.asarray(covariance, np.float64))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    axis0 = vectors[:, 0] * np.sqrt(max(values[0], 0.0))
    axis1 = vectors[:, 1] * np.sqrt(max(values[1], 0.0))
    return np.stack([mean, mean + axis0, mean - axis0, mean + axis1])


def replace_learned_tail(
    tgt: torch.Tensor,
    query_pos: torch.Tensor,
    reference: torch.Tensor,
    contents: torch.Tensor,
    positions: torch.Tensor,
    encoded_positions: torch.Tensor,
    learned_queries: int = 644,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, slice]:
    count = int(contents.shape[0])
    if not 0 < count <= learned_queries:
        raise ValueError("invalid injected query count")
    target = slice(learned_queries - count, learned_queries)
    output_tgt = tgt.clone()
    output_pos = query_pos.clone()
    output_ref = reference.clone()
    output_tgt[0, target] = contents.detach()
    output_pos[0, target] = encoded_positions.reshape(
        count, encoded_positions.shape[-1]
    )
    output_ref[0, target] = positions
    assert output_ref.shape[1] == reference.shape[1]
    return output_tgt, output_pos, output_ref, target


def constant_velocity(
    center: np.ndarray, velocity: np.ndarray, elapsed: float
) -> np.ndarray:
    output = np.asarray(center, np.float64).copy()
    output[:2] += np.asarray(velocity, np.float64)[:2] * elapsed
    return output


def ego_compensated_propagation(
    center: np.ndarray,
    velocity: np.ndarray,
    previous_rotation: np.ndarray,
    previous_translation: np.ndarray,
    current_rotation: np.ndarray,
    current_translation: np.ndarray,
    elapsed: float,
) -> tuple[np.ndarray, np.ndarray]:
    velocity_global = previous_rotation @ np.r_[velocity[:2], 0.0]
    center_global = (
        previous_rotation @ center[:3]
        + previous_translation
        + velocity_global * elapsed
    )
    return (
        current_rotation.T @ (center_global - current_translation),
        (current_rotation.T @ velocity_global)[:2],
    )


@dataclass
class Detection:
    query: int
    label: int
    score: float
    center: np.ndarray
    velocity: np.ndarray
    content: np.ndarray
    logits: np.ndarray


@dataclass
class Track:
    identity: int
    detection: Detection
    sample_token: str
    consecutive: int
    age: int
    pre_fault_reliable: bool


def _inside(center: np.ndarray) -> bool:
    return bool(
        np.all(np.asarray(center) >= POST_RANGE[:3])
        and np.all(np.asarray(center) <= POST_RANGE[3:])
    )


def _schedule_active(plan: dict, scene: str, frame: int) -> bool:
    values = list(plan.get("scenes", {}).get("*", []))
    values += list(plan.get("scenes", {}).get(scene, []))
    return any(
        int(value["start_frame"]) <= frame <= int(value["end_frame"])
        for value in values
    )


def _snapshot(head) -> dict:
    names = (
        "memory_embedding",
        "memory_reference_point",
        "memory_timestamp",
        "memory_egopose",
        "memory_velo",
    )
    return {
        name: (
            None
            if getattr(head, name) is None
            else getattr(head, name).detach().clone()
        )
        for name in names
    }


def _restore(head, state: dict) -> None:
    for name, value in state.items():
        setattr(head, name, None if value is None else value.detach().clone())


def _state_diff(left: dict, right: dict) -> float:
    difference = 0.0
    for name in left:
        if left[name] is None or right[name] is None:
            if left[name] is not right[name]:
                return float("inf")
            continue
        if left[name].numel():
            difference = max(
                difference,
                float((left[name] - right[name]).abs().max().item()),
            )
    return difference


def _install() -> None:
    mode = os.environ.get("ACTIVE_RECOVERY_MODE", "").upper()
    if mode not in {"Q1", "Q2", "Q3"}:
        return
    output = Path(os.environ["ACTIVE_RECOVERY_TRACE_DIR"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = os.environ.get("EVIDENCE3D_PROTOCOL")
    plan = (
        {"scenes": {}}
        if not protocol_path
        else json.loads(Path(protocol_path).read_text())
    )

    from nuscenes.eval.detection.utils import category_to_detection_name
    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion
    from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
    from projects.mmdet3d_plugin.models.dense_heads.streampetr_head import (
        StreamPETRHead,
    )
    from projects.mmdet3d_plugin.models.utils.positional_encoding import (
        pos2posemb1d,
        pos2posemb3d,
    )

    if getattr(StreamPETRHead, "_active_recovery_installed", False):
        return
    root = Path(__file__).resolve().parents[1]
    nusc = NuScenes(
        version="v1.0-mini",
        dataroot=str(root / "data/nuscenes-mini"),
        verbose=False,
    )
    class_names = (
        "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
        "motorcycle", "bicycle", "pedestrian", "traffic_cone",
    )
    class_to_index = {name: index for index, name in enumerate(class_names)}
    original_forward = StreamPETRHead.forward
    deploy_tracks: list[Track] = []
    oracle_history = {}
    next_identity = 0

    def lidar_pose(token: str) -> tuple[np.ndarray, np.ndarray]:
        sample = nusc.get("sample", token)
        sample_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        calibrated = nusc.get(
            "calibrated_sensor", sample_data["calibrated_sensor_token"]
        )
        ego = nusc.get("ego_pose", sample_data["ego_pose_token"])
        sensor_rotation = Quaternion(
            calibrated["rotation"]
        ).rotation_matrix
        ego_rotation = Quaternion(ego["rotation"]).rotation_matrix
        return (
            ego_rotation @ sensor_rotation,
            ego_rotation @ np.asarray(calibrated["translation"], float)
            + np.asarray(ego["translation"], float),
        )

    def propagate(
        detection: Detection,
        previous_token: str,
        current_token: str,
        elapsed: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        previous_rotation, previous_translation = lidar_pose(previous_token)
        current_rotation, current_translation = lidar_pose(current_token)
        return ego_compensated_propagation(
            detection.center,
            detection.velocity,
            previous_rotation,
            previous_translation,
            current_rotation,
            current_translation,
            elapsed,
        )

    def local_gt(token: str) -> list[dict]:
        sample = nusc.get("sample", token)
        _, boxes, _ = nusc.get_sample_data(sample["data"]["LIDAR_TOP"])
        values = []
        for box in boxes:
            name = category_to_detection_name(box.name)
            if name not in class_to_index:
                continue
            annotation = nusc.get("sample_annotation", box.token)
            values.append({
                "instance": annotation["instance_token"],
                "center": np.asarray(box.center, np.float32),
                "label": class_to_index[name],
            })
        return values

    def capture_forward(head, memory_center, img_metas, topk_indexes, data):
        captured = {}
        transformer_forward = head.transformer.forward

        def transformer(*args, **kwargs):
            decoder_start = decoder_end = None
            if args[0].is_cuda:
                decoder_start = torch.cuda.Event(enable_timing=True)
                decoder_end = torch.cuda.Event(enable_timing=True)
                decoder_start.record()
            result = transformer_forward(*args, **kwargs)
            if decoder_end is not None:
                decoder_end.record()
                captured["decoder_events"] = (decoder_start, decoder_end)
            captured["outs_dec"] = result[0]
            captured["decoder_tgt_shape"] = tuple(args[1].shape)
            return result

        head.transformer.forward = transformer
        start_event = end_event = None
        if memory_center.is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        cpu_start = time.perf_counter()
        try:
            result = original_forward(
                head, memory_center.clone(), img_metas, topk_indexes, **data
            )
        finally:
            head.transformer.forward = transformer_forward
        cpu_ms = (time.perf_counter() - cpu_start) * 1000
        gpu_ms = float("nan")
        if end_event is not None:
            end_event.record()
            torch.cuda.synchronize()
            gpu_ms = float(start_event.elapsed_time(end_event))
        if "decoder_events" in captured:
            captured["decoder_gpu_ms"] = float(
                captured["decoder_events"][0].elapsed_time(
                    captured["decoder_events"][1]
                )
            )
        return result, captured, cpu_ms, gpu_ms

    def detections(head, result, captured) -> list[Detection]:
        logits = result["all_cls_scores"][-1, 0]
        boxes_raw = result["all_bbox_preds"][-1, 0]
        probabilities = logits.sigmoid()
        scores, labels = probabilities.max(-1)
        order = scores.argsort(descending=True)
        values = []
        for query in order.tolist():
            if len(values) >= int(head.bbox_coder.max_num):
                break
            score = float(scores[query])
            if score < SCORE_THRESHOLD:
                continue
            box = denormalize_bbox(
                boxes_raw[query : query + 1], head.pc_range
            )[0]
            center = box[:3].detach().cpu().numpy()
            if not _inside(center):
                continue
            values.append(Detection(
                query=query,
                label=int(labels[query]),
                score=score,
                center=center,
                velocity=box[7:9].detach().cpu().numpy(),
                content=captured["outs_dec"][-1, 0, query]
                .detach().cpu().numpy(),
                logits=logits[query].detach().cpu().numpy(),
            ))
        return values

    def associate_tracks(
        values: list[Detection],
        token: str,
        active: bool,
    ) -> tuple[list[Track], list[Track]]:
        nonlocal next_identity, deploy_tracks
        used = set()
        updated, dormant = [], []
        for track in deploy_tracks:
            center, velocity = propagate(
                track.detection, track.sample_token, token, 0.5
            )
            candidates = [
                (index, value)
                for index, value in enumerate(values)
                if index not in used
                and value.label == track.detection.label
                and np.linalg.norm(value.center[:2] - center[:2]) <= 2.0
            ]
            if candidates:
                index, value = min(
                    candidates,
                    key=lambda pair: np.linalg.norm(
                        pair[1].center[:2] - center[:2]
                    ),
                )
                used.add(index)
                updated.append(Track(
                    track.identity,
                    value,
                    token,
                    track.consecutive + 1,
                    0,
                    track.pre_fault_reliable
                    or (not active and track.consecutive + 1 >= 2),
                ))
            else:
                age = track.age + 1
                predicted = Detection(
                    track.detection.query,
                    track.detection.label,
                    track.detection.score,
                    center,
                    velocity,
                    track.detection.content,
                    track.detection.logits,
                )
                aged = Track(
                    track.identity,
                    predicted,
                    token,
                    0,
                    age,
                    track.pre_fault_reliable,
                )
                if (
                    active
                    and aged.pre_fault_reliable
                    and age <= MAX_AGE
                    and _inside(center)
                ):
                    dormant.append(aged)
                if age <= MAX_AGE and _inside(center):
                    updated.append(aged)
        for index, value in enumerate(values):
            if index in used:
                continue
            updated.append(
                Track(next_identity, value, token, 1, 0, False)
            )
            next_identity += 1
        deploy_tracks = updated
        dormant.sort(
            key=lambda track: (
                track.age,
                -track.detection.score,
                token,
                track.detection.query,
            )
        )
        return updated, dormant

    def q1_specs(
        values: list[Detection],
        gt: list[dict],
        active: bool,
        token: str,
        frame_idx: int,
    ) -> list[dict]:
        matches = {}
        used = set()
        for item in gt:
            candidates = [
                (index, value)
                for index, value in enumerate(values)
                if index not in used
                and value.label == item["label"]
                and np.linalg.norm(value.center - item["center"]) <= 2.0
            ]
            if candidates:
                index, value = min(
                    candidates,
                    key=lambda pair: np.linalg.norm(
                        pair[1].center - item["center"]
                    ),
                )
                used.add(index)
                matches[item["instance"]] = value
                history = oracle_history.get(item["instance"])
                streak = 1 if history is None else history["streak"] + 1
                if not active:
                    oracle_history[item["instance"]] = {
                        "detection": value,
                        "streak": streak,
                        "token": token,
                        "frame_idx": frame_idx,
                    }
        if not active:
            return []
        specs = []
        for item in gt:
            if item["instance"] in matches:
                continue
            history = oracle_history.get(item["instance"])
            if history is None or history["streak"] < 2:
                continue
            value = history["detection"]
            specs.append({
                "identity": item["instance"],
                "query": value.query,
                "content": value.content,
                "logits": value.logits,
                "center": item["center"],
                "age": max(1, frame_idx - int(history["frame_idx"])),
                "history_score": value.score,
                "oracle_gt_center": item["center"],
            })
        specs.sort(
            key=lambda item: (
                item["age"], -item["history_score"], token, item["query"]
            )
        )
        return specs[:PRIMARY_R]

    def deploy_specs(
        dormant: list[Track],
        token: str,
        uncertainty: bool,
    ) -> list[dict]:
        if not dormant:
            return []
        if not uncertainty:
            selected = dormant[:PRIMARY_R]
            return [{
                "identity": str(track.identity),
                "query": track.detection.query,
                "content": track.detection.content,
                "logits": track.detection.logits,
                "center": track.detection.center,
                "age": track.age,
                "history_score": track.detection.score,
            } for track in selected]
        track = dormant[0]
        rotation, _ = lidar_pose(token)
        covariance = (
            rotation[:2, :2].T
            @ Q_VELOCITY
            @ rotation[:2, :2]
            * (track.age * 0.5) ** 2
        )
        points = sigma_points(track.detection.center[:2], covariance)
        return [{
            "identity": str(track.identity),
            "query": track.detection.query,
            "content": track.detection.content,
            "logits": track.detection.logits,
            "center": np.r_[point, track.detection.center[2]],
            "age": track.age,
            "history_score": track.detection.score,
            "sigma_index": index,
        } for index, point in enumerate(points)]

    def injected_forward(self, memory_center, img_metas, topk_indexes=None, **data):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("active recovery audit requires frozen inference")
        if len(img_metas) != 1:
            raise RuntimeError("active recovery audit requires batch size one")
        token = str(img_metas[0].get("sample_idx"))
        scene = str(img_metas[0].get("scene_token", ""))
        frame_idx = int(img_metas[0].get("frame_idx", -1))
        active = _schedule_active(plan, scene, frame_idx)
        before = _snapshot(self)
        q0, q0_capture, q0_cpu_ms, q0_gpu_ms = capture_forward(
            self, memory_center, img_metas, topk_indexes, data
        )
        q0_after = _snapshot(self)
        values = detections(self, q0, q0_capture)
        _, dormant = associate_tracks(values, token, active)
        if mode == "Q1":
            specs = q1_specs(
                values, local_gt(token), active, token, frame_idx
            )
        elif mode == "Q2":
            specs = deploy_specs(dormant, token, False) if active else []
        else:
            specs = deploy_specs(dormant, token, True) if active else []

        _restore(self, before)
        injected_capture = {}
        temporal_alignment = self.temporal_alignment
        transformer_forward = self.transformer.forward

        def alignment(*args, **kwargs):
            aligned = temporal_alignment(*args, **kwargs)
            tgt, query_pos, reference = aligned[:3]
            if not specs:
                return aligned
            positions_np = np.stack([
                normalize_reference(spec["center"]) for spec in specs
            ]).clip(0, 1)
            positions = reference.new_tensor(positions_np)
            contents = tgt.new_tensor(np.stack([
                spec["content"] for spec in specs
            ]))
            ages = reference.new_tensor([
                spec["age"] * 0.5 for spec in specs
            ]).view(1, -1, 1)
            encoded = self.query_embedding(pos2posemb3d(positions)).unsqueeze(0)
            encoded = encoded + self.time_embedding(
                pos2posemb1d(ages).float()
            )
            tgt, query_pos, reference, target = replace_learned_tail(
                tgt, query_pos, reference, contents, positions, encoded
            )
            injected_capture["slice"] = target
            injected_capture["initial_reference"] = positions
            return (
                tgt, query_pos, reference, *aligned[3:]
            )

        def transformer(*args, **kwargs):
            decoder_start = decoder_end = None
            if args[0].is_cuda:
                decoder_start = torch.cuda.Event(enable_timing=True)
                decoder_end = torch.cuda.Event(enable_timing=True)
                decoder_start.record()
            result = transformer_forward(*args, **kwargs)
            if decoder_end is not None:
                decoder_end.record()
                injected_capture["decoder_events"] = (
                    decoder_start, decoder_end
                )
            injected_capture["outs_dec"] = result[0]
            injected_capture["decoder_tgt_shape"] = tuple(args[1].shape)
            return result

        self.temporal_alignment = alignment
        self.transformer.forward = transformer
        start_event = end_event = None
        if memory_center.is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        cpu_start = time.perf_counter()
        try:
            injected = original_forward(
                self, memory_center.clone(), img_metas, topk_indexes, **data
            )
        finally:
            self.temporal_alignment = temporal_alignment
            self.transformer.forward = transformer_forward
        injected_cpu_ms = (time.perf_counter() - cpu_start) * 1000
        injected_gpu_ms = float("nan")
        if end_event is not None:
            end_event.record()
            torch.cuda.synchronize()
            injected_gpu_ms = float(start_event.elapsed_time(end_event))
        injected_decoder_gpu_ms = float("nan")
        if "decoder_events" in injected_capture:
            injected_decoder_gpu_ms = float(
                injected_capture["decoder_events"][0].elapsed_time(
                    injected_capture["decoder_events"][1]
                )
            )
        injection_after = _snapshot(self)
        _restore(self, q0_after)
        restored = _snapshot(self)
        memory_restore_diff = _state_diff(restored, q0_after)

        q0_logits = q0["all_cls_scores"][-1, 0]
        q0_boxes = q0["all_bbox_preds"][-1, 0]
        inj_logits = injected["all_cls_scores"][:, 0]
        inj_boxes = injected["all_bbox_preds"][:, 0]
        if specs:
            target = injected_capture["slice"]
            layer_logits = inj_logits[:, target]
            layer_boxes = inj_boxes[:, target]
            layer_features = injected_capture["outs_dec"][:, 0, target]
            slots = np.arange(target.start, target.stop, dtype=np.int16)
        else:
            layer_logits = inj_logits[:, :0]
            layer_boxes = inj_boxes[:, :0]
            layer_features = injected_capture["outs_dec"][:, 0, :0]
            slots = np.empty(0, np.int16)
        payload = {
            "mode": np.asarray(mode),
            "sample_token": np.asarray(token),
            "scene_token": np.asarray(scene),
            "frame_idx": np.asarray(frame_idx),
            "active": np.asarray(active),
            "injected_count": np.asarray(len(specs)),
            "retained_count": np.asarray(900 - len(specs)),
            "slots": slots,
            "identity": np.asarray([spec["identity"] for spec in specs]),
            "history_query": np.asarray(
                [spec["query"] for spec in specs], np.int16
            ),
            "history_score": np.asarray(
                [spec["history_score"] for spec in specs], np.float32
            ),
            "age": np.asarray([spec["age"] for spec in specs], np.int16),
            "initial_center": np.asarray(
                [spec["center"] for spec in specs], np.float32
            ).reshape(-1, 3),
            "oracle_gt_center": np.asarray(
                [spec.get("oracle_gt_center", [np.nan] * 3) for spec in specs],
                np.float32,
            ).reshape(-1, 3),
            "q0_logits": q0_logits.detach().cpu().numpy().astype(np.float16),
            "q0_boxes": q0_boxes.detach().cpu().numpy().astype(np.float32),
            "injected_final_logits": injected["all_cls_scores"][-1, 0]
            .detach().cpu().numpy().astype(np.float16),
            "injected_final_boxes": injected["all_bbox_preds"][-1, 0]
            .detach().cpu().numpy().astype(np.float32),
            "layer_logits": layer_logits.detach().cpu().numpy().astype(np.float16),
            "layer_boxes": layer_boxes.detach().cpu().numpy().astype(np.float32),
            "layer_features": layer_features.detach().cpu().numpy().astype(np.float16),
            "q0_cpu_ms": np.asarray(q0_cpu_ms),
            "q0_gpu_ms": np.asarray(q0_gpu_ms),
            "q0_decoder_gpu_ms": np.asarray(
                q0_capture.get("decoder_gpu_ms", float("nan"))
            ),
            "injected_cpu_ms": np.asarray(injected_cpu_ms),
            "injected_gpu_ms": np.asarray(injected_gpu_ms),
            "injected_decoder_gpu_ms": np.asarray(
                injected_decoder_gpu_ms
            ),
            "memory_restore_diff": np.asarray(memory_restore_diff),
            "injection_memory_would_differ": np.asarray(
                _state_diff(injection_after, q0_after)
            ),
            "gpu_allocated_mb": np.asarray(
                torch.cuda.memory_allocated() / (1024 ** 2)
                if memory_center.is_cuda else 0.0
            ),
            "gpu_peak_allocated_mb": np.asarray(
                torch.cuda.max_memory_allocated() / (1024 ** 2)
                if memory_center.is_cuda else 0.0
            ),
            "gpu_reserved_mb": np.asarray(
                torch.cuda.memory_reserved() / (1024 ** 2)
                if memory_center.is_cuda else 0.0
            ),
            "decoder_tgt_shape": np.asarray(
                injected_capture["decoder_tgt_shape"], np.int16
            ),
        }
        np.savez_compressed(output / f"{token}.npz", **payload)
        return injected

    StreamPETRHead.forward = injected_forward
    StreamPETRHead._active_recovery_installed = True


_install()
