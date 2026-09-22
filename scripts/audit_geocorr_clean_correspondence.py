"""Audit one real consecutive clean nuScenes pair with frozen StreamPETR."""

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
STREAM_ROOT = ROOT / "repos/StreamPETR"
for path in (ROOT, STREAM_ROOT, STREAM_ROOT / "mmdetection3d"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.geocorr_correspondence_audit import (  # noqa: E402
    confidence_group_indices,
    correspondence_audit_metrics,
    select_object_group,
    shuffle_history_objects,
    summarize_correspondence_audit,
    summarize_valid_view_groups,
)
from configs.geocorr_recovery.stage1_infrastructure import candidate_offsets  # noqa: E402
from models.adapters import from_streampetr_result  # noqa: E402
from models.geocorr_recovery import (  # noqa: E402
    GeometryCandidateSampler,
    find_center_candidate_index,
    masked_softmax,
    sample_candidate_features,
    sample_history_candidate_features,
)


CAMERA_FROM_FILENAME = "__CAM_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/full_nuscenes/stream_petr_r50_90e_clean_val.py",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/official/stream_petr_r50_flash_704_bs2_seq_90e.pth",
    )
    parser.add_argument("--data-root", default="data/nuscenes")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--output-json")
    return parser.parse_args()


def _data_value(data: Dict[str, object], key: str) -> object:
    """Unwrap the single-augmentation, single-batch DataContainer value."""
    value = data[key][0].data[0]
    return value[0] if isinstance(value, list) else value


def _image_shapes(meta: Dict[str, object], key: str) -> torch.Tensor:
    return torch.tensor([[shape[0], shape[1]] for shape in meta[key]], dtype=torch.long)


def _camera_names(meta: Dict[str, object]) -> List[str]:
    names = []
    for filename in meta["filename"]:
        stem = Path(filename).stem
        marker = stem.find(CAMERA_FROM_FILENAME)
        names.append(stem[marker + 2 :].split("__", 1)[0] if marker >= 0 else "UNKNOWN")
    return names


def _clean_logits(current_tokens: torch.Tensor, history_tokens: torch.Tensor) -> torch.Tensor:
    center = find_center_candidate_index(candidate_offsets)
    query = F.normalize(current_tokens[:, :, center].detach(), dim=-1)
    keys = F.normalize(history_tokens.detach(), dim=-1)
    return torch.einsum("bnvc,bnjtwc->bnvjtw", query, keys)


def _correspondence(
    current_tokens: torch.Tensor,
    history_tokens: torch.Tensor,
    current_valid: torch.Tensor,
    history_valid: torch.Tensor,
    object_valid: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    center = find_center_candidate_index(candidate_offsets)
    query_valid = current_valid[:, :, center] & object_valid[:, :, None]
    valid = query_valid[:, :, :, None, None, None] & history_valid[:, :, None]
    logits = _clean_logits(current_tokens, history_tokens)
    probabilities, _ = masked_softmax(logits, valid, temperature)
    return probabilities, valid, correspondence_audit_metrics(
        probabilities, valid, candidate_offsets
    )


def _find_pair(data_infos: List[Dict[str, object]]) -> Tuple[int, int]:
    for current_index in range(1, len(data_infos)):
        previous = data_infos[current_index - 1]
        current = data_infos[current_index]
        if (
            current.get("scene_token") == previous.get("scene_token")
            and current.get("prev") == previous.get("token")
        ):
            return current_index - 1, current_index
    raise RuntimeError("no consecutive same-scene frame pair found")


def _audit_object_group(
    indices: torch.Tensor,
    scores: torch.Tensor,
    current_tokens: torch.Tensor,
    history_tokens: torch.Tensor,
    current_valid: torch.Tensor,
    history_valid: torch.Tensor,
    temperature: float,
) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    """Audit one aligned confidence group, including a group-local shuffle."""
    group_current = select_object_group(current_tokens, indices)
    group_history = select_object_group(history_tokens, indices)
    group_current_valid = select_object_group(current_valid, indices)
    group_history_valid = select_object_group(history_valid, indices)
    object_valid = torch.ones(
        group_current.shape[:2], dtype=torch.bool, device=group_current.device
    )
    _, _, metrics = _correspondence(
        group_current,
        group_history,
        group_current_valid,
        group_history_valid,
        object_valid,
        temperature,
    )
    shuffled_tokens, shuffled_valid, _, comparable = shuffle_history_objects(
        group_history, group_history_valid, object_valid
    )
    _, _, shuffled_metrics = _correspondence(
        group_current,
        shuffled_tokens,
        group_current_valid,
        shuffled_valid,
        object_valid,
        temperature,
    )
    selected_scores = scores.index_select(0, indices.to(scores.device))
    return summarize_correspondence_audit(
        metrics,
        candidate_offsets,
        shuffled_metrics,
        comparable,
        num_frame_pairs=1,
        scores=selected_scores,
    ), metrics


def run(args: argparse.Namespace) -> Dict[str, object]:
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    importlib.import_module("projects.mmdet3d_plugin")
    config_path = (ROOT / args.config).resolve()
    checkpoint_path = (ROOT / args.checkpoint).resolve()
    data_root = (ROOT / args.data_root).resolve()
    annotation = data_root / "nuscenes2d_temporal_infos_val.pkl"
    for required in (config_path, checkpoint_path, data_root, annotation):
        if not required.exists():
            raise FileNotFoundError(str(required))

    cfg = Config.fromfile(str(config_path))
    cfg.data.test.data_root = str(data_root) + "/"
    cfg.data.test.ann_file = str(annotation)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    previous_index, current_index = _find_pair(dataset.data_infos)
    if current_index != previous_index + 1:
        raise RuntimeError("audit requires adjacent dataset indices")
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    detector = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16") is not None:
        wrap_fp16_model(detector)
    load_checkpoint(detector, str(checkpoint_path), map_location="cpu")
    detector.cuda().eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)

    captured: List[torch.Tensor] = []
    original_extract = detector.extract_img_feat

    def capture_features(*feature_args, **feature_kwargs):
        features = original_extract(*feature_args, **feature_kwargs)
        captured.append(features.detach().clone())
        return features

    detector.extract_img_feat = capture_features
    parallel = MMDataParallel(detector, device_ids=[0])
    pair_data = []
    pair_results = []
    with torch.no_grad():
        for index, data in enumerate(loader):
            if index < previous_index:
                continue
            pair_data.append(data)
            pair_results.append(parallel(return_loss=False, rescale=True, **data))
            if index == current_index:
                break
    if len(pair_results) != 2 or len(captured) != 2:
        raise RuntimeError("expected exactly two inference results and feature tensors")

    previous_info = dataset.data_infos[previous_index]
    current_info = dataset.data_infos[current_index]
    previous_meta = _data_value(pair_data[0], "img_metas")
    current_meta = _data_value(pair_data[1], "img_metas")
    previous_prediction = from_streampetr_result(
        pair_results[0][0], float(previous_info["timestamp"]) / 1e6
    )
    previous_pose = _data_value(pair_data[0], "ego_pose").float()
    current_pose_inverse = _data_value(pair_data[1], "ego_pose_inv").float()
    current_from_previous = current_pose_inverse @ previous_pose
    delta_t = float(current_info["timestamp"] - previous_info["timestamp"]) / 1e6

    geometry_sampler = GeometryCandidateSampler(candidate_offsets)
    previous_features = captured[0]
    current_features = captured[1]
    geometry = geometry_sampler.forward_with_history(
        previous_prediction,
        current_from_previous,
        delta_t,
        _data_value(pair_data[1], "lidar2img").float(),
        _image_shapes(current_meta, "img_shape"),
        tuple(current_features.shape[-2:]),
        _data_value(pair_data[0], "lidar2img").float(),
        _image_shapes(previous_meta, "img_shape"),
        tuple(previous_features.shape[-2:]),
        current_padded_image_shapes=_image_shapes(current_meta, "pad_shape"),
        previous_padded_image_shapes=_image_shapes(previous_meta, "pad_shape"),
    )
    device = current_features.device
    current_grid = geometry["current_grid_coords"].unsqueeze(0).to(device)
    current_valid = geometry["current_valid_mask"].unsqueeze(0).to(device)
    history_grid = geometry["previous_grid_coords"].unsqueeze(0).to(device)
    history_valid = geometry["previous_valid_mask"].unsqueeze(0).to(device)
    current_tokens = sample_candidate_features(current_features, current_grid, current_valid)
    history_tokens = sample_history_candidate_features(
        previous_features.unsqueeze(1), history_grid, history_valid
    )
    scores = previous_prediction.score.detach().flatten()
    if scores.numel() != current_tokens.shape[1]:
        raise RuntimeError("prediction scores and sampled objects are not aligned")
    group_indices = confidence_group_indices(scores)
    all_summary, all_metrics = _audit_object_group(
        group_indices["all"], scores, current_tokens, history_tokens,
        current_valid, history_valid, args.temperature,
    )
    summary = {"all": all_summary}
    for name in ("top25", "top50", "top100"):
        summary[name], _ = _audit_object_group(
            group_indices[name], scores, current_tokens, history_tokens,
            current_valid, history_valid, args.temperature,
        )
    summary["quartiles"] = {}
    for name, indices in group_indices["quartiles"].items():
        summary["quartiles"][name], _ = _audit_object_group(
            indices, scores, current_tokens, history_tokens,
            current_valid, history_valid, args.temperature,
        )
    summary["valid_view_groups"] = summarize_valid_view_groups(
        all_metrics, candidate_offsets
    )
    summary["metadata"] = dict(
        scene_token=current_info["scene_token"],
        previous_sample_token=previous_info["token"],
        sample_token=current_info["token"],
        same_scene=current_info["scene_token"] == previous_info["scene_token"],
        previous_link_correct=current_info["prev"] == previous_info["token"],
        delta_t_seconds=delta_t,
        prediction_object_count=int(previous_prediction.center_3d.shape[0]),
        valid_projected_object_count=int(
            all_metrics["query_valid"].any(dim=2).sum().item()
        ),
        camera_order=_camera_names(current_meta),
        feature_shapes={
            "current_fpn": list(current_features.shape),
            "history_fpn": list(previous_features.shape),
            "current_query": [
                current_tokens.shape[0], current_tokens.shape[1],
                current_tokens.shape[3], current_tokens.shape[4],
            ],
            "history_keys": list(history_tokens.shape),
            "probability": [
                current_tokens.shape[0], current_tokens.shape[1],
                current_tokens.shape[3], history_tokens.shape[2],
                history_tokens.shape[3], history_tokens.shape[4],
            ],
            "position_marginal": list(all_metrics["position_marginal"].shape),
            "view_marginal": list(all_metrics["view_marginal"].shape),
        },
        gt_boxes_used=False,
        tracking_ids_used=False,
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    return summary


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
