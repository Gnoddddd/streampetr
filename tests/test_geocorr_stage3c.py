import json

import numpy as np
import pytest
import torch
from torch import nn

from evaluation.streampetr_runtime import (
    assert_streampetr_model_batch,
    prepare_streampetr_model_batch,
)
from training.geocorr_stage3c_trainer import (
    GeoCorrStage3CModel,
    JsonlLogger,
    Stage3CConfig,
    _assert_nested_equal,
    assert_finite_gradients,
    assert_finite_tensor,
    assert_synchronized_preprocessing,
    build_optimizer,
    compose_losses,
    deterministic_train_pipeline,
    freeze_detector,
    frozen_detector_parameter_checksum,
    limit_pairs,
    load_checkpoint,
    reached_max_steps,
    save_checkpoint,
    snapshot_parameters,
    streampetr_detection_losses,
)


class ToyFrozenDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.5))

    def forward_pts_train(
        self, gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels,
        img_metas, centers2d, depths, requires_grad, return_losses, **data
    ):
        del gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels
        del img_metas, centers2d, depths
        assert requires_grad and return_losses
        feature = data["img_feats"]
        return {
            "loss_bbox": (feature * self.scale).square().mean(),
            "diagnostic": feature.detach().mean(),
        }


class DataWrapper:
    """Lightweight stand-in for MMCV DataContainer's ``.data`` payload."""

    def __init__(self, data):
        self.data = data


def _model_batch_source(use_numpy=False, wrapped=False):
    def value(index):
        tensor = torch.full((4, 4), float(index + 1))
        return tensor.numpy() if use_numpy else tensor

    cameras = [value(index) for index in range(6)]
    metadata = {"scene_token": "scene", "img_shape": [(2, 2, 3)] * 6}
    batch = {
        "img": [torch.full((3, 2, 2), float(index)) for index in range(6)],
        "intrinsics": list(cameras),
        "extrinsics": list(cameras),
        "lidar2img": list(cameras),
        "img_timestamp": [torch.tensor(float(index)) for index in range(6)],
        "ego_pose": torch.eye(4),
        "ego_pose_inv": torch.eye(4),
        "timestamp": torch.tensor(1.0),
        "prev_exists": torch.tensor(0.0),
        "img_metas": [[metadata]],
        "gt_bboxes_3d": [object()],
        "gt_labels_3d": [torch.tensor([1])],
        "gt_bboxes": [[torch.zeros(0, 4)]],
        "gt_labels": [[torch.zeros(0, dtype=torch.long)]],
        "centers2d": [[torch.zeros(0, 2)]],
        "depths": [[torch.zeros(0)]],
    }
    if wrapped:
        for name in ("intrinsics", "extrinsics", "lidar2img", "ego_pose", "ego_pose_inv", "img_metas"):
            batch[name] = DataWrapper(batch[name])
    return batch


def test_model_batch_packs_six_tensor_cameras_and_preserves_gt_types():
    source = _model_batch_source()
    gt_boxes = source["gt_bboxes_3d"]
    packed = prepare_streampetr_model_batch(source, torch.device("cpu"))

    for name in ("intrinsics", "extrinsics", "lidar2img"):
        assert isinstance(packed[name], torch.Tensor)
        assert packed[name].shape == (1, 6, 4, 4)
        assert packed[name].device.type == "cpu"
        assert torch.is_floating_point(packed[name])
    assert packed["img"].shape == (1, 6, 3, 2, 2)
    assert packed["img_metas"] == source["img_metas"][0]
    assert packed["gt_bboxes_3d"] is gt_boxes


def test_model_batch_packs_six_ndarray_cameras_and_data_container_payloads():
    packed = prepare_streampetr_model_batch(
        _model_batch_source(use_numpy=True, wrapped=True), torch.device("cpu")
    )

    assert packed["intrinsics"].shape == (1, 6, 4, 4)
    assert packed["extrinsics"].shape == (1, 6, 4, 4)
    assert packed["lidar2img"].shape == (1, 6, 4, 4)
    assert isinstance(packed["img_metas"], list)
    assert isinstance(packed["img_metas"][0], dict)


def test_model_batch_preserves_already_batched_stage3b_compatible_tensors():
    source = _model_batch_source()
    for name in ("img", "intrinsics", "extrinsics", "lidar2img"):
        source[name] = torch.stack(source[name], dim=0).unsqueeze(0)
    packed = prepare_streampetr_model_batch(source, torch.device("cpu"))

    assert packed["img"].shape == (1, 6, 3, 2, 2)
    assert packed["intrinsics"].shape == (1, 6, 4, 4)
    assert packed["intrinsics"].data_ptr() == source["intrinsics"].data_ptr()


def test_model_batch_uses_same_packing_for_previous_clean_and_dirty_paths():
    sources = [_model_batch_source(wrapped=True) for _ in range(3)]
    packed = [
        prepare_streampetr_model_batch(source, torch.device("cpu"))
        for source in sources
    ]

    assert all(item["lidar2img"].shape == (1, 6, 4, 4) for item in packed)
    assert all(isinstance(item["img_metas"], list) for item in packed)
    assert all(item["gt_labels_3d"] is source["gt_labels_3d"]
               for item, source in zip(packed, sources))


def test_model_batch_rejects_malformed_camera_structure_with_contract_diagnostics():
    source = _model_batch_source()
    source["intrinsics"] = source["intrinsics"][:5]
    with pytest.raises(RuntimeError) as raised:
        prepare_streampetr_model_batch(source, torch.device("cpu"))
    message = str(raised.value)
    assert "intrinsics" in message
    assert "type=list" in message
    assert "nested=list[Tensor]" in message


def test_model_batch_contract_assertion_describes_type_shape_device_and_dtype():
    source = _model_batch_source()
    source["intrinsics"] = [torch.eye(4)] * 6
    with pytest.raises(RuntimeError) as raised:
        assert_streampetr_model_batch(source, torch.device("cpu"))
    message = str(raised.value)
    assert "intrinsics" in message
    assert "type=list" in message
    assert "shape=None" in message
    assert "device=n/a" in message
    assert "dtype=n/a" in message


def test_detection_loss_batch_uses_packed_geometry_without_breaking_gt():
    source = _model_batch_source(wrapped=True)
    packed = prepare_streampetr_model_batch(source, torch.device("cpu"))
    losses = streampetr_detection_losses(
        ToyFrozenDetector(), torch.ones(1, 6, 4, 2, 2), packed
    )

    assert "loss_bbox" in losses
    assert packed["gt_bboxes_3d"] is source["gt_bboxes_3d"]


def _config(lambda_corr=1.0, lambda_rec=1.0):
    return Stage3CConfig(
        feature_dim=4,
        candidate_offsets=((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)),
        top_k=1,
        temperature=0.2,
        lambda_corr=lambda_corr,
        lambda_rec=lambda_rec,
    )


def _inputs(valid=True):
    dirty_center = torch.tensor([1.0, 0.2, -0.1, 0.3])
    clean_center = torch.tensor([0.8, 0.3, -0.2, 0.4])
    dirty = dirty_center.reshape(1, 1, 1, 1, 4).expand(1, 1, 3, 1, 4).clone()
    clean = clean_center.reshape(1, 1, 1, 1, 4).expand_as(dirty).clone()
    history = torch.tensor([
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        -1.0, 0.0, 0.0, 0.0,
    ]).reshape(1, 1, 3, 1, 1, 4)
    current_valid = torch.full((1, 1, 3, 1), valid, dtype=torch.bool)
    history_valid = torch.full((1, 1, 3, 1, 1), valid, dtype=torch.bool)
    fpn = torch.randn(1, 1, 4, 3, 3)
    coords = torch.tensor([[[[1.0, 1.0]]]])
    projected_valid = torch.full((1, 1, 1), valid, dtype=torch.bool)
    return clean, dirty, history, current_valid, history_valid, fpn, coords, projected_valid


def _forward(model, valid=True, require_teacher_grad=False):
    values = list(_inputs(valid))
    if require_teacher_grad:
        values[0].requires_grad_()
        values[2].requires_grad_()
        values[5].requires_grad_()
    output = model(*values)
    return output, values


def _batch():
    return {
        "gt_bboxes_3d": [object()], "gt_labels_3d": [torch.zeros(1)],
        "gt_bboxes": [[]], "gt_labels": [[]], "img_metas": [{}],
        "centers2d": [[]], "depths": [[]], "img": torch.zeros(1),
    }


def test_optimizer_contains_only_geocorr_parameters():
    detector = ToyFrozenDetector()
    model = GeoCorrStage3CModel(_config())
    freeze_detector(detector)
    optimizer = build_optimizer(detector, model, 1e-3)

    optimized = {id(value) for group in optimizer.param_groups for value in group["params"]}
    assert optimized == {id(value) for value in model.parameters() if value.requires_grad}
    assert not optimized & {id(value) for value in detector.parameters()}


def test_optimizer_hard_fails_if_detector_is_not_frozen():
    with pytest.raises(RuntimeError, match="not frozen"):
        build_optimizer(ToyFrozenDetector(), GeoCorrStage3CModel(_config()), 1e-3)


def test_clean_teacher_previous_history_and_dirty_backbone_are_detached():
    model = GeoCorrStage3CModel(_config())
    output, values = _forward(model, require_teacher_grad=True)
    output.loss_corr.backward(retain_graph=True)

    assert values[0].grad is None
    assert values[2].grad is None
    assert values[5].grad is None
    assert output.recovery.q_clean.requires_grad is False
    assert output.recovery.previous_historical_features.requires_grad is False


def test_official_detection_loss_has_recovered_fpn_gradient():
    detector = ToyFrozenDetector()
    freeze_detector(detector)
    model = GeoCorrStage3CModel(_config())
    output, _ = _forward(model)
    recovered = output.recovery.recovered_current_fpn
    recovered.retain_grad()
    losses = streampetr_detection_losses(detector, recovered, _batch())
    composed = compose_losses(losses, output.loss_corr, output.loss_rec, 1.0, 1.0)
    composed.total.backward()

    assert recovered.grad is not None
    assert torch.count_nonzero(recovered.grad) > 0
    assert detector.scale.grad is None
    assert any(
        value.grad is not None and torch.count_nonzero(value.grad) > 0
        for value in model.recovery.recovery.parameters()
    )
    assert any(
        value.grad is not None and torch.count_nonzero(value.grad) > 0
        for value in model.correlation.adapter.parameters()
    )


def test_zero_init_first_backward_and_optimizer_step_open_recovery():
    detector = ToyFrozenDetector()
    freeze_detector(detector)
    model = GeoCorrStage3CModel(_config())
    optimizer = build_optimizer(detector, model, 1e-2)
    before_detector = frozen_detector_parameter_checksum(detector)
    before = snapshot_parameters(model.recovery.recovery)
    output, _ = _forward(model)
    assert torch.count_nonzero(output.recovery.delta_q) == 0
    losses = compose_losses(
        streampetr_detection_losses(
            detector, output.recovery.recovered_current_fpn, _batch()
        ), output.loss_corr, output.loss_rec, 1.0, 1.0,
    )
    losses.total.backward()
    assert_finite_gradients(model, "GeoCorr")
    optimizer.step()

    assert frozen_detector_parameter_checksum(detector) == before_detector
    last_weight = model.recovery.recovery.layers[-1].weight
    assert torch.count_nonzero(last_weight) > 0
    assert any(
        not torch.equal(value.detach(), before[name])
        for name, value in model.recovery.recovery.named_parameters()
    )


def test_loss_composition_preserves_official_components():
    det = torch.tensor(2.0, requires_grad=True)
    corr = torch.tensor(3.0, requires_grad=True)
    rec = torch.tensor(5.0, requires_grad=True)
    result = compose_losses(
        {"loss_cls": det, "accuracy": torch.tensor(0.9)}, corr, rec, 0.1, 0.2
    )
    assert torch.allclose(result.total, torch.tensor(3.3))
    assert set(result.detection_components) == {"loss_cls", "accuracy"}


def test_invalid_and_all_invalid_queries_are_numerically_stable():
    model = GeoCorrStage3CModel(_config())
    output, _ = _forward(model, valid=False)

    assert not output.recovery.query_valid.any()
    assert output.loss_corr.item() == 0.0
    assert output.loss_rec.item() == 0.0
    assert torch.count_nonzero(output.recovery.confidence) == 0
    assert torch.isfinite(output.recovery.recovered_current_fpn).all()


def test_deterministic_pipeline_keeps_gt_transform_and_removes_random_geometry():
    pipeline = [
        {"type": "ResizeCropFlipRotImage", "data_aug_conf": {
            "H": 900, "W": 1600, "final_dim": (256, 704),
            "resize_lim": (0.38, 0.55), "bot_pct_lim": (0.0, 0.0),
            "rot_lim": (0.0, 0.0), "rand_flip": True,
        }, "training": True},
        {"type": "GlobalRotScaleTransImage", "rot_range": (-1.0, 1.0),
         "scale_ratio_range": (0.9, 1.1), "translation_std": (1, 1, 1)},
    ]
    result = deterministic_train_pipeline(pipeline)

    image, global_transform = result
    assert image["training"] is True
    assert image["data_aug_conf"]["resize_lim"][0] == image["data_aug_conf"]["resize_lim"][1]
    assert image["data_aug_conf"]["rand_flip"] is False
    assert global_transform["rot_range"] == (0.0, 0.0)
    assert global_transform["scale_ratio_range"] == (1.0, 1.0)
    assert global_transform["translation_std"] == (0.0, 0.0, 0.0)


def test_paired_preprocessing_contract_checks_geometry_not_pixels():
    geometry = {
        "lidar2img": torch.eye(4).reshape(1, 4, 4),
        "intrinsics": torch.eye(4).reshape(1, 4, 4),
        "extrinsics": torch.eye(4).reshape(1, 4, 4),
        "ego_pose": torch.eye(4), "ego_pose_inv": torch.eye(4),
    }
    clean = dict(geometry, img=torch.zeros(1, 1, 3, 2, 2))
    dirty = dict(geometry, img=torch.ones(1, 1, 3, 2, 2))
    assert_synchronized_preprocessing(clean, dirty)
    dirty["lidar2img"] = torch.zeros(1, 4, 4)
    with pytest.raises(RuntimeError, match="lidar2img"):
        assert_synchronized_preprocessing(clean, dirty)


def test_nested_preprocessing_tensor_and_numpy_leaves_allow_float_tolerance():
    _assert_nested_equal(
        "lidar2img", torch.tensor([[1.0, 2.0]]), torch.tensor([[1.0, 2.0000005]])
    )
    _assert_nested_equal(
        "intrinsics", np.array([[1.0, 2.0]], dtype=np.float32),
        np.array([[1.0, 2.0000005]], dtype=np.float32),
    )
    _assert_nested_equal(
        "ego_pose", torch.tensor([1, 2], dtype=torch.int64),
        torch.tensor([1, 2], dtype=torch.int64),
    )
    _assert_nested_equal(
        "extrinsics", np.eye(2, dtype=np.float32), torch.eye(2, dtype=torch.float32)
    )


def test_nested_preprocessing_recurses_all_six_cameras_and_nested_containers():
    camera_values = [torch.full((2, 2), float(index)) for index in range(6)]
    clean = DataWrapper({
        "cameras": (camera_values, [np.eye(2, dtype=np.float32)]),
        "name": "six-camera-batch",
    })
    dirty = DataWrapper({
        "cameras": (list(camera_values), [np.eye(2, dtype=np.float32)]),
        "name": "six-camera-batch",
    })

    _assert_nested_equal("lidar2img", clean, dirty)

    dirty.data["cameras"][0][5] = torch.full((2, 2), 99.0)
    with pytest.raises(RuntimeError) as raised:
        _assert_nested_equal("lidar2img", clean, dirty)
    message = str(raised.value)
    assert "lidar2img['cameras'][0][5]" in message
    assert "clean type=" in message
    assert "dirty type=" in message
    assert "clean shape=(2, 2)" in message
    assert "max abs diff=" in message


def test_nested_preprocessing_handles_singleton_batch_wrapper_without_squeeze():
    clean = [DataWrapper([torch.eye(4)])]
    dirty = [DataWrapper([torch.eye(4)])]

    _assert_nested_equal("ego_pose", clean, dirty)


def test_synchronized_preprocessing_accepts_wrapped_six_camera_metadata():
    cameras = [torch.eye(4) for _ in range(6)]
    geometry = {
        "lidar2img": DataWrapper([cameras]),
        "intrinsics": DataWrapper(tuple(cameras)),
        "extrinsics": DataWrapper({"views": cameras}),
        "ego_pose": DataWrapper([torch.eye(4)]),
        "ego_pose_inv": DataWrapper([torch.eye(4)]),
    }
    clean = dict(geometry, img=DataWrapper(torch.zeros(1, 6, 3, 2, 2)))
    dirty = dict(geometry, img=DataWrapper(torch.ones(1, 6, 3, 2, 2)))

    assert_synchronized_preprocessing(clean, dirty)


@pytest.mark.parametrize(
    ("clean", "dirty", "reason"),
    (
        (torch.zeros(2, 2), torch.zeros(3, 2), "shape mismatch"),
        (torch.tensor([1, 2]), torch.tensor([1, 3]), "numeric mismatch"),
        ([torch.zeros(1)], [torch.zeros(1), torch.zeros(1)], "sequence length mismatch"),
    ),
)
def test_nested_preprocessing_mismatches_hard_fail_with_path(clean, dirty, reason):
    with pytest.raises(RuntimeError) as raised:
        _assert_nested_equal("extrinsics", clean, dirty)
    message = str(raised.value)
    assert reason in message
    assert "extrinsics" in message
    assert "clean type=" in message
    assert "dirty type=" in message


def test_nested_preprocessing_requires_exact_dictionary_keys_and_integer_values():
    with pytest.raises(RuntimeError, match="dictionary key mismatch"):
        _assert_nested_equal("intrinsics", {"CAM_FRONT": torch.eye(3)}, {})
    with pytest.raises(RuntimeError, match="numeric mismatch"):
        _assert_nested_equal(
            "intrinsics", np.array([1, 2], dtype=np.int64),
            np.array([1, 3], dtype=np.int64),
        )


def test_gt_is_only_accepted_by_detection_boundary():
    model = GeoCorrStage3CModel(_config())
    assert "gt" not in model.forward.__code__.co_varnames
    with pytest.raises(KeyError, match="GT fields"):
        streampetr_detection_losses(
            ToyFrozenDetector(), torch.zeros(1, 1, 4, 2, 2), {}
        )


def test_checkpoint_round_trip_restores_state_step_epoch_and_pair(tmp_path):
    detector = ToyFrozenDetector()
    freeze_detector(detector)
    checksum = frozen_detector_parameter_checksum(detector)
    model = GeoCorrStage3CModel(_config())
    optimizer = build_optimizer(detector, model, 1e-3)
    path = tmp_path / "stage3c.pth"
    save_checkpoint(path, model, optimizer, 7, 2, _config(), checksum, pair_index=4)
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        for value in model.parameters():
            value.add_(1)
    assert load_checkpoint(path, model, optimizer, checksum) == (7, 2, 4)
    assert all(torch.equal(model.state_dict()[name], value) for name, value in expected.items())


def test_checkpoint_rejects_different_frozen_detector(tmp_path):
    detector = ToyFrozenDetector()
    freeze_detector(detector)
    model = GeoCorrStage3CModel(_config())
    optimizer = build_optimizer(detector, model, 1e-3)
    path = tmp_path / "stage3c.pth"
    save_checkpoint(path, model, optimizer, 0, 0, _config(), "first")
    with pytest.raises(RuntimeError, match="different frozen detector"):
        load_checkpoint(path, model, optimizer, "second")


def test_max_pairs_and_max_steps_contracts():
    assert limit_pairs([1, 2, 3], 2) == [1, 2]
    assert reached_max_steps(4, 4)
    assert not reached_max_steps(3, 4)
    with pytest.raises(ValueError):
        limit_pairs([1], 0)
    with pytest.raises(ValueError):
        reached_max_steps(0, 0)


def test_nan_hard_fail():
    with pytest.raises(FloatingPointError, match="non-finite"):
        assert_finite_tensor("loss", torch.tensor(float("nan")))


def test_structured_jsonl_logging(tmp_path):
    path = tmp_path / "train.jsonl"
    JsonlLogger(path).log({
        "step": 1, "loss_total": 2.0, "confidence_mean": 0.1,
        "grad_norm_descriptor_adapter": 3.0,
    })
    record = json.loads(path.read_text().strip())
    assert record["step"] == 1
    assert record["loss_total"] == 2.0
    assert "confidence_mean" in record
    assert "grad_norm_descriptor_adapter" in record
