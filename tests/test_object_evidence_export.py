import torch
from torch import nn

from models.object_evidence.paired_training import (
    ObjectEvidenceObjective, ObjectEvidenceTrainingWrapper,
)
from models.object_evidence.privileged_geometry.cache import (
    LidarObjectEvidenceCache, save_cache,
)
from scripts.export_object_evidence_detector_only import detector_only_state_dict


def test_detector_only_export_loads_strict_and_is_bit_exact():
    torch.manual_seed(9)
    detector = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2)).eval()
    wrapper = ObjectEvidenceTrainingWrapper(
        detector, objective=ObjectEvidenceObjective(teacher_dim=7)
    ).eval()
    inputs = torch.randn(3, 4)
    expected = wrapper(inputs)
    state = detector_only_state_dict(wrapper.state_dict())
    assert not any("object_evidence" in key or "pg_projector" in key for key in state)
    vanilla = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2)).eval()
    vanilla.load_state_dict(state, strict=True)
    actual = vanilla(inputs)
    assert torch.equal(actual, expected)
    assert (actual - expected).abs().max().item() == 0


def test_raw_vanilla_state_is_unchanged():
    model = nn.Linear(2, 1)
    state = detector_only_state_dict(model.state_dict())
    clone = nn.Linear(2, 1)
    clone.load_state_dict(state, strict=True)


def test_bevdepth_lightning_model_prefix_is_unwrapped():
    model = nn.Linear(2, 1)
    prefixed = {f"model.{key}": value for key, value in model.state_dict().items()}
    prefixed["teacher.projector.weight"] = torch.randn(1)
    state = detector_only_state_dict(prefixed, detector_prefixes=("detector.", "model."))
    nn.Linear(2, 1).load_state_dict(state, strict=True)


def test_train_teacher_cache_round_trip_is_detached_float16(tmp_path):
    manifest = dict(
        teacher_config_sha256="a", teacher_checkpoint_sha256="b",
        feature_layer="head.input", feature_dim=3, split="train",
        scene_count=1, sample_count=1,
    )
    save_cache(tmp_path, [dict(
        sample_token="sample", annotation_token="annotation", instance_token="instance",
        label=2, num_lidar_pts=10, teacher_token=torch.tensor([1.0, 2.0, 3.0]),
    )], manifest)
    cache = LidarObjectEvidenceCache(tmp_path)
    tokens, points, valid = cache.teacher_tokens([("sample", "annotation")])
    assert tokens.tolist() == [[1, 2, 3]] and points.tolist() == [10]
    assert valid.tolist() == [True] and not tokens.requires_grad
    assert torch.load(tmp_path / "records.pt")[0]["teacher_token"].dtype == torch.float16
