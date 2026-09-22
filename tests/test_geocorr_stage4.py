from types import SimpleNamespace

import pytest
import torch

from evaluation.geocorr_stage4_evaluator import (
    aggregate_fault_summary,
    average_diagnostics,
    checkpoint_payload,
    condition_records,
    current_tokens,
    load_progress,
    normalize_prediction,
    recovery_diagnostics,
    save_progress,
    unique_records_by_current_token,
    validate_result_token_sets,
)
from training.geocorr_stage3c_trainer import (
    GeoCorrStage3CModel,
    Stage3CConfig,
    epoch_permutation,
)


def _record(condition, token, scene="scene-a"):
    return {
        "raw_condition": condition,
        "current_sample_token": token,
        "previous_sample_token": "prev-" + token,
        "scene_token": scene,
    }


def test_condition_filter_happens_before_per_condition_limit():
    records = [
        _record("Dirt/0.1_dirt", "a"),
        _record("Water-blur/0.1_water-blur", "b"),
        _record("Dirt/0.1_dirt", "c"),
    ]
    grouped = condition_records(records, ["Dirt/0.1_dirt"], max_pairs=1)
    assert list(grouped) == ["Dirt/0.1_dirt"]
    assert current_tokens(grouped["Dirt/0.1_dirt"]) == ["a"]


def test_condition_duplicate_sample_protection():
    records = [_record("Dirt/0.1_dirt", "a"), _record("Dirt/0.1_dirt", "a")]
    with pytest.raises(ValueError, match="duplicate"):
        condition_records(records)


def test_unique_clean_records_deduplicate_across_conditions():
    grouped = {
        "Dirt/0.1_dirt": [_record("Dirt/0.1_dirt", "a")],
        "Water-blur/0.1_water-blur": [_record("Water-blur/0.1_water-blur", "a")],
    }
    records = unique_records_by_current_token(grouped)
    assert len(records) == 1
    assert records[0]["current_sample_token"] == "a"


def test_epoch_shuffle_is_reproducible_complete_and_changes_between_epochs():
    first = epoch_permutation(64, seed=2026, epoch=0, shuffle=True)
    repeat = epoch_permutation(64, seed=2026, epoch=0, shuffle=True)
    second = epoch_permutation(64, seed=2026, epoch=1, shuffle=True)
    assert first == repeat
    assert sorted(first) == list(range(64))
    assert len(set(first)) == 64
    assert second != first
    assert epoch_permutation(4, 1, 0, False) == [0, 1, 2, 3]


def test_resume_cursor_reconstructs_same_shuffled_suffix():
    order = epoch_permutation(20, seed=7, epoch=3, shuffle=True)
    cursor = 9
    rebuilt = epoch_permutation(20, seed=7, epoch=3, shuffle=True)
    assert rebuilt[cursor:] == order[cursor:]


def test_inference_forward_uses_dirty_and_history_without_clean_teacher():
    config = Stage3CConfig(
        feature_dim=4,
        candidate_offsets=((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)),
        top_k=1,
        temperature=0.2,
        lambda_corr=1.0,
        lambda_rec=1.0,
    )
    model = GeoCorrStage3CModel(config)
    dirty = torch.randn(1, 1, 3, 1, 4)
    history = torch.randn(1, 1, 3, 1, 1, 4)
    current_valid = torch.ones(1, 1, 3, 1, dtype=torch.bool)
    history_valid = torch.ones(1, 1, 3, 1, 1, dtype=torch.bool)
    fpn = torch.randn(1, 1, 4, 3, 3)
    coords = torch.tensor([[[[1.0, 1.0]]]])
    center_valid = torch.ones(1, 1, 1, dtype=torch.bool)
    output = model.inference_forward(
        dirty, history, current_valid, history_valid, fpn, coords, center_valid
    )
    assert output.p_teacher is None
    assert output.current_clean_fpn is None
    assert output.recovered_current_fpn.shape == fpn.shape
    assert torch.isfinite(output.recovered_current_fpn).all()


def test_baseline_geocorr_token_contract_is_order_strict():
    validate_result_token_sets(["a", "b"], ["a", "b"])
    with pytest.raises(ValueError, match="orders differ"):
        validate_result_token_sets(["a", "b"], ["b", "a"])


def test_prediction_normalization_preserves_outer_pts_bbox_contract():
    inner = {
        "boxes_3d": object(),
        "scores_3d": torch.tensor([1.0]),
        "labels_3d": torch.tensor([0]),
    }
    assert normalize_prediction(inner)["pts_bbox"] is inner
    assert normalize_prediction({"pts_bbox": inner})["pts_bbox"] is inner


def test_fault_aggregation_keeps_clean_out_and_reports_delta():
    metrics = {
        "Dirt/0.1_dirt": {
            "baseline": {"mean_ap": 0.2, "nd_score": 0.3},
            "geocorr": {"mean_ap": 0.25, "nd_score": 0.32},
        },
        "Dirt/0.2_dirt": {
            "baseline": {"mean_ap": 0.1, "nd_score": 0.2},
            "geocorr": {"mean_ap": 0.12, "nd_score": 0.25},
        },
        "Water-blur/0.1_water-blur": {
            "baseline": {"mean_ap": 0.3, "nd_score": 0.4},
            "geocorr": {"mean_ap": 0.33, "nd_score": 0.44},
        },
    }
    summary = aggregate_fault_summary(metrics)
    assert summary["Average Dirt"]["baseline"]["mAP"] == pytest.approx(0.15)
    assert summary["Average Fault"]["delta"]["NDS"] > 0
    assert "clean" not in summary


def test_progress_resume_validates_condition_tokens_and_cursor(tmp_path):
    path = tmp_path / "progress.pth"
    save_progress(path, {
        "condition": "Dirt/0.1_dirt",
        "tokens": ["a", "b"],
        "next_index": 1,
        "baseline_results": [{"x": 1}],
    })
    loaded = load_progress(path, "Dirt/0.1_dirt", ["a", "b"])
    assert loaded["next_index"] == 1
    with pytest.raises(RuntimeError, match="token order"):
        load_progress(path, "Dirt/0.1_dirt", ["b", "a"])


def test_checkpoint_detector_checksum_validation(tmp_path):
    path = tmp_path / "ckpt.pth"
    torch.save({
        "geocorr": {},
        "config": {},
        "detector_checksum": "abc",
        "step": 10,
        "epoch": 2,
    }, path)
    assert checkpoint_payload(path, "abc")["step"] == 10
    with pytest.raises(RuntimeError, match="checksum"):
        checkpoint_payload(path, "def")


def test_recovery_diagnostics_and_average_are_finite():
    dirty = torch.zeros(1, 1, 2, 2, 2)
    recovered = dirty.clone()
    recovered[..., 0, 0] = 1.0
    recovery = SimpleNamespace(
        query_valid=torch.tensor([[[True]]]),
        confidence=torch.tensor([[[[0.5]]]]),
        delta_q=torch.tensor([[[[1.0, 0.0]]]]),
        recovered_current_fpn=recovered,
        current_dirty_fpn=dirty,
    )
    values = recovery_diagnostics(recovery)
    assert values["confidence_mean"] == pytest.approx(0.5)
    assert values["g_delta_q_l2"] == pytest.approx(0.5)
    assert average_diagnostics([values, values]) == values
