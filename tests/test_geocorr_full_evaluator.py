import gzip
import json
import math

import pytest

from evaluation.geocorr_full_evaluator import (
    EvaluationStore,
    GROUP_NAMES,
    VALID_VIEW_NAMES,
    ResultAccumulator,
    atomic_write_json,
    build_frame_and_pair_plan,
    cache_rollover,
    descriptive_statistics,
    iter_jsonl_gz,
    resume_frame_start,
)


def _info(token, scene, timestamp, previous=""):
    return {
        "token": token,
        "scene_token": scene,
        "timestamp": timestamp,
        "prev": previous,
    }


def _group(value):
    return {
        "num_objects": 25,
        "num_valid_queries": 4,
        "num_same_view_queries": 3,
        "center_as_top1_rate": value,
        "same_view_center_as_top1_rate": value / 2,
        "position_top1_offset_histogram": {"(0,0)": 2, "(1,0)": 2},
        "shuffled_history_control": {"delta_center_mass": value / 4},
    }


def _view_group(value):
    return {
        "query_count": 2,
        "normalized_position_entropy": value,
        "position_top1_probability": value,
        "center_candidate_mass": value,
        "center_as_top1_rate": value,
        "position_top1_minus_top2_margin": value,
    }


def _pair(number, value=0.5):
    return {
        "pair_key": "p%d->c%d" % (number, number),
        "scene_token": "scene",
        "previous_sample_token": "p%d" % number,
        "current_sample_token": "c%d" % number,
        "delta_t": 0.5,
        "num_predictions": 300,
        "num_valid_queries": 4,
        "num_same_view_queries": 3,
        "groups": {name: _group(value) for name in GROUP_NAMES},
        "valid_view_groups": {name: _view_group(value) for name in VALID_VIEW_NAMES},
    }


def test_frame_plan_sorts_within_scene_and_never_crosses_boundary():
    infos = [
        _info("b1", "b", 20, "b0"),
        _info("a1", "a", 20, "a0"),
        _info("b0", "b", 10),
        _info("a0", "a", 10),
    ]
    frames, pairs = build_frame_and_pair_plan(infos)
    assert [frame.sample_token for frame in frames] == ["a0", "a1", "b0", "b1"]
    assert [pair.key for pair in pairs] == ["a0->a1", "b0->b1"]


def test_pair_requires_explicit_previous_token_link():
    _, pairs = build_frame_and_pair_plan(
        [_info("a0", "a", 10), _info("a2", "a", 30, "missing")]
    )
    assert pairs == []


def test_resume_starts_at_scene_first_frame_and_cache_rolls_one_item():
    frames, pairs = build_frame_and_pair_plan([
        _info("a0", "a", 10), _info("a1", "a", 20, "a0"),
        _info("a2", "a", 30, "a1"), _info("b0", "b", 10),
        _info("b1", "b", 20, "b0"),
    ])
    assert resume_frame_start(frames, pairs, {"a0->a1"}) == 0
    assert resume_frame_start(frames, pairs, {"a0->a1", "a1->a2"}) == 3
    old = object()
    new = object()
    assert cache_rollover(old, new) is new


def test_result_accumulation_has_required_statistics_and_histogram():
    accumulator = ResultAccumulator()
    accumulator.add(_pair(1, 0.25))
    accumulator.add(_pair(2, 0.75))
    summary = accumulator.summary()
    assert summary["num_pairs"] == 2
    assert summary["num_prediction_objects"] == 600
    stats = summary["top25"]["center_as_top1_rate"]
    assert set(stats) == {"count", "mean", "median", "std", "p25", "p75"}
    assert stats["mean"] == 0.5
    assert summary["all"]["position_top1_offset_histogram"]["(0,0)"] == 4


def test_json_and_gzip_resume_rebuild_prevent_duplicates(tmp_path):
    store = EvaluationStore(tmp_path, resume=False)
    assert store.append_pair(_pair(1))
    assert not store.append_pair(_pair(1))
    atomic_write_json(tmp_path / "summary.json", store.accumulator.summary())
    assert json.loads((tmp_path / "summary.json").read_text())["num_pairs"] == 1
    assert list(iter_jsonl_gz(tmp_path / "per_pair.jsonl.gz"))[0]["pair_key"] == "p1->c1"
    resumed = EvaluationStore(tmp_path, resume=True)
    assert len(resumed.accumulator.records) == 1
    assert not resumed.append_pair(_pair(1))
    assert resumed.append_pair(_pair(2))
    with gzip.open(str(tmp_path / "per_pair.jsonl.gz"), "rt") as handle:
        assert len([line for line in handle if line.strip()]) == 2


def test_error_logging_is_resumed_and_not_duplicated(tmp_path):
    store = EvaluationStore(tmp_path, resume=False)
    error = {"pair_key": "p->c", "scene_token": "s", "sample_token": "c",
             "exception": "bad projection"}
    assert store.append_error(error)
    assert not store.append_error(error)
    resumed = EvaluationStore(tmp_path, resume=True)
    assert "p->c" in resumed.processed_keys
    assert not resumed.append_error(error)
    assert len((tmp_path / "errors.jsonl").read_text().splitlines()) == 1


def test_empty_accumulator_is_finite_and_serializable(tmp_path):
    summary = ResultAccumulator().summary()
    assert summary["num_pairs"] == 0
    atomic_write_json(tmp_path / "empty.json", summary)
    loaded = json.loads((tmp_path / "empty.json").read_text())
    assert loaded["all"]["center_as_top1_rate"]["count"] == 0
    for group in GROUP_NAMES:
        for metric in loaded[group].values():
            if isinstance(metric, dict) and "mean" in metric:
                assert math.isfinite(metric["mean"])


def test_nonfinite_values_are_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        descriptive_statistics([1.0, float("nan")])
