import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis.care3d_p2a2_r1_confirmation as confirmation
from analysis.care3d_p2a2_r1_arbiter import FrozenIdentityArbiter
from analysis.care3d_p2a2_r1_confirmation import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    COHORT_CLEAN_STATUS,
    COHORT_STOP_STATUS,
    CONFIRMED_STATUS,
    EXPECTED_OFFICIAL_VAL_SCENES,
    FROZEN_ARTIFACT_SHA256,
    INSTANCE_CLUSTER_COLUMNS,
    NOT_CONFIRMED_STATUS,
    SCENE_CLUSTER_COLUMNS,
    TAU_DEFER,
    TAU_PREFERENCE,
    FrozenConfirmatoryArbiter,
    attach_offline_outcomes,
    confirmatory_point_metrics,
    default_discipline,
    evaluate_confirmatory_gate,
    heldout_lineage_audit,
    load_frozen_arbiter,
    paired_confirmatory_bootstrap,
    reconstruct_full_population,
    run_frozen_runtime,
    sha256_file,
)
from analysis.care3d_p2a2_r1_features import MODEL_FEATURE_COLUMNS
from analysis.care3d_p2a_association import PROTOCOLS
from scripts.export_care3d_p2a2_r1_confirmation import (
    CONFIRMATORY_ANN_FILE,
    CONFIRMATORY_CONFIG,
    P2A0_CONFIG_COLUMNS,
    SOURCE_AUDIT_FLAGS,
    export_scene,
    marker_valid,
    parse_args,
    parse_named_manifest,
    reject_locked_data_path,
    validate_confirmatory_test_ann_file,
)
from scripts.analyze_care3d_p2a2_r1_confirmation import load_completed_population


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "reports/care3d/p2a2_r1_identity_arbiter/frozen_arbiter.json"


def _official_tokens(count=EXPECTED_OFFICIAL_VAL_SCENES):
    return [f"official-{index:03d}" for index in range(count)]


def _official_manifest(tokens=None):
    values = _official_tokens() if tokens is None else list(tokens)
    return pd.DataFrame({"scene_token": values, "split": "official_val"})


def _features(rows: int, value: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        np.full((rows, len(MODEL_FEATURE_COLUMNS)), value),
        columns=MODEL_FEATURE_COLUMNS,
    )


def _constant_frozen(preference: float, ambiguity: float) -> FrozenConfirmatoryArbiter:
    width = len(MODEL_FEATURE_COLUMNS)

    def head(probability):
        return {
            "scaler_mean": [0.0] * width,
            "scaler_scale": [1.0] * width,
            "coef": [0.0] * width,
            "intercept": [float(np.log(probability / (1.0 - probability)))],
        }

    runtime = FrozenIdentityArbiter(
        head(preference), head(ambiguity), TAU_PREFERENCE, TAU_DEFER
    )
    return FrozenConfirmatoryArbiter(Path("frozen.json"), "x", {}, runtime)


def _passing_gate_inputs():
    metrics = pd.DataFrame([
        {
            "protocol": protocol,
            "arbiter_wrong_rate": 0.10,
            "arbiter_unmatched_rate": 0.01,
            "delta_exact": 0.01,
            "delta_wrong": -0.01,
            "agreement_modified_count": 0,
        }
        for protocol in PROTOCOLS
    ])
    summaries = []
    for protocol in PROTOCOLS:
        summaries.extend([
            {"protocol": protocol, "metric": "delta_exact", "ci_low": 0.001, "ci_high": 0.02},
            {"protocol": protocol, "metric": "delta_wrong", "ci_low": -0.02, "ci_high": -0.001},
        ])
    return metrics, pd.DataFrame(summaries), default_discipline()


def _outcome_frame() -> pd.DataFrame:
    decisions = pd.DataFrame({
        "p_lineage": [np.nan, 0.9, 0.8, 0.2, 0.4, 0.3],
        "p_both_wrong": [np.nan, 0.1, 0.1, 0.1, 0.9, 0.9],
        "arbiter_selected_query": [1, 20, 31, 41, -1, -1],
        "arbiter_decision": [
            "AGREEMENT", "LINEAGE", "LINEAGE", "LINEAGE", "DEFER", "DEFER"
        ],
    })
    output = attach_offline_outcomes(
        decisions,
        [1, 2, 3, 4, 5, 6],
        [1, 20, 30, 4, 5, 60],
    )
    output["p2a0_selected_query"] = [1, 2, 3, 4, 5, 6]
    output["lineage_child_query"] = [1, 20, 30, 40, 50, 60]
    output["scene_token"] = ["s1", "s1", "s2", "s2", "s3", "s3"]
    output["instance_token"] = ["i1", "i2", "i3", "i4", "i5", "i6"]
    return output


def test_frozen_artifact_sha_is_exact_and_mismatch_is_hard_failure(tmp_path):
    assert sha256_file(ARTIFACT) == FROZEN_ARTIFACT_SHA256
    frozen = load_frozen_arbiter(ARTIFACT)
    assert frozen.artifact_sha256 == FROZEN_ARTIFACT_SHA256
    changed = tmp_path / "frozen.json"
    changed.write_bytes(ARTIFACT.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        load_frozen_arbiter(changed)


def test_frozen_thresholds_must_be_052_and_088(tmp_path):
    value = json.loads(ARTIFACT.read_text())
    value["tau_preference"] = 0.53
    changed = tmp_path / "frozen.json"
    changed.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="tau_preference"):
        load_frozen_arbiter(changed, expected_sha256=sha256_file(changed))
    value["tau_preference"] = 0.52
    value["tau_defer"] = 0.87
    changed.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="tau_defer"):
        load_frozen_arbiter(changed, expected_sha256=sha256_file(changed))


def test_formal_config_is_mechanism_val_and_parses_val_ann_file():
    from mmcv import Config

    assert CONFIRMATORY_CONFIG.name == "stream_petr_r50_90e_mechanism_val.py"
    config = Config.fromfile(str(CONFIRMATORY_CONFIG))
    observed = validate_confirmatory_test_ann_file(config)
    assert Path(observed) == CONFIRMATORY_ANN_FILE.resolve()
    assert Path(config.data.test.ann_file).name == "nuscenes2d_temporal_infos_val.pkl"


def test_formal_config_guard_hard_fails_train_ann_file():
    from mmcv import ConfigDict

    config = ConfigDict({
        "data": {
            "test": {
                "ann_file": str(
                    ROOT / "data/nuscenes/nuscenes2d_temporal_infos_train.pkl"
                )
            }
        }
    })
    with pytest.raises(RuntimeError, match="refuses the nuScenes train"):
        validate_confirmatory_test_ann_file(config)


def test_confirmation_module_has_no_fit_or_search_interface():
    source = inspect.getsource(confirmation)
    assert "sklearn" not in source
    assert ".fit(" not in source
    assert not hasattr(confirmation, "select_thresholds")
    assert not hasattr(confirmation, "make_binary_head")


def test_agreement_is_exact_bypass_even_with_nonfinite_features():
    frozen = _constant_frozen(0.99, 0.99)
    features = _features(2)
    features.loc[:, :] = np.nan
    result = run_frozen_runtime(features, [7, 9], [7, 9], frozen)
    assert result.arbiter_selected_query.tolist() == [7, 9]
    assert result.arbiter_decision.tolist() == ["AGREEMENT", "AGREEMENT"]
    assert result[["p_lineage", "p_both_wrong"]].isna().all().all()


def test_defer_precedes_preference_at_frozen_thresholds():
    frozen = _constant_frozen(0.99, 0.90)
    result = run_frozen_runtime(_features(1), [4], [8], frozen)
    assert result.arbiter_selected_query.tolist() == [-1]
    assert result.arbiter_decision.tolist() == ["DEFER"]


def test_protocol_is_not_a_model_feature_and_is_rejected_at_runtime():
    assert "protocol" not in MODEL_FEATURE_COLUMNS
    frame = _features(1).assign(protocol="blur_back")
    with pytest.raises(RuntimeError, match="forbidden arbiter runtime"):
        run_frozen_runtime(frame, [1], [2], _constant_frozen(0.2, 0.2))


@pytest.mark.parametrize(
    "forbidden",
    ["gt", "oracle_query_index", "clean_future", "target_clean_query_index", "severity_id"],
)
def test_gt_oracle_clean_future_and_severity_are_forbidden_runtime_inputs(forbidden):
    frame = _features(1)
    frame[forbidden] = 0
    with pytest.raises(RuntimeError, match="forbidden arbiter runtime"):
        run_frozen_runtime(frame, [1], [2], _constant_frozen(0.2, 0.2))


def test_full_population_outcomes_and_repairs_are_exhaustive():
    frame = _outcome_frame()
    assert ((frame.p2a0_exact + frame.p2a0_wrong + frame.p2a0_unmatched) == 1).all()
    assert ((frame.arbiter_exact + frame.arbiter_wrong + frame.arbiter_unmatched) == 1).all()
    metrics = confirmatory_point_metrics(frame)
    assert metrics["agreement_rows"] + metrics["disagreement_rows"] == len(frame)
    assert metrics["wrong_repaired_count"] == 1
    assert metrics["correct_broken_to_wrong_count"] == 1
    assert metrics["correct_broken_to_unmatched_count"] == 1
    assert metrics["wrong_to_unmatched_count"] == 1
    assert metrics["wrong_remaining_wrong_count"] == 1


def test_full_population_reconstruction_keeps_agreement_and_disagreement():
    base = pd.DataFrame({
        "scene_token": ["s", "s"],
        "instance_token": ["agree", "disagree"],
        "anchor_frame_idx": [2, 2],
        "target_frame_idx": [3, 3],
        "protocol": ["blur_back", "blur_back"],
        "p2a0_selected_query": [7, 4],
        "lineage_child_query": [7, 8],
        "oracle_query_index": [7, 8],
    })
    evidence = pd.concat([
        base.loc[[1], list(confirmation.KEY_COLUMNS)].reset_index(drop=True),
        _features(1),
    ], axis=1)
    full = reconstruct_full_population(base, evidence)
    assert len(full) == 2
    assert full.loc[0, list(MODEL_FEATURE_COLUMNS)].isna().all()
    assert np.isfinite(full.loc[1, list(MODEL_FEATURE_COLUMNS)].to_numpy(float)).all()


def test_bootstrap_constants_and_cluster_keys_are_frozen():
    frame = _outcome_frame()
    scene, _ = paired_confirmatory_bootstrap(
        frame,
        cluster_columns=SCENE_CLUSTER_COLUMNS,
        cluster_population=("s0", "s1", "s2", "s3"),
    )
    instance, _ = paired_confirmatory_bootstrap(
        frame, cluster_columns=INSTANCE_CLUSTER_COLUMNS
    )
    assert (scene.replicates == BOOTSTRAP_REPLICATES).all()
    assert (scene.seed == BOOTSTRAP_SEED).all()
    assert (instance.replicates == 5000).all()
    with pytest.raises(ValueError, match="cluster key"):
        paired_confirmatory_bootstrap(frame, cluster_columns=("instance_token",))


@pytest.mark.parametrize(
    "column,value,passes",
    [
        ("arbiter_wrong_rate", 0.10, True),
        ("arbiter_wrong_rate", 0.1001, False),
        ("arbiter_unmatched_rate", 0.01, True),
        ("arbiter_unmatched_rate", 0.0101, False),
    ],
)
def test_gate_rate_boundaries(column, value, passes):
    metrics, summary, discipline = _passing_gate_inputs()
    metrics.loc[metrics.protocol == "blur_back", column] = value
    gate, status = evaluate_confirmatory_gate(metrics, summary, summary, discipline)
    assert bool(gate.loc[gate.protocol == "blur_back", "protocol_pass"].item()) is passes
    assert (status == CONFIRMED_STATUS) is passes


def test_gate_zero_ci_boundaries_fail_strictly():
    metrics, scene, discipline = _passing_gate_inputs()
    instance = scene.copy()
    scene.loc[
        (scene.protocol == "blur_back") & (scene.metric == "delta_exact"), "ci_low"
    ] = 0.0
    instance.loc[
        (instance.protocol == "dark_back") & (instance.metric == "delta_wrong"), "ci_high"
    ] = 0.0
    gate, status = evaluate_confirmatory_gate(metrics, scene, instance, discipline)
    assert not gate.loc[gate.protocol == "blur_back", "C4_scene_delta_exact_ci_low_positive"].item()
    assert not gate.loc[gate.protocol == "dark_back", "C8_instance_delta_wrong_ci_high_negative"].item()
    assert status == NOT_CONFIRMED_STATUS


def test_crash_failure_makes_overall_not_confirmed():
    metrics, summary, discipline = _passing_gate_inputs()
    metrics.loc[metrics.protocol == "crash_back", "delta_exact"] = 0.0
    gate, status = evaluate_confirmatory_gate(metrics, summary, summary, discipline)
    assert not gate.loc[gate.protocol == "crash_back", "protocol_pass"].item()
    assert status == NOT_CONFIRMED_STATUS


def test_discipline_c10_requires_exact_frozen_contract():
    metrics, summary, discipline = _passing_gate_inputs()
    for name in (*confirmation.DISCIPLINE_TRUE, *confirmation.DISCIPLINE_FALSE):
        changed = dict(discipline)
        changed[name] = not changed[name]
        gate, status = evaluate_confirmatory_gate(metrics, summary, summary, changed)
        assert not gate.C10_frozen_discipline.any()
        assert status == NOT_CONFIRMED_STATUS


def _valid_marker_value():
    return {
        "complete": True,
        "schema_version": 1,
        "split": "official_val",
        "scene_token": "scene-a",
        "source_rows_sha256": "a" * 64,
        "rows_sha256": "b" * 64,
        "frozen_artifact_sha256": FROZEN_ARTIFACT_SHA256,
        "tau_preference": 0.52,
        "tau_defer": 0.88,
        **default_discipline(),
    }


@pytest.mark.parametrize(
    "field,tampered",
    [
        ("tau_preference", 0.51),
        ("tau_defer", 0.87),
        ("model_refit", True),
        ("threshold_search", True),
        ("feature_search", True),
        ("model_search", True),
        ("calibration_search", True),
        ("protocol_used_as_feature", True),
        ("protocol_specific_threshold", True),
        ("probe_val_read", True),
        ("probe_test_read", True),
    ],
)
def test_marker_full_discipline_contract_rejects_tampering(
    tmp_path, field, tampered
):
    value = _valid_marker_value()
    path = tmp_path / "scene-a.complete.json"
    path.write_text(json.dumps(value))
    assert marker_valid(path, "scene-a", "a" * 64)
    value[field] = tampered
    path.write_text(json.dumps(value))
    assert not marker_valid(path, "scene-a", "a" * 64)


def test_probe_val_raw_and_probe_test_are_locked():
    with pytest.raises(RuntimeError, match="raw probe_val"):
        reject_locked_data_path(Path("incremental/probe_val/scene.rows.csv"))
    with pytest.raises(RuntimeError, match="probe_test"):
        reject_locked_data_path(Path("anything/probe_test/scene.rows.csv"))
    with pytest.raises(Exception):
        parse_named_manifest("probe_test_manifest=/tmp/manifest.csv")
    with pytest.raises(SystemExit):
        parse_args(["--export", "--input-dir", "/tmp/probe_test"])


def _audit(official, metadata_tokens=None, sources=None, required=("history",)):
    return heldout_lineage_audit(
        official,
        tuple(_official_tokens() if metadata_tokens is None else metadata_tokens),
        {"history": pd.DataFrame({"scene_token": ["historical-train"]})}
        if sources is None else sources,
        required_sources=required,
        official_val_manifest_sha256="a" * 64,
    )


def test_official_val_manifest_requires_split():
    with pytest.raises(RuntimeError, match=COHORT_STOP_STATUS):
        _audit(pd.DataFrame({"scene_token": _official_tokens()}))


@pytest.mark.parametrize("count", [149, 151])
def test_official_val_manifest_requires_exactly_150_scenes(count):
    with pytest.raises(RuntimeError, match=COHORT_STOP_STATUS):
        _audit(_official_manifest(_official_tokens(count)))


def test_official_val_manifest_token_set_must_match_official_metadata():
    changed = _official_tokens()
    changed[-1] = "replacement-scene"
    with pytest.raises(RuntimeError, match=COHORT_STOP_STATUS):
        _audit(_official_manifest(changed))


def test_exact_official_val_identity_set_passes_and_records_provenance():
    cohort, audit = _audit(_official_manifest())
    assert len(cohort) == EXPECTED_OFFICIAL_VAL_SCENES
    assert audit["status"] == COHORT_CLEAN_STATUS
    assert audit["official_val_expected_scenes"] == 150
    assert audit["official_val_candidate_scenes"] == 150
    assert audit["official_val_identity_exact_match"] is True
    assert audit["official_val_manifest_sha256"] == "a" * 64
    assert audit["excluded_used_scenes"] == 0
    assert audit["heldout_intersection_count"] == 0


def test_heldout_lineage_intersection_is_removed_and_audited():
    tokens = _official_tokens()
    sources = {
        "train": pd.DataFrame({"scene_token": [tokens[0], "historical-train"]}),
        "val": pd.DataFrame({"scene_token": [tokens[1], "historical-val"]}),
        "redesign": pd.DataFrame({"scene_token": ["historical-redesign"]}),
    }
    cohort, audit = _audit(
        _official_manifest(), sources=sources, required=("train", "val", "redesign")
    )
    assert cohort.scene_token.tolist() == tokens[2:]
    assert audit["excluded_used_scenes"] == 2
    assert audit["confirmatory_scenes"] == 148
    assert audit["heldout_intersection_count"] == 0


def test_incomplete_lineage_registry_stops_confirmation():
    cohort, audit = heldout_lineage_audit(
        _official_manifest(),
        tuple(_official_tokens()),
        {"train": pd.DataFrame({"scene_token": ["x"]})},
        required_sources=("train", "missing_tuning_manifest"),
        official_val_manifest_sha256="a" * 64,
    )
    assert cohort.empty
    assert audit["status"] == COHORT_STOP_STATUS


def test_cohort_selection_rejects_outcome_columns():
    official = _official_manifest().assign(wrong=0)
    with pytest.raises(RuntimeError, match="contains outcomes"):
        _audit(official)


def test_export_does_not_rewrite_frozen_artifact_and_resume_is_noop(tmp_path):
    source = pd.concat([
        pd.concat([_features(1), pd.DataFrame({
            "scene_token": ["scene-a"],
            "instance_token": ["instance-a"],
            "anchor_frame_idx": [2],
            "target_frame_idx": [3],
            "protocol": [protocol],
            "p2a0_selected_query": [7],
            "lineage_child_query": [7],
            "oracle_query_index": [7],
            **{name: [value] for name, value in SOURCE_AUDIT_FLAGS.items()},
            **{name: [value] for name, value in P2A0_CONFIG_COLUMNS.items()},
        })], axis=1)
        for protocol in PROTOCOLS
    ], ignore_index=True)
    source_path = tmp_path / "scene-a.features.csv"
    source.to_csv(source_path, index=False)
    artifact_copy = tmp_path / "frozen_arbiter.json"
    artifact_copy.write_bytes(ARTIFACT.read_bytes())
    before = artifact_copy.read_bytes()
    output_dir = tmp_path / "incremental" / "official_val"
    assert export_scene(
        "scene-a", source_path, artifact_path=artifact_copy, output_dir=output_dir
    ) is True
    assert artifact_copy.read_bytes() == before
    assert export_scene(
        "scene-a", source_path, artifact_path=artifact_copy, output_dir=output_dir
    ) is False
    assert artifact_copy.read_bytes() == before

    report = tmp_path / "report"
    report.mkdir()
    pd.DataFrame({
        "scene_token": ["scene-a"], "split": ["official_val"]
    }).to_csv(report / "confirmatory_manifest.csv", index=False)
    (report / "heldout_lineage_audit.json").write_text(json.dumps({
        "status": COHORT_CLEAN_STATUS,
        "heldout_intersection_count": 0,
        "official_val_expected_scenes": 150,
        "official_val_candidate_scenes": 150,
        "official_val_identity_exact_match": True,
        "official_val_manifest_sha256": "c" * 64,
        "probe_val_read": False,
        "probe_test_read": False,
    }))
    (report / "progress_manifest.json").write_text(json.dumps({
        "status": "CONFIRMATORY_EXTRACTION_COMPLETE_ANALYSIS_ELIGIBLE",
        "completed_scenes": 1,
        "frozen_artifact_sha256": FROZEN_ARTIFACT_SHA256,
        "probe_val_read": False,
        "probe_test_read": False,
    }))
    _manifest, population, _audit, verified = load_completed_population(
        report=report, incremental=output_dir
    )
    assert len(population) == 3
    assert verified == default_discipline()

    marker_path = output_dir / "scene-a.complete.json"
    marker = json.loads(marker_path.read_text())
    marker["threshold_search"] = True
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(RuntimeError, match="invalid confirmatory scene marker"):
        load_completed_population(report=report, incremental=output_dir)
